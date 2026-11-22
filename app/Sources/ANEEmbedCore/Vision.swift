import CoreGraphics
import CoreML
import Foundation
import ImageIO

/// manifest.json of the vision assets, written by convert_gemma2_vision_v2.py export into
/// <Application Support>/ANEEmbed/embeddinggemma2-vision-w16 (the text side stays in embeddinggemma2-w16).
public struct VisionManifest: Decodable {
    /// The text model the soft tokens feed: images are embedded in its vector space.
    public let model_tag: String
    public let patch_size: Int
    public let pooling: Int
    public let max_patches: Int
    public let width: Int
    public let out_dim: Int
    public let head_dim: Int
    public let table_size: Int
    public let inv_freq: [Float]
    public let token_ids: [String: Int]
    public let packages: [String: String]
    public let default_layout: String
}

/// One image as the model's processor (transformers Gemma4ImageProcessor) prepares it: real patches
/// only, [count, patch_size * patch_size * 3] in raster order; padding up to max_patches is implicit zeros.
public struct PreparedImage {
    public let patches: [Float]
    public let columns: Int
    public let rows: Int
    public var count: Int { columns * rows }
}

/// EmbeddingGemma 2's vision encoder on the ANE (ANE Embed 0.2). Two layouts of the same encoder ship
/// side by side: "v1" traces the model's own layers ([1, P, C] tensors), "v2" is the rewrite in Apple's
/// ANE layout ([1, C, 1, P]), 1.40x faster. A request may pick one; the default is the manifest's.
public final class Vision {
    public let root: URL
    public let manifest: VisionManifest
    /// The two position tables (x, then y), each table_size x width, float32.
    let posTable: Data
    let queue = DispatchQueue(label: "ane-embed.vision", qos: .userInitiated)
    private var loading: [String: Task<MLModel, Error>] = [:]
    private var loaded: Set<String> = []
    private let lock = NSLock()
    /// Counters for /health: images done, and seconds in CPU preparation and in the encoder's predict.
    public private(set) var images = 0
    public private(set) var prepSeconds = 0.0
    public private(set) var predictSeconds = 0.0

    public static var defaultRoot: URL { Assets.root(named: "embeddinggemma2-vision-w16") }

    public init(root: URL = Vision.defaultRoot) throws {
        self.root = root
        manifest = try JSONDecoder().decode(VisionManifest.self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
        posTable = try Data(contentsOf: root.appendingPathComponent("pos_table.f32"), options: .alwaysMapped)
        let m = manifest
        guard posTable.count == 2 * m.table_size * m.width * 4 else { throw EmbedError.assets("pos_table.f32 size does not match manifest.json") }
        guard m.inv_freq.count * 4 == m.head_dim, ["bos", "boi", "image", "eoi", "eos"].allSatisfy({ m.token_ids[$0] != nil }) else {
            throw EmbedError.assets("vision manifest: inv_freq or token_ids incomplete")
        }
    }

    public var layouts: [String] { manifest.packages.keys.sorted() }
    public var loadedLayouts: [String] { lock.withLock { loaded.sorted() } }

    /// The layout's model: compiled once into <root>/compiled/<layout>.mlmodelc, loaded on first use
    /// (the v2 package has 13,721 ops and takes a minute or two to compile for the ANE). Concurrent
    /// first requests share one load.
    public func model(_ layout: String) async throws -> MLModel {
        let task: Task<MLModel, Error> = lock.withLock {
            if let t = loading[layout] { return t }
            let t = Task { try await self.load(layout) }
            loading[layout] = t
            return t
        }
        do { return try await task.value } catch {
            lock.withLock { loading[layout] = nil }
            throw error
        }
    }

    private func load(_ layout: String) async throws -> MLModel {
        guard let rel = manifest.packages[layout] else {
            throw EmbedError.input("vision layout \(layout) not available; have \(layouts.joined(separator: ", "))")
        }
        let cache = root.appendingPathComponent("compiled")
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let target = cache.appendingPathComponent("\(layout).mlmodelc")
        if !FileManager.default.fileExists(atPath: target.path) {
            let temp = try await MLModel.compileModel(at: root.appendingPathComponent(rel))
            try? FileManager.default.removeItem(at: target)
            try FileManager.default.moveItem(at: temp, to: target)
        }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let m = try MLModel(contentsOf: target, configuration: config)
        lock.withLock { _ = loaded.insert(layout) }
        return m
    }

    /// The input embeddings of one image, ready for the text core: [bos, boi, n soft tokens, eoi, eos],
    /// each row `hidden` float16; text rows come from the scaled token table, image rows from the encoder.
    func sequence(for data: Data, layout: String, assets: Assets) async throws -> [Float16] {
        let model = try await model(layout)
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    let t0 = Date()
                    let image = try prepareImage(data, manifest: self.manifest)
                    let input = try self.encoderInput(image, layout: layout)
                    let t1 = Date()
                    let out = try model.prediction(from: input)
                    let t2 = Date()
                    guard let soft = out.featureValue(for: "soft")?.multiArrayValue else { throw EmbedError.runtime("vision: no soft output") }
                    let rows = try self.assemble(soft: soft, image: image, layout: layout, assets: assets)
                    self.lock.withLock {
                        self.images += 1
                        self.prepSeconds += t1.timeIntervalSince(t0) + Date().timeIntervalSince(t2)
                        self.predictSeconds += t2.timeIntervalSince(t1)
                    }
                    continuation.resume(returning: rows)
                } catch { continuation.resume(throwing: error) }
            }
        }
    }

    /// The encoder's inputs, built from the patch positions alone (as side_inputs in convert_gemma2_vision.py):
    /// position embeddings (x table row + y table row), axial rotary cos/sin, the valid-patch mask and the
    /// 3 x 3 average-pool matrix. v1 takes [1, P, C]-shaped arrays, v2 channels-first [1, C, 1, P].
    func encoderInput(_ image: PreparedImage, layout: String) throws -> MLFeatureProvider {
        let m = manifest
        let P = m.max_patches, C = m.width, D = m.head_dim, k = m.pooling
        let T = P / (k * k), n = image.count, freqs = m.inv_freq.count
        let v2 = layout == "v2"
        func array(_ shape: [Int]) throws -> MLMultiArray {
            let a = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float16)
            a.withUnsafeMutableBytes { p, _ in _ = memset(p.baseAddress!, 0, p.count) }
            return a
        }
        let patches = try array(v2 ? [1, C, 1, P] : [1, P, C])
        let pos = try array(v2 ? [1, C, 1, P] : [1, P, C])
        let cos = try array(v2 ? [1, D, 1, P] : [1, P, D])
        let sin = try array(v2 ? [1, D, 1, P] : [1, P, D])
        let valid = try array(v2 ? [1, 1, 1, P] : [1, P])
        let pool = try array(v2 ? [1, P, T] : [1, T, P])
        // element (p, c) of a [P, C] quantity: v1 stores p * C + c, v2 (channel-major) c * P + p
        let at: (Int, Int, Int) -> Int = v2 ? { p, c, _ in c * P + p } : { p, c, width in p * width + c }
        let ninth = Float16(1.0 / Float(k * k))
        posTable.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float.self)
            patches.withUnsafeMutableBytes { pp, _ in
                pos.withUnsafeMutableBytes { qp, _ in
                    cos.withUnsafeMutableBytes { cp, _ in
                        sin.withUnsafeMutableBytes { sp, _ in
                            valid.withUnsafeMutableBytes { vp, _ in
                                pool.withUnsafeMutableBytes { op, _ in
                                    let px = pp.bindMemory(to: Float16.self), ps = qp.bindMemory(to: Float16.self)
                                    let pc = cp.bindMemory(to: Float16.self), pn = sp.bindMemory(to: Float16.self)
                                    let pv = vp.bindMemory(to: Float16.self), po = op.bindMemory(to: Float16.self)
                                    for p in 0..<P {
                                        // padding patches sit at position (-1, -1), as the processor pads them
                                        let x = p < n ? p % image.columns : -1, y = p < n ? p / image.columns : -1
                                        for j in 0..<D {
                                            let angle = Float(j < D / 2 ? x : y) * m.inv_freq[j % freqs]
                                            pc[at(p, j, D)] = Float16(Foundation.cos(angle))
                                            pn[at(p, j, D)] = Float16(Foundation.sin(angle))
                                        }
                                        guard p < n else { continue }
                                        pv[p] = 1
                                        let rowX = x * C, rowY = (m.table_size + y) * C, src = p * C
                                        for c in 0..<C {
                                            px[at(p, c, C)] = Float16(image.patches[src + c])
                                            ps[at(p, c, C)] = Float16(table[rowX + c] + table[rowY + c])
                                        }
                                        let t = x / k + (image.columns / k) * (y / k)
                                        po[v2 ? p * T + t : t * P + p] = ninth
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        let names = ["patches": patches, "pos": pos, "cos": cos, "sin": sin, "valid": valid, (v2 ? "poolT" : "pool"): pool]
        return try MLDictionaryFeatureProvider(dictionary: names)
    }

    /// The first n soft tokens are the image's (n = (rows / 3) * (columns / 3)); each becomes one row.
    func assemble(soft: MLMultiArray, image: PreparedImage, layout: String, assets: Assets) throws -> [Float16] {
        let hidden = assets.manifest.hidden, k = manifest.pooling
        guard manifest.out_dim == hidden else { throw EmbedError.assets("vision out_dim \(manifest.out_dim) != text hidden \(hidden)") }
        let n = (image.rows / k) * (image.columns / k)
        let ids = manifest.token_ids
        var rows = [Float16](repeating: 0, count: (n + 4) * hidden)
        let strides = soft.strides.map(\.intValue)
        assets.embedScaled.withUnsafeBytes { raw in
            let table = raw.bindMemory(to: Float16.self)
            for (slot, id) in [(0, ids["bos"]!), (1, ids["boi"]!), (n + 2, ids["eoi"]!), (n + 3, ids["eos"]!)] {
                for d in 0..<hidden { rows[slot * hidden + d] = table[id * hidden + d] }
            }
        }
        soft.withUnsafeBytes { raw in
            let s = raw.bindMemory(to: Float16.self)
            for t in 0..<n {
                for d in 0..<hidden {   // v2 output [1, 512, T]; v1 [1, T, 512]
                    let index = layout == "v2" ? d * strides[1] + t * strides[2] : t * strides[1] + d * strides[2]
                    rows[(t + 2) * hidden + d] = s[index]
                }
            }
        }
        return rows
    }
}

// MARK: - image preparation (the processor, in Swift)

/// decode -> RGB uint8 -> the largest size that keeps the aspect ratio, is a multiple of 48 (16 x 3) and
/// stays within max_patches -> antialiased bicubic resize -> / 255 -> 16 x 16 patches, each
/// [row][column][channel] -> raster order (positions x = column, y = row).
public func prepareImage(_ data: Data, manifest m: VisionManifest) throws -> PreparedImage {
    let (rgb, width, height) = try decodeRGB(data)
    let (h2, w2) = try targetSize(height: height, width: width, patch: m.patch_size, maxPatches: m.max_patches, pool: m.pooling)
    let img = (h2 == height && w2 == width) ? rgb : resizeBicubic(rgb, width: width, height: height, toWidth: w2, toHeight: h2)
    let ps = m.patch_size, cols = w2 / ps, rows = h2 / ps, dim = ps * ps * 3
    var patches = [Float](repeating: 0, count: cols * rows * dim)
    let scale = Float(0.00392156862745098)       // the processor's rescale_factor, applied in float32
    for py in 0..<rows {
        for px in 0..<cols {
            let base = (py * cols + px) * dim
            for r in 0..<ps {
                let line = ((py * ps + r) * w2 + px * ps) * 3
                for c in 0..<(ps * 3) { patches[base + r * ps * 3 + c] = Float(img[line + c]) * scale }
            }
        }
    }
    return PreparedImage(patches: patches, columns: cols, rows: rows)
}

/// get_aspect_ratio_preserving_size from transformers' image_processing_gemma4.py.
func targetSize(height: Int, width: Int, patch: Int, maxPatches: Int, pool: Int) throws -> (Int, Int) {
    let targetPx = Double(maxPatches * patch * patch)
    let factor = (targetPx / Double(height * width)).squareRoot()
    let side = pool * patch
    var th = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
    var tw = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
    guard th > 0 || tw > 0 else { throw EmbedError.input("image \(width)x\(height) would resize to 0 x 0") }
    let maxSide = (maxPatches / (pool * pool)) * side
    if th == 0 {
        th = side; tw = min(Int((Double(width) / Double(height)).rounded(.down)) * side, maxSide)
    } else if tw == 0 {
        tw = side; th = min(Int((Double(height) / Double(width)).rounded(.down)) * side, maxSide)
    }
    guard Double(th * tw) <= targetPx else { throw EmbedError.input("image \(width)x\(height) does not fit \(maxPatches) patches") }
    return (th, tw)
}

/// RGB uint8 as stored in the file, like PIL's convert("RGB"): drawn in the image's own RGB color
/// space, so no color management changes the values. Alpha is dropped (opaque images are exact).
func decodeRGB(_ data: Data) throws -> ([UInt8], Int, Int) {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw EmbedError.input("not a decodable image") }
    let w = image.width, h = image.height
    guard w > 0, h > 0, w * h <= 64_000_000 else { throw EmbedError.input("image size \(w)x\(h) not supported") }
    let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    let drawn: Bool = rgba.withUnsafeMutableBytes { buffer in
        guard let ctx = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                  space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        ctx.interpolationQuality = .none
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return true
    }
    guard drawn else { throw EmbedError.input("could not draw the image") }
    var rgb = [UInt8](repeating: 0, count: w * h * 3)
    for i in 0..<(w * h) {
        rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2]
    }
    return (rgb, w, h)
}

/// Pillow's BICUBIC resample (a = -0.5) on uint8, which torchvision's antialiased uint8 resize follows:
/// per axis, filter support scaled by the downscale factor, weights normalised and turned into 22-bit
/// fixed point; horizontal pass, then vertical, each rounded and clipped to uint8.
func resizeBicubic(_ src: [UInt8], width: Int, height: Int, toWidth w2: Int, toHeight h2: Int) -> [UInt8] {
    let horizontal = w2 == width ? src : pass(src, lines: height, inSize: width, outSize: w2, horizontal: true)
    return h2 == height ? horizontal : pass(horizontal, lines: w2, inSize: height, outSize: h2, horizontal: false)
}

private let precisionBits = 22

private func bicubic(_ x: Double) -> Double {
    let a = -0.5, x = abs(x)
    if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
    if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
    return 0
}

/// Fixed-point weights per output position (Pillow's precompute_coeffs + normalize_coeffs_8bpc).
private func coefficients(inSize: Int, outSize: Int) -> (bounds: [(Int, Int)], weights: [Int], ksize: Int) {
    let scale = Double(inSize) / Double(outSize)
    let filterScale = max(scale, 1.0)
    let support = 2.0 * filterScale
    let ksize = Int(support.rounded(.up)) * 2 + 1
    var bounds: [(Int, Int)] = []
    var weights = [Int](repeating: 0, count: outSize * ksize)
    var w = [Double](repeating: 0, count: ksize)
    for xx in 0..<outSize {
        let center = (Double(xx) + 0.5) * scale
        var xmin = Int(center - support + 0.5); if xmin < 0 { xmin = 0 }
        var xmax = Int(center + support + 0.5); if xmax > inSize { xmax = inSize }
        xmax -= xmin
        var total = 0.0
        for x in 0..<xmax { w[x] = bicubic((Double(x + xmin) - center + 0.5) / filterScale); total += w[x] }
        for x in 0..<xmax {
            let v = total != 0 ? w[x] / total : w[x]
            weights[xx * ksize + x] = Int(v < 0 ? -0.5 + v * Double(1 << precisionBits) : 0.5 + v * Double(1 << precisionBits))
        }
        bounds.append((xmin, xmax))
    }
    return (bounds, weights, ksize)
}

/// One resampling pass over RGB uint8. horizontal: `lines` rows of inSize pixels -> outSize pixels.
/// vertical: inSize rows of `lines` pixels -> outSize rows.
private func pass(_ src: [UInt8], lines: Int, inSize: Int, outSize: Int, horizontal: Bool) -> [UInt8] {
    let (bounds, weights, ksize) = coefficients(inSize: inSize, outSize: outSize)
    var out = [UInt8](repeating: 0, count: lines * outSize * 3)
    let half = 1 << (precisionBits - 1)
    func clip(_ v: Int) -> UInt8 { let s = v >> precisionBits; return s < 0 ? 0 : (s > 255 ? 255 : UInt8(s)) }
    src.withUnsafeBufferPointer { s in
        out.withUnsafeMutableBufferPointer { o in
            for line in 0..<lines {
                for xx in 0..<outSize {
                    let (xmin, count) = bounds[xx]
                    var r = half, g = half, b = half
                    for x in 0..<count {
                        let k = weights[xx * ksize + x]
                        let i = horizontal ? (line * inSize + xmin + x) * 3 : ((xmin + x) * lines + line) * 3
                        r += Int(s[i]) * k; g += Int(s[i + 1]) * k; b += Int(s[i + 2]) * k
                    }
                    let j = horizontal ? (line * outSize + xx) * 3 : (xx * lines + line) * 3
                    o[j] = clip(r); o[j + 1] = clip(g); o[j + 2] = clip(b)
                }
            }
        }
    }
    return out
}
