import ANEEmbedCore
import ANEEmbedHTTP
import Foundation

/// ane-embed embed [--workers N] TEXT...      print {"embeddings": [...]} for the texts
/// ane-embed serve [--port 11435] [--host 127.0.0.1] [--allow-network] [--workers N]
let usage = "usage: ane-embed ane-stats [--interval 1] | ane-embed embed TEXT... | ane-embed serve [--port 11435] [--host 127.0.0.1] [--allow-network] [--workers N] [--assets DIR] | ane-embed prep IMAGE OUT_PREFIX | ane-embed images --assets <embeddinggemma2-w16> [--layout v1|v2] IMAGE... | ane-embed join anejoin://HOST:PORT/TOKEN [--name N] [--slots 2]"

var args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    defer { args.removeSubrange(i...(i + 1)) }
    return args[i + 1]
}
func flag(_ name: String) -> Bool {
    guard let i = args.firstIndex(of: name) else { return false }
    args.remove(at: i)
    return true
}

do {
    guard let command = args.first else { print(usage); exit(2) }
    args.removeFirst()
    if command == "ane-stats" {
        // ane-embed ane-stats [--interval 1]: one JSON line per interval with the Neural Engine's
        // utilization and bandwidth (IOReport, no root, no model load). ane-watch reads this.
        let interval = Double(option("--interval") ?? "1") ?? 1
        guard let monitor = ANEMonitor() else {
            FileHandle.standardError.write("ane-embed: IOReport ANE channels not available\n".data(using: .utf8)!)
            exit(1)
        }
        setvbuf(stdout, nil, _IOLBF, 0)
        signal(SIGPIPE, SIG_DFL)
        while true {
            try await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
            if let r = monitor.read() {
                print(String(format: "{\"ts\":%.3f,\"ane_util_pct\":%.2f,\"ane_gbs\":%.3f}",
                             Date().timeIntervalSince1970, r.utilizationPercent, r.bandwidthGBs))
            }
        }
    }
    if command == "prep" {
        // ane-embed prep IMAGE OUT_PREFIX: the Swift image processor's output, for a parity check against
        // transformers: OUT_PREFIX.f32 (patches x 768 float32) and OUT_PREFIX.json (patch grid)
        guard args.count == 2 else { print(usage); exit(2) }
        let vision = try Vision()
        let image = try prepareImage(try Data(contentsOf: URL(fileURLWithPath: args[0])), manifest: vision.manifest)
        try image.patches.withUnsafeBytes { Data($0) }.write(to: URL(fileURLWithPath: args[1] + ".f32"))
        try Data("{\"columns\": \(image.columns), \"rows\": \(image.rows)}\n".utf8).write(to: URL(fileURLWithPath: args[1] + ".json"))
        print("\(args[0]): \(image.columns) x \(image.rows) patches -> \(args[1]).f32")
        exit(0)
    }
    let workers = Int(option("--workers") ?? "2") ?? 2
    let layout = option("--layout")
    let root = option("--assets").map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) } ?? Assets.defaultRoot
    let started = Date()
    let engine = try await Engine(assets: try Assets(root: root), workers: workers)
    FileHandle.standardError.write("ane-embed: \(engine.workerCount) workers ready in \(String(format: "%.1f", Date().timeIntervalSince(started))) s\n".data(using: .utf8)!)
    switch command {
    case "images":
        // ane-embed images --assets <embeddinggemma2-w16> [--layout v1|v2] IMAGE...: image vectors, as JSON
        let vision = try Vision()
        let images = try args.map { try Data(contentsOf: URL(fileURLWithPath: $0)) }
        let t0 = Date()
        let vectors = try await engine.embedImages(images, vision: vision, layout: layout)
        FileHandle.standardError.write("ane-embed: \(images.count) images in \(String(format: "%.2f", Date().timeIntervalSince(t0))) s (vision \(layout ?? vision.manifest.default_layout), first includes its compile)\n".data(using: .utf8)!)
        print(String(decoding: encodeEmbedResponse(model: engine.assets.manifest.model_tag, vectors: vectors), as: UTF8.self))
    case "join":
        // ane-embed join anejoin://host:port/TOKEN  |  ane-embed join http://host:port --token T  [--name N]
        // this Mac pulls parts of the pool's work and embeds them on its own ANE, until Ctrl-C
        let tokenOption = option("--token")
        let slots = Int(option("--slots") ?? "") ?? 2
        let name = option("--name") ?? (Host.current().localizedName ?? ProcessInfo.processInfo.hostName)
            .replacingOccurrences(of: " ", with: "-")
        guard let target = args.first else { print(usage); exit(2) }
        let (pool, token): (URL, String)
        if let invite = Volunteer.parseInvite(target) { (pool, token) = invite }
        else if let url = URL(string: target), let t = tokenOption { (pool, token) = (url, t) }
        else { FileHandle.standardError.write("ane-embed: give an anejoin:// invite, or a pool URL with --token\n".data(using: .utf8)!); exit(2) }
        let volunteer = Volunteer(pool: pool, token: token, name: name, engine: engine)
        setvbuf(stdout, nil, _IOLBF, 0)
        print("joined \(pool.absoluteString) as \(name): asking for work, \(slots) chunks at a time (Ctrl-C to leave)")
        await volunteer.run(slots: slots) { print("\(Date().formatted(date: .omitted, time: .standard))  \($0)") }
    case "embed":
        let vectors = try await engine.embed(args)
        print(String(decoding: encodeEmbedResponse(model: engine.assets.manifest.model_tag, vectors: vectors), as: UTF8.self))
    case "serve":
        let host = option("--host") ?? "127.0.0.1"
        let port = UInt16(option("--port") ?? "11435") ?? 11435
        let allowNetwork = flag("--allow-network")
        let server = try EmbedServer(engine: engine, host: host, port: port, allowNetwork: allowNetwork,
                                     joinHosts: flag("--join") ? EmbedServer.joinAddresses(lan: flag("--lan")) : [])
        if let join = server.joinCommand { FileHandle.standardError.write("ane-embed: another Mac joins with:  \(join)\n".data(using: .utf8)!) }
        if let extra = option("--also") {          // --also <assets root>: serve a second model by its tag
            let e2 = try await Engine(assets: try Assets(root: URL(fileURLWithPath: (extra as NSString).expandingTildeInPath)), workers: 1)
            server.addModel(e2)
        }
        try server.start()
        FileHandle.standardError.write("ane-embed: serving http://\(host):\(port)/api/embed\n".data(using: .utf8)!)
        while true { try await Task.sleep(nanoseconds: 3_600_000_000_000) }
    case "bench":
        // ane-embed bench [short|long] [batch] [reps]: one request at a time, then the split.
        let kind = args.first ?? "long"
        let batch = args.count > 1 ? Int(args[1]) ?? 256 : 256
        let reps = args.count > 2 ? Int(args[2]) ?? 3 : 3
        let long = String(repeating: "The model reads its weights from memory for every token it produces. ", count: 30)
        let texts = (0..<batch).map { kind == "short" ? "Short note about the weather today. #\($0)" : String(long.prefix(2000)) + " #\($0)" }
        _ = try await engine.embed(texts)
        engine.stats.resetForBench(); engine.tokenizeSeconds = 0
        let t0 = Date()
        for _ in 0..<reps { _ = try await engine.embed(texts) }
        let wall = Date().timeIntervalSince(t0)
        let s = engine.stats.snapshot()
        let per = { (x: Double) in String(format: "%.1f", 1000 * x / Double(reps)) }
        print("\(kind) batch \(batch): \(String(format: "%.1f", Double(batch * reps) / wall)) texts/s, \(per(wall)) ms/request, calls \(s.calls / reps)/request")
        print("  per request (ms): tokenize \(per(engine.tokenizeSeconds)) stage \(per(s.stageSeconds)) predict \(per(s.predictSeconds)) pool \(per(s.poolSeconds))  [stage/predict/pool summed over \(engine.workerCount) workers]")
    default:
        print(usage); exit(2)
    }
} catch {
    FileHandle.standardError.write("ane-embed: \(error)\n".data(using: .utf8)!)
    exit(1)
}
