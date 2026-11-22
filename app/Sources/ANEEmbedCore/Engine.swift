import CoreML
import Foundation

/// One worker: its own MLModel per bucket, run on its own serial queue. Several workers
/// keep the ANE fed while another worker stages inputs or pools outputs on the CPU —
/// the same reason the Python service runs two worker processes.
final class Worker {
    let models: [Int: MLModel]
    let queue: DispatchQueue
    let index: Int
    let count: Int

    init(index: Int, of count: Int, compiled: [Int: URL], cpuBuckets: Set<Int> = []) throws {
        self.index = index
        self.count = count
        var models: [Int: MLModel] = [:]
        for (bucket, url) in compiled {
            let config = MLModelConfiguration()
            config.computeUnits = cpuBuckets.contains(bucket) ? .cpuOnly : .cpuAndNeuralEngine
            models[bucket] = try MLModel(contentsOf: url, configuration: config)
        }
        self.models = models
        queue = DispatchQueue(label: "ane-embed.worker.\(index)", qos: .userInitiated)
    }

    func run(_ job: Job, tokenIds: [[Int]], assets: Assets, stats: Stats) async throws -> [[Float]] {
        try await run(job, assets: assets, stats: stats, tokenIds: tokenIds) { try stageInputs(job: job, tokenIds: tokenIds, assets: assets) }
    }

    /// Images: the rows are built already (token-table rows around the vision encoder's soft tokens).
    func run(_ job: Job, rows: [[Float16]], assets: Assets, stats: Stats) async throws -> [[Float]] {
        guard assets.isPooled else { throw EmbedError.input("images need a pooled (EmbeddingGemma 2) text core") }
        return try await run(job, assets: assets, stats: stats, tokenIds: []) { try stageRows(job: job, rows: rows, assets: assets) }
    }

    private func run(_ job: Job, assets: Assets, stats: Stats, tokenIds: [[Int]],
                     stage: @escaping () throws -> (MLMultiArray, MLMultiArray)) async throws -> [[Float]] {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                stats.setBusy(worker: self.index, of: self.count, true)
                defer { stats.setBusy(worker: self.index, of: self.count, false) }
                do {
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    let (embeds, mask) = try stage()
                    let input = try MLDictionaryFeatureProvider(dictionary: ["embeds": embeds, "mask": mask])
                    let t1 = DispatchTime.now().uptimeNanoseconds
                    let output = assets.isPooled ? "pooled" : "hidden"
                    guard let model = self.models[job.bucket],
                          let array = try model.prediction(from: input).featureValue(for: output)?.multiArrayValue
                    else { throw EmbedError.runtime("no \(output) output for bucket \(job.bucket)") }
                    let t2 = DispatchTime.now().uptimeNanoseconds
                    let vectors = assets.isPooled ? normalizePooled(array, job: job)
                                                  : projectHidden(array, job: job, tokenIds: tokenIds, assets: assets)
                    let t3 = DispatchTime.now().uptimeNanoseconds
                    stats.recordCall(stage: Double(t1 - t0) / 1e9, predict: Double(t2 - t1) / 1e9, pool: Double(t3 - t2) / 1e9)
                    continuation.resume(returning: vectors)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

public final class Engine {
    public let assets: Assets
    public let stats = Stats()
    /// HF tokenizers (Rust): encodes a whole batch in parallel internally.
    let tokenizer: TextTokenizer
    let workers: [Worker]
    var next = 0
    public var tokenizeSeconds = 0.0
    let lock = NSLock()

    /// Compiles each bucket's mlpackage once into <assets>/compiled, then loads `workers`
    /// copies of every bucket on CPU + Neural Engine.
    public init(assets: Assets, workers count: Int = 2) async throws {
        self.assets = assets
        tokenizer = try TextTokenizer(folder: assets.tokenizerFolder, maxTokens: assets.manifest.max_tokens)
        var compiled: [Int: URL] = [:]
        let cacheDir = assets.root.appendingPathComponent("compiled")
        try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
        for bucket in assets.manifest.buckets {
            let target = cacheDir.appendingPathComponent("\(bucket).mlmodelc")
            if !FileManager.default.fileExists(atPath: target.path) {
                let temp = try await MLModel.compileModel(at: try assets.packageURL(bucket: bucket))
                try FileManager.default.moveItem(at: temp, to: target)
            }
            compiled[bucket] = target
        }
        let cpu = Set(assets.manifest.cpu_buckets ?? [])
        workers = try (0..<max(1, count)).map { try Worker(index: $0, of: max(1, count), compiled: compiled, cpuBuckets: cpu) }
        _ = try await embed(["warmup"])
        stats.reset()
    }

    public var workerCount: Int { workers.count }

    func encode(_ texts: [String]) throws -> [[Int]] { try tokenizer.encode(texts) }

    /// Ordered unit vectors for `texts`. Jobs go round-robin across workers.
    public func embed(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        stats.begin()
        defer { stats.end() }
        let started = Date()
        let tokenIds = try encode(texts)
        tokenizeSeconds += Date().timeIntervalSince(started)
        let jobs = try assignJobs(lengths: tokenIds.map(\.count), buckets: assets.manifest.buckets,
                                  slots: assets.manifest.slots_per_call)
        let picked: [Worker] = lock.withLock {
            defer { next += jobs.count }
            return jobs.indices.map { workers[(next + $0) % workers.count] }
        }
        var out = [[Float]](repeating: [], count: texts.count)
        try await withThrowingTaskGroup(of: (Job, [[Float]]).self) { group in
            for (job, worker) in zip(jobs, picked) {
                group.addTask { (job, try await worker.run(job, tokenIds: tokenIds, assets: self.assets, stats: self.stats)) }
            }
            for try await (job, vectors) in group {
                for (row, vector) in zip(job.rows, vectors) { out[row] = vector }
            }
        }
        stats.record(texts: texts.count, tokens: tokenIds.reduce(0) { $0 + $1.count },
                     slots: jobs.reduce(0) { $0 + max(1, assets.manifest.slots_per_call / $1.bucket) * $1.bucket },
                     seconds: Date().timeIntervalSince(started))
        return out
    }

    /// Ordered unit vectors for images (ANE Embed 0.2): each image through the vision encoder on the
    /// ANE (`layout` v1 or v2, default the manifest's), its sequence then through this text core like a
    /// text. Same vector space as the texts this engine embeds.
    public func embedImages(_ images: [Data], vision: Vision, layout: String? = nil) async throws -> [[Float]] {
        guard !images.isEmpty else { return [] }
        guard vision.manifest.model_tag == assets.manifest.model_tag else {
            throw EmbedError.input("vision assets are for \(vision.manifest.model_tag), not \(assets.manifest.model_tag)")
        }
        stats.begin()
        defer { stats.end() }
        let started = Date()
        let chosen = layout ?? vision.manifest.default_layout
        var rows: [[Float16]] = []
        for image in images { rows.append(try await vision.sequence(for: image, layout: chosen, assets: assets)) }
        let lengths = rows.map { $0.count / assets.manifest.hidden }
        let jobs = try assignJobs(lengths: lengths, buckets: assets.manifest.buckets, slots: assets.manifest.slots_per_call)
        let picked: [Worker] = lock.withLock {
            defer { next += jobs.count }
            return jobs.indices.map { workers[(next + $0) % workers.count] }
        }
        var out = [[Float]](repeating: [], count: images.count)
        try await withThrowingTaskGroup(of: (Job, [[Float]]).self) { group in
            for (job, worker) in zip(jobs, picked) {
                group.addTask { (job, try await worker.run(job, rows: rows, assets: self.assets, stats: self.stats)) }
            }
            for try await (job, vectors) in group {
                for (row, vector) in zip(job.rows, vectors) { out[row] = vector }
            }
        }
        stats.record(texts: images.count, tokens: lengths.reduce(0, +),
                     slots: jobs.reduce(0) { $0 + max(1, assets.manifest.slots_per_call / $1.bucket) * $1.bucket },
                     seconds: Date().timeIntervalSince(started))
        return out
    }
}
