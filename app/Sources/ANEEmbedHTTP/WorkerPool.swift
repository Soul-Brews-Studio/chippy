import ANEEmbedCore
import Foundation
import Security

/// The embed workers this server spreads requests over: always "local" (this Mac's
/// engine) plus any remote /api/embed services added through /api/workers.
///
/// Work stealing: a cluster request is cut into chunks of `chunkSize` texts in one shared queue,
/// and every worker (local pumps, remote pumps, volunteers claiming) takes the next chunk when it is
/// free, so a faster Mac simply does more. A chunk that fails or times out goes back to the front of
/// the queue (at most `maxTries` times). Weight only says whether a worker takes part (> 0). A remote is
/// only accepted when its /health names the same vector_space_identity as the local
/// engine, so vectors from different models can never be mixed.
public final class WorkerPool {
    public struct Info: Codable {
        public var id: String
        public var url: String?
        public var weight: Double
        public var texts = 0, requests = 0, errors = 0, inFlight = 0
        public var textsPerSecond = 0.0
        public var lastError: String?
    }

    final class Member {
        var info: Info
        var window: [(Date, Int)] = []
        /// Volunteers (pull workers, url "pull:<name>"): chunks they hold, last claim.
        var held: [String: PullJob] = [:]
        var lastSeen = Date.distantPast
        var badResults = 0
        init(_ info: Info) { self.info = info }
        var isPull: Bool { info.url?.hasPrefix("pull:") ?? false }
    }

    /// One chunk of a request; whoever finishes it (or gives up on it) resumes the continuation.
    struct PullJob {
        let id: String
        let texts: [String]
        let continuation: CheckedContinuation<[[Float]], Error>
        var tries = 0
    }

    public static let hopHeader = "X-ANE-Embed-Hop"
    /// A volunteer that has not claimed work for this long gets no new parts.
    static let pullAlive: TimeInterval = 30
    /// A chunk a volunteer has not answered in this long goes back to the queue.
    static let pullTimeout: TimeInterval = 60
    /// Texts per chunk of a cluster request, and how often one chunk may fail before the request does.
    public static var chunkSize = 32
    static let maxTries = 3
    /// Most chunks this Mac's engine takes in one go (8 x 32 = 256 texts, about one full request).
    static let localTake = 8
    /// The shared queue, and how many pump tasks each local/remote worker has running on it.
    private var shared: [PullJob] = []
    private var pumping: [ObjectIdentifier: Int] = [:]
    let engine: Engine
    let file: URL
    /// The secret a volunteer shows to claim work (join.json beside workers.json, made once).
    public let joinToken: String
    private var members: [Member]
    private let lock = NSLock()

    public init(engine: Engine, file: URL = Assets.defaultRoot.deletingLastPathComponent().appendingPathComponent("workers.json")) {
        self.engine = engine
        self.file = file
        var saved: [Info] = []
        if let data = try? Data(contentsOf: file), let list = try? JSONDecoder().decode([Info].self, from: data) { saved = list }
        let local = saved.first { $0.id == "local" } ?? Info(id: "local", url: nil, weight: 100)
        members = [Member(Info(id: "local", url: nil, weight: local.weight))] +
                  saved.filter { $0.id != "local" && !($0.url ?? "").hasPrefix("pull:") }
                       .map { Member(Info(id: $0.id, url: $0.url, weight: $0.weight)) }
        let joinFile = file.deletingLastPathComponent().appendingPathComponent("join.json")
        if let data = try? Data(contentsOf: joinFile), let saved = try? JSONDecoder().decode([String: String].self, from: data),
           let token = saved["token"], !token.isEmpty {
            joinToken = token
        } else {
            var bytes = [UInt8](repeating: 0, count: 18)
            _ = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
            joinToken = Data(bytes).base64EncodedString().replacingOccurrences(of: "/", with: "_").replacingOccurrences(of: "+", with: "-")
            if let data = try? JSONEncoder().encode(["token": joinToken]) {
                try? data.write(to: joinFile, options: .atomic)
                try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: joinFile.path)
            }
        }
    }

    public func list() -> [Info] {
        lock.withLock {
            let cutoff = Date().addingTimeInterval(-10)
            return members.map { m in
                m.window.removeAll { $0.0 < cutoff }
                var info = m.info
                info.textsPerSecond = Double(m.window.reduce(0) { $0 + $1.1 }) / 10
                return info
            }
        }
    }

    public var hasRemotes: Bool { lock.withLock { members.count > 1 } }

    /// Add or re-weight a remote after checking it serves the same vector space.
    public func add(url raw: String, weight: Double) async throws -> Info {
        guard let url = URL(string: raw.hasSuffix("/") ? String(raw.dropLast()) : raw),
              url.scheme == "http" || url.scheme == "https", url.host != nil else { throw HTTPError(400, "url must be http(s)://host:port") }
        guard weight > 0 else { throw HTTPError(400, "weight must be > 0") }
        var request = URLRequest(url: url.appendingPathComponent("health"))
        request.timeoutInterval = 5
        let data: Data
        do { (data, _) = try await URLSession.shared.data(for: request) } catch {
            throw HTTPError(502, "\(url.absoluteString) did not answer /health: \(error.localizedDescription)")
        }
        let theirs = Self.findIdentity(in: try? JSONSerialization.jsonObject(with: data))
        let ours = engine.assets.manifest.vector_space_identity
        guard theirs == ours else {
            throw HTTPError(409, "refused: \(url.absoluteString) serves \(theirs ?? "an unknown vector space"), this pool serves \(ours)")
        }
        let id = url.absoluteString
        let info: Info = lock.withLock {
            if let m = members.first(where: { $0.info.id == id }) { m.info.weight = weight; return m.info }
            let m = Member(Info(id: id, url: id, weight: weight))
            members.append(m)
            return m.info
        }
        save()
        return info
    }

    public func setLocalWeight(_ weight: Double) { lock.withLock { members[0].info.weight = max(0, weight) }; save() }

    public func remove(id: String) throws {
        guard id != "local" else { throw HTTPError(400, "the local worker cannot be removed; set its weight to 0 instead") }
        let removed = lock.withLock { () -> Bool in
            let before = members.count
            members.removeAll { $0.info.id == id }
            return members.count < before
        }
        guard removed else { throw HTTPError(404, "no worker \(id)") }
        save()
    }

    /// Ordered vectors for `texts`. `hop` (solo, or forwarded by another pool) stays local.
    public func embed(_ texts: [String], hop: Bool) async throws -> [[Float]] {
        let now = Date()
        let active = lock.withLock { members.filter { $0.info.weight > 0 && (!$0.isPull || now.timeIntervalSince($0.lastSeen) < Self.pullAlive) } }
        if hop || active.count <= 1 || texts.count < 2 {
            let one = hop ? members[0] : (active.first ?? members[0])
            return one.isPull ? try await enqueue(texts) : try await run(texts, on: one)
        }
        let size = max(1, Self.chunkSize)
        var out = [[Float]](repeating: [], count: texts.count)
        try await withThrowingTaskGroup(of: (Int, [[Float]]).self) { group in
            for start in stride(from: 0, to: texts.count, by: size) {
                let part = Array(texts[start..<min(start + size, texts.count)])
                group.addTask { (start, try await self.enqueue(part)) }
            }
            for try await (start, vectors) in group { out.replaceSubrange(start..<start + vectors.count, with: vectors) }
        }
        return out
    }

    /// Put one chunk on the shared queue and wait until some worker has embedded it.
    private func enqueue(_ texts: [String]) async throws -> [[Float]] {
        try await withCheckedThrowingContinuation { continuation in
            lock.withLock { shared.append(PullJob(id: UUID().uuidString, texts: texts, continuation: continuation)) }
            startPumps()
        }
    }

    /// A chunk that failed: back to the front of the queue, or fail its request after maxTries.
    private func requeue(_ job: PullJob, _ error: Error) {
        var job = job
        job.tries += 1
        if job.tries >= Self.maxTries { job.continuation.resume(throwing: error); return }
        lock.withLock { shared.insert(job, at: 0) }
        startPumps()
    }

    /// Make sure every local/remote worker with weight > 0 has its pump tasks running on the queue
    /// (local: one per engine worker plus one, so the ANE stays busy between chunks; a remote: two).
    private func startPumps() {
        let start: [Member] = lock.withLock {
            guard !shared.isEmpty else { return [] }
            var list: [Member] = []
            for m in members where !m.isPull && m.info.weight > 0 {
                let want = m.info.url == nil ? engine.workerCount + 1 : 2
                let have = pumping[ObjectIdentifier(m)] ?? 0
                if have < want {
                    pumping[ObjectIdentifier(m)] = want
                    list += Array(repeating: m, count: want - have)
                }
            }
            return list
        }
        for m in start { Task { await self.pump(m) } }
    }

    /// Take the next chunk(s), embed them on `member`, repeat until the queue is empty. This Mac takes
    /// several chunks at once while the queue is long (half an even share per taker, up to `localTake`), so its ANE
    /// packs a full batch; a remote takes one. A failure puts the chunks back and ends this pump (a
    /// broken remote must not eat the queue).
    private func pump(_ member: Member) async {
        let local = member.info.url == nil
        while true {
            let jobs: [PullJob] = lock.withLock {
                guard member.info.weight > 0, !shared.isEmpty else {
                    pumping[ObjectIdentifier(member)] = max(0, (pumping[ObjectIdentifier(member)] ?? 1) - 1)
                    return []
                }
                var n = 1
                if local {
                    let now = Date()
                    // guided self-scheduling: half of an even share per taker, so the share shrinks as
                    // the queue drains and nobody is left holding a long tail
                    let remotes = members.filter { $0.info.weight > 0 && $0.info.url != nil && (!$0.isPull || now.timeIntervalSince($0.lastSeen) < Self.pullAlive) }.count
                    n = min(Self.localTake, max(1, shared.count / (2 * (engine.workerCount + 1 + remotes))))
                }
                let taken = Array(shared.prefix(n))
                shared.removeFirst(taken.count)
                return taken
            }
            if jobs.isEmpty { return }
            do {
                let vectors = try await run(jobs.flatMap(\.texts), on: member)
                var at = 0
                for job in jobs {
                    job.continuation.resume(returning: Array(vectors[at..<at + job.texts.count]))
                    at += job.texts.count
                }
            } catch {
                lock.withLock { pumping[ObjectIdentifier(member)] = max(0, (pumping[ObjectIdentifier(member)] ?? 1) - 1) }
                for job in jobs { requeue(job, error) }
                return
            }
        }
    }

    private func run(_ texts: [String], on member: Member) async throws -> [[Float]] {
        lock.withLock { member.info.inFlight += 1 }
        defer { lock.withLock { member.info.inFlight -= 1 } }
        do {
            let vectors: [[Float]]
            if member.info.url == nil { vectors = try await engine.embed(texts) }
            else { vectors = try await remote(texts, url: member.info.url!) }
            lock.withLock {
                member.info.texts += texts.count; member.info.requests += 1; member.info.lastError = nil
                member.window.append((Date(), texts.count))
            }
            return vectors
        } catch {
            lock.withLock { member.info.errors += 1; member.info.lastError = String(describing: error) }
            throw error
        }
    }

    private func remote(_ texts: [String], url: String) async throws -> [[Float]] {
        var request = URLRequest(url: URL(string: url + "/api/embed")!)
        request.httpMethod = "POST"
        request.timeoutInterval = 600
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("1", forHTTPHeaderField: Self.hopHeader)
        request.httpBody = try JSONSerialization.data(withJSONObject: ["model": engine.assets.manifest.model_tag, "input": texts])
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = object["embeddings"] as? [[NSNumber]], rows.count == texts.count
        else { throw HTTPError(502, "\(url) returned no or wrong embeddings") }
        return rows.map { $0.map(\.floatValue) }
    }

    // MARK: volunteers: they pull work, so a Mac behind any router can join

    /// A volunteer asks for work (long poll, up to `wait` seconds). It is registered on its first claim,
    /// after its vector space is checked against this pool's.
    public func claim(worker name: String, identity: String, wait: TimeInterval = 20) async throws -> (id: String, texts: [String])? {
        let ours = engine.assets.manifest.vector_space_identity
        guard identity == ours else { throw HTTPError(409, "refused: you serve \(identity), this pool serves \(ours)") }
        let clean = name.filter { $0.isLetter || $0.isNumber || "-_.".contains($0) }.prefix(40)
        guard !clean.isEmpty else { throw HTTPError(400, "worker name is required (letters, digits, - _ .)") }
        let id = "pull:" + clean
        let member: Member = lock.withLock {
            if let m = members.first(where: { $0.info.id == id }) { m.lastSeen = Date(); return m }
            let m = Member(Info(id: id, url: id, weight: 50))
            m.lastSeen = Date()
            members.append(m)
            return m
        }
        if lock.withLock({ member.badResults >= 3 }) { throw HTTPError(403, "this worker sent 3 bad answers and gets no more work") }
        let deadline = Date().addingTimeInterval(wait)
        while Date() < deadline {
            let job: PullJob? = lock.withLock {
                member.lastSeen = Date()
                guard member.info.weight > 0, !shared.isEmpty else { return nil }
                let job = shared.removeFirst()
                member.held[job.id] = job
                member.info.inFlight += 1
                return job
            }
            if let job {
                let timeout = Self.pullTimeout
                Task { [weak self] in
                    try? await Task.sleep(nanoseconds: UInt64(timeout * 1e9))
                    guard let self else { return }
                    let late: PullJob? = self.lock.withLock {
                        guard let j = member.held.removeValue(forKey: job.id) else { return nil }
                        member.info.inFlight -= 1; member.info.errors += 1
                        member.info.lastError = "did not answer in \(Int(timeout)) s"
                        return j
                    }
                    if let late { self.requeue(late, HTTPError(504, "\(member.info.id) did not answer in \(Int(timeout)) s")) }
                }
                return (job.id, job.texts)
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        return nil
    }

    /// A volunteer's answer. Shape and unit length are checked now; the spot check (the chunk's first
    /// text embedded here must match, cosine >= 0.995) runs after the volunteer has its reply, so it
    /// never waits on this Mac's busy ANE. Nobody sees the vectors before the spot check passes; a
    /// failed chunk goes back to the queue. Three bad answers and the volunteer gets no more work.
    public func deliver(worker name: String, id: String, vectors: [[Float]]) async throws {
        let mid = "pull:" + name.filter { $0.isLetter || $0.isNumber || "-_.".contains($0) }.prefix(40)
        let found: (Member, PullJob)? = lock.withLock {
            guard let m = members.first(where: { $0.info.id == mid }), let job = m.held.removeValue(forKey: id) else { return nil }
            m.lastSeen = Date()
            m.info.inFlight -= 1
            return (m, job)
        }
        guard let (member, job) = found else { throw HTTPError(410, "no part \(id) for \(mid) (already timed out?)") }
        let dim = engine.assets.manifest.output_dim ?? engine.assets.manifest.hidden
        if vectors.count != job.texts.count || vectors.contains(where: { $0.count != dim }) {
            throw reject(member, job, "wrong shape")
        }
        if vectors.contains(where: { v in let n = v.reduce(0) { $0 + $1 * $1 }; return !n.isFinite || abs(n - 1) > 0.01 }) {
            throw reject(member, job, "not unit vectors")
        }
        Task {
            var problem: String?
            if let mine = try? await engine.embed([job.texts[0]]).first {
                let cos = zip(mine, vectors[0]).reduce(0) { $0 + $1.0 * $1.1 }
                if cos < 0.995 { problem = String(format: "spot check failed (cosine %.4f)", cos) }
            }
            if let problem { _ = self.reject(member, job, problem); return }
            self.lock.withLock {
                member.info.texts += job.texts.count; member.info.requests += 1; member.info.lastError = nil
                member.window.append((Date(), job.texts.count))
            }
            job.continuation.resume(returning: vectors)
        }
    }

    /// Count a bad answer against a volunteer, put its chunk back on the queue, return the error for it.
    private func reject(_ member: Member, _ job: PullJob, _ problem: String) -> HTTPError {
        let off = lock.withLock { () -> Bool in
            member.badResults += 1
            member.info.errors += 1; member.info.lastError = problem
            if member.badResults >= 3 { member.info.weight = 0 }
            return member.badResults >= 3
        }
        requeue(job, HTTPError(502, "\(member.info.id): \(problem)"))
        return HTTPError(422, "\(problem)\(off ? "; this worker gets no more work" : "")")
    }

    private func save() {
        let infos = lock.withLock { members.filter { !$0.isPull }.map { Info(id: $0.info.id, url: $0.info.url, weight: $0.info.weight) } }
        if let data = try? JSONEncoder().encode(infos) { try? data.write(to: file, options: .atomic) }
    }

    /// The first "vector_space_identity" anywhere in a /health document (the Swift
    /// service puts it at the top, the Python service under its model's metadata).
    static func findIdentity(in value: Any?) -> String? {
        if let dict = value as? [String: Any] {
            if let id = dict["vector_space_identity"] as? String { return id }
            for (_, v) in dict { if let id = findIdentity(in: v) { return id } }
        } else if let list = value as? [Any] {
            for v in list { if let id = findIdentity(in: v) { return id } }
        }
        return nil
    }
}
