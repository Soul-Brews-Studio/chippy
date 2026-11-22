import ANEEmbedCore
import Foundation

/// This Mac as a volunteer in someone's pool, BOINC style: it ASKS for work (POST /api/work/claim,
/// a long poll), embeds the part on its own ANE, and posts the vectors back (POST /api/work/result).
/// The pool never connects to us, so it works behind any router, as long as we can reach the pool.
public final class Volunteer {
    public let pool: URL
    let token: String
    public let name: String
    let engine: Engine
    public private(set) var parts = 0, texts = 0, errors = 0
    public private(set) var lastError: String?
    public private(set) var running = false
    private var stopped = false
    private let lock = NSLock()

    public init(pool: URL, token: String, name: String, engine: Engine) {
        self.pool = pool; self.token = token; self.name = name; self.engine = engine
    }

    /// "ane-embed join http://host:11445 --token T" or an invite "anejoin://host:11445/T" -> (pool, token).
    public static func parseInvite(_ s: String) -> (URL, String)? {
        guard s.hasPrefix("anejoin://"), let u = URL(string: "http://" + s.dropFirst("anejoin://".count)) else { return nil }
        let token = u.lastPathComponent
        guard let host = u.host, !token.isEmpty, token != "/" else { return nil }
        return (URL(string: "http://\(host):\(u.port ?? 11435)")!, token)
    }

    public func stop() { stopped = true }

    /// Claim, embed, deliver, forever (until stop()), in `slots` loops at once: while one loop embeds a
    /// chunk, another is already claiming the next or uploading its vectors, so this Mac's ANE does not
    /// sit idle between chunks (prefetch). Errors back off and retry, per loop.
    public func run(slots: Int = 2, log: @escaping (String) -> Void = { _ in }) async {
        running = true
        defer { running = false }
        await withTaskGroup(of: Void.self) { group in
            for slot in 1...max(1, slots) { group.addTask { await self.loop(slot: slot, log: log) } }
        }
    }

    private func loop(slot: Int, log: @escaping (String) -> Void) async {
        var backoff: UInt64 = 1
        let identity = engine.assets.manifest.vector_space_identity
        while !stopped {
            do {
                let (status, job) = try await post("api/work/claim", ["token": token, "worker": name, "identity": identity, "wait": 20])
                if status == 204 { backoff = 1; continue }
                guard status == 200, let id = job["id"] as? String, let input = job["input"] as? [String] else {
                    throw HTTPError(status, (job["error"] as? String) ?? "claim answered \(status)")
                }
                let vectors = try await engine.embed(input)
                let (s2, reply) = try await post("api/work/result", ["token": token, "worker": name, "id": id,
                                                                    "embeddings": vectors.map { $0.map(Double.init) }])
                guard s2 == 200 else { throw HTTPError(s2, (reply["error"] as? String) ?? "result answered \(s2)") }
                let (p, t) = lock.withLock { parts += 1; texts += input.count; return (parts, texts) }
                backoff = 1
                log("part \(p): \(input.count) texts (total \(t))")
            } catch {
                lock.withLock { errors += 1; lastError = String(describing: error) }
                log("error (slot \(slot)): \(error); retrying in \(backoff) s")
                try? await Task.sleep(nanoseconds: backoff * 1_000_000_000)
                backoff = min(backoff * 2, 30)
            }
        }
    }

    private func post(_ path: String, _ body: [String: Any]) async throws -> (Int, [String: Any]) {
        var request = URLRequest(url: pool.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.timeoutInterval = 60
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        return (status, (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:])
    }
}
