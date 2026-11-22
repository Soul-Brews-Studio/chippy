import ANEEmbedCore
import Foundation
import Network

/// A small HTTP/1.1 server for the Ollama embed API: POST /api/embed, GET /api/tags,
/// GET /health. One request per connection (Connection: close), like the Python service.
/// Binds loopback only, unless allowNetwork and the host is a Tailscale address.
public final class EmbedServer {
    let engine: Engine
    /// More models served side by side, keyed by model tag (EmbeddingGemma 2 next to v1). A request
    /// naming one goes straight to it; the pool and every request without a model stay on `engine`.
    private var extraModels: [String: Engine] = [:]
    private let extraLock = NSLock()
    var extra: [String: Engine] { extraLock.withLock { extraModels } }
    /// Attach another model while serving (it can take minutes to compile for the ANE).
    public func addModel(_ e: Engine) { extraLock.withLock { extraModels[e.assets.manifest.model_tag] = e } }
    /// Vision encoders by the text model tag they feed (ANE Embed 0.2): requests with "images" go there.
    private var visionModels: [String: Vision] = [:]
    var visions: [String: Vision] { extraLock.withLock { visionModels } }
    public func attachVision(_ v: Vision) { extraLock.withLock { visionModels[v.manifest.model_tag] = v } }
    static let maxImages = 32
    public let pool: WorkerPool
    let listener: NWListener
    /// Extra listeners on this Mac's tailscale addresses for volunteers only (join kit + work
    /// endpoints + /health): another Mac can join without the embed API leaving loopback.
    let joinListeners: [NWListener]
    public let joinHosts: [String]
    /// The ane-embed binary handed to joining Macs (the app ships it beside itself).
    public var kitBinary: URL?
    let queue = DispatchQueue(label: "ane-embed.http", qos: .userInitiated, attributes: .concurrent)
    public let host: String
    public let port: UInt16
    static let maxBody = 16 * 1024 * 1024
    static let maxInputs = 256

    public init(engine: Engine, host: String, port: UInt16, allowNetwork: Bool, joinHosts: [String] = []) throws {
        guard EmbedServer.bindAllowed(host: host, allowNetwork: allowNetwork) else {
            throw HTTPError(0, "refusing to bind \(host): loopback only, or a Tailscale 100.64.0.0/10 address with --allow-network")
        }
        self.engine = engine
        pool = WorkerPool(engine: engine)
        self.host = host
        self.port = port
        let parameters = NWParameters.tcp
        parameters.allowLocalEndpointReuse = true     // the join listeners share the port on another address
        parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port)!)
        listener = try NWListener(using: parameters)
        let extra = joinHosts.filter { $0 != host && (EmbedServer.bindAllowed(host: $0, allowNetwork: true) || EmbedServer.isPrivateLAN($0)) }
        self.joinHosts = extra
        joinListeners = try extra.map { h in
            let p = NWParameters.tcp
            p.allowLocalEndpointReuse = true
            p.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(h), port: NWEndpoint.Port(rawValue: port)!)
            return try NWListener(using: p)
        }
        kitBinary = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent().appendingPathComponent("ane-embed")
    }

    /// 10/8, 172.16/12, 192.168/16: a home or office network. Volunteers on the same LAN join there
    /// (found 2026-11-22: a Mac mini shared from another tailnet pings m5 but cannot open its ports).
    static func isPrivateLAN(_ ip: String) -> Bool {
        let o = ip.split(separator: ".").compactMap { Int($0) }
        guard o.count == 4 else { return false }
        return o[0] == 10 || (o[0] == 172 && (16...31).contains(o[1])) || (o[0] == 192 && o[1] == 168)
    }

    /// This Mac's addresses volunteers can reach: tailscale (100.64.0.0/10), and private LAN ones when `lan`.
    public static func joinAddresses(lan: Bool) -> [String] {
        localIPv4().filter { (bindAllowed(host: $0, allowNetwork: true) && !$0.hasPrefix("127.")) || (lan && isPrivateLAN($0)) }
    }

    /// This Mac's tailscale IPv4 addresses (100.64.0.0/10).
    public static func tailscaleAddresses() -> [String] { joinAddresses(lan: false) }

    static func localIPv4() -> [String] {
        var out: [String] = []
        var head: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&head) == 0 else { return [] }
        defer { freeifaddrs(head) }
        var p = head
        while let a = p {
            if let sa = a.pointee.ifa_addr, sa.pointee.sa_family == UInt8(AF_INET) {
                var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
                if getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 {
                    out.append(String(cString: host))
                }
            }
            p = a.pointee.ifa_next
        }
        return Array(Set(out)).sorted()
    }

    /// The one line another Mac runs to join (nil when no tailscale listener is up).
    public var joinCommand: String? {
        let preferred = joinHosts.first { $0.hasPrefix("100.") } ?? joinHosts.first
        guard let h = preferred ?? (host == "127.0.0.1" ? nil : host) else { return nil }
        return "curl -fsS 'http://\(h):\(port)/join.sh?t=\(pool.joinToken)' | zsh"
    }

    static func bindAllowed(host: String, allowNetwork: Bool) -> Bool {
        if host == "127.0.0.1" || host == "::1" || host == "localhost" { return true }
        let octets = host.split(separator: ".").compactMap { UInt8($0) }
        return allowNetwork && octets.count == 4 && octets[0] == 100 && (64...127).contains(octets[1])
    }

    public func start() throws {
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { return }
            connection.start(queue: self.queue)
            self.receive(connection, buffer: Data())
        }
        listener.start(queue: queue)
        for l in joinListeners {
            l.stateUpdateHandler = { state in
                if case .failed(let e) = state { FileHandle.standardError.write("ane-embed: join listener failed: \(e)\n".data(using: .utf8)!) }
                if case .waiting(let e) = state { FileHandle.standardError.write("ane-embed: join listener waiting: \(e)\n".data(using: .utf8)!) }
            }
            l.newConnectionHandler = { [weak self] connection in
                guard let self else { return }
                connection.start(queue: self.queue)
                self.receive(connection, buffer: Data(), joinOnly: true)
            }
            l.start(queue: queue)
        }
    }

    public func stop() { listener.cancel(); joinListeners.forEach { $0.cancel() } }

    func receive(_ connection: NWConnection, buffer: Data, joinOnly: Bool = false) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, done, error in
            guard let self else { return }
            var buffer = buffer
            if let data { buffer.append(data) }
            if let request = HTTPRequest(buffer) {
                Task(priority: .userInitiated) { await self.respond(connection, to: request, joinOnly: joinOnly) }
            } else if buffer.count > EmbedServer.maxBody + 65536 {
                self.send(connection, status: 413, body: Data("{\"error\":\"request body is too large\"}".utf8))
            } else if done || error != nil {
                connection.cancel()
            } else {
                self.receive(connection, buffer: buffer, joinOnly: joinOnly)
            }
        }
    }

    /// Changing the worker list is only allowed from this Mac.
    static func isLoopback(_ connection: NWConnection) -> Bool {
        guard case .hostPort(let host, _) = connection.endpoint else { return false }
        switch host {
        case .ipv4(let a): return a.isLoopback
        case .ipv6(let a): return a.isLoopback || "\(a)".hasPrefix("::ffff:127.")
        case .name(let n, _): return n == "localhost"
        @unknown default: return false
        }
    }

    func respond(_ connection: NWConnection, to request: HTTPRequest, joinOnly: Bool = false) async {
        do {
            if joinOnly, !(request.path.hasPrefix("/join") || request.path.hasPrefix("/api/work/") || request.path == "/health") {
                throw HTTPError(403, "this address is for volunteers: /join.sh, /api/work/*, /health")
            }
            if request.path.hasPrefix("/join"), request.path != "/api/join" {
                try serveKit(connection, request)
                return
            }
            if request.path == "/api/workers", request.method != "GET", !Self.isLoopback(connection) {
                throw HTTPError(403, "workers can only be changed from this Mac")
            }
            switch (request.method, request.path) {
            case ("GET", "/api/workers"):
                send(connection, status: 200, body: try JSONEncoder().encode(["workers": pool.list()]))
            case ("POST", "/api/workers"):
                guard let object = try JSONSerialization.jsonObject(with: request.body) as? [String: Any] else {
                    throw HTTPError(400, "body must be {\"url\": ..., \"weight\": ...}")
                }
                let weight = (object["weight"] as? NSNumber)?.doubleValue ?? 100
                if let url = object["url"] as? String {
                    send(connection, status: 200, body: try JSONEncoder().encode(try await pool.add(url: url, weight: weight)))
                } else if object["id"] as? String == "local" {
                    pool.setLocalWeight(weight)
                    send(connection, status: 200, body: try JSONEncoder().encode(["workers": pool.list()]))
                } else { throw HTTPError(400, "url is required (or id: local with a weight)") }
            case ("DELETE", "/api/workers"):
                let id = request.query["url"] ?? request.query["id"] ??
                    ((try? JSONSerialization.jsonObject(with: request.body)) as? [String: Any])?["url"] as? String
                guard let id else { throw HTTPError(400, "give ?url=<worker url>") }
                try pool.remove(id: id)
                send(connection, status: 200, body: try JSONEncoder().encode(["workers": pool.list()]))
            case ("POST", "/api/embed"):
                let body = try EmbedRequest(request.body)
                let tag = engine.assets.manifest.model_tag
                if !body.images.isEmpty {
                    guard body.images.count <= EmbedServer.maxImages else { throw HTTPError(413, "at most \(EmbedServer.maxImages) images") }
                    let visions = self.visions
                    // the named model, else the only one that has a vision encoder attached
                    guard let model = body.model ?? (visions.count == 1 ? visions.keys.first : nil), let vision = visions[model] else {
                        throw HTTPError(404, "images: no vision encoder for \(body.model ?? "this request"); have \(visions.keys.sorted().joined(separator: ", "))")
                    }
                    guard let target = model == tag ? engine : extra[model] else { throw HTTPError(503, "\(model) is still loading") }
                    do {
                        send(connection, status: 200, body: encodeEmbedResponse(model: model, vectors: try await target.embedImages(body.images, vision: vision, layout: body.vision)))
                    } catch let error as EmbedError {
                        throw HTTPError(400, "\(error)")
                    }
                    return
                }
                guard body.texts.count <= EmbedServer.maxInputs else { throw HTTPError(413, "at most \(EmbedServer.maxInputs) inputs") }
                if let model = body.model, let other = extra[model] {
                    send(connection, status: 200, body: encodeEmbedResponse(model: model, vectors: try await other.embed(body.texts)))
                    return
                }
                if let model = body.model, model != tag {
                    throw HTTPError(404, "model \(model) not found; this service serves \(([tag] + extra.keys.sorted()).joined(separator: ", "))")
                }
                // solo: this Mac only (also what a forwarded request gets); cluster: spread over the pool
                let solo = body.pool == "solo" || request.headers["x-ane-embed-pool"] == "solo"
                let vectors = try await pool.embed(body.texts, hop: solo || request.headers[WorkerPool.hopHeader.lowercased()] != nil)
                send(connection, status: 200, body: encodeEmbedResponse(model: tag, vectors: vectors))
            case ("GET", "/api/join"):
                // how another Mac joins this pool (BOINC style: it pulls work). Only shown on this Mac.
                guard Self.isLoopback(connection) else { throw HTTPError(403, "the join token is only shown on this Mac") }
                let reach = joinHosts.first ?? (host == "127.0.0.1" ? "<this Mac's tailscale name>" : host)
                let invite = "anejoin://\(reach):\(port)/\(pool.joinToken)"
                send(connection, status: 200, body: try JSONSerialization.data(withJSONObject: [
                    "token": pool.joinToken, "invite": invite, "command": joinCommand ?? "ane-embed join \(invite)",
                    "commands": joinHosts.map { "curl -fsS 'http://\($0):\(port)/join.sh?t=\(pool.joinToken)' | NAME=$(scutil --get LocalHostName) SERVICE=1 zsh" },
                    "listening": ([host] + joinHosts).map { "\($0):\(port)" },
                    "note": joinCommand == nil ? "loopback only: no tailscale address found, other Macs cannot reach this pool" : "any Mac on the tailnet joins with the command"]))
            case ("POST", "/api/work/claim"):
                guard let o = try JSONSerialization.jsonObject(with: request.body) as? [String: Any], o["token"] as? String == pool.joinToken else {
                    throw HTTPError(401, "wrong or missing join token")
                }
                let wait = min(25, max(0, (o["wait"] as? NSNumber)?.doubleValue ?? 20))
                if let job = try await pool.claim(worker: o["worker"] as? String ?? "", identity: o["identity"] as? String ?? "", wait: wait) {
                    send(connection, status: 200, body: try JSONSerialization.data(withJSONObject: [
                        "id": job.id, "model": engine.assets.manifest.model_tag, "input": job.texts]))
                } else {
                    send(connection, status: 204, body: Data())
                }
            case ("POST", "/api/work/result"):
                guard let o = try JSONSerialization.jsonObject(with: request.body) as? [String: Any], o["token"] as? String == pool.joinToken else {
                    throw HTTPError(401, "wrong or missing join token")
                }
                guard let id = o["id"] as? String, let rows = o["embeddings"] as? [[NSNumber]] else { throw HTTPError(400, "id and embeddings are required") }
                try await pool.deliver(worker: o["worker"] as? String ?? "", id: id, vectors: rows.map { $0.map(\.floatValue) })
                send(connection, status: 200, body: Data("{\"ok\":true}".utf8))
            case ("GET", "/api/tags"):
                let tag = engine.assets.manifest.model_tag
                let tags = [tag] + extra.keys.sorted()
                send(connection, status: 200, body: try JSONSerialization.data(withJSONObject: ["models": tags.map { ["name": $0, "model": $0] }]))
            case ("GET", "/health"):
                let health = Health(engine: engine, extra: extra, visions: visions)
                send(connection, status: 200, body: try JSONEncoder().encode(health))
            default:
                throw HTTPError(404, "not found")
            }
        } catch let error as HTTPError {
            engine.stats.recordError()
            send(connection, status: error.status, body: Data("{\"error\":\(quote(error.message))}".utf8))
        } catch {
            engine.stats.recordError()
            send(connection, status: 500, body: Data("{\"error\":\"embedding inference failed\"}".utf8))
        }
    }

    /// The join kit (token in ?t=): /join.sh (the script), /join/ane-embed (the binary),
    /// /join/assets.tar (this pool's model, compiled for Core ML, ~1.3 GB, built once and cached).
    func serveKit(_ connection: NWConnection, _ request: HTTPRequest) throws {
        guard request.query["t"] == pool.joinToken else { throw HTTPError(401, "wrong or missing join token (?t=)") }
        let assets = engine.assets.root
        switch request.path {
        case "/join.sh":
            // behind a jump box (Caddy/Kong) the volunteer must talk back through it: forwarded host + scheme
            let h = request.headers["x-forwarded-host"] ?? request.headers["host"] ?? "\(joinHosts.first ?? host):\(port)"
            let scheme = request.headers["x-forwarded-proto"] == "https" ? "https" : "http"
            let kit = String(engine.assets.manifest.vector_space_identity.suffix(12))
            let name = assets.lastPathComponent
            let script = """
            #!/bin/zsh
            # Join the ANE Embed pool at \(scheme)://\(h): this Mac's Neural Engine embeds parts of its work.
            # Installs into ~/ane-volunteer (binary + model, ~1.3 GB once), then asks for work until Ctrl-C.
            # NAME=<label> names this Mac in the pool; SERVICE=1 keeps it volunteering after logout/reboot.
            set -e
            P="\(scheme)://\(h)"; T="\(pool.joinToken)"; D="$HOME/ane-volunteer"
            [[ $(uname -m) == arm64 ]] || { echo "needs Apple silicon (an ANE)"; exit 1; }
            mkdir -p "$D"
            echo "== binary"; curl -fsS "$P/join/ane-embed?t=$T" -o "$D/ane-embed.new"; chmod +x "$D/ane-embed.new"; mv "$D/ane-embed.new" "$D/ane-embed"
            if [[ "$(cat "$D/kit-id" 2>/dev/null)" != "\(kit)" ]]; then
              echo "== model (~1.3 GB, once)"; rm -rf "$D/\(name)"
              curl -fS --progress-bar "$P/join/assets.tar?t=$T" | tar -x -C "$D"
              echo "\(kit)" > "$D/kit-id"
            fi
            ARGS=(join "$P" --token "$T" --assets "$D/\(name)" --workers 1)
            [[ -n "$NAME" ]] && ARGS+=(--name "$NAME")
            if [[ "$SERVICE" == 1 ]]; then
              # keep volunteering across logins and crashes: a LaunchAgent, logs in ~/ane-volunteer/volunteer.log
              L="$HOME/Library/LaunchAgents/studio.soulbrews.ane-volunteer.plist"; mkdir -p "${L:h}"
              { echo '<?xml version="1.0" encoding="UTF-8"?><plist version="1.0"><dict>'
                echo '<key>Label</key><string>studio.soulbrews.ane-volunteer</string><key>ProgramArguments</key><array>'
                echo "<string>$D/ane-embed</string>"; for a in $ARGS; do echo "<string>$a</string>"; done
                echo '</array><key>RunAtLoad</key><true/><key>KeepAlive</key><true/>'
                echo "<key>StandardOutPath</key><string>$D/volunteer.log</string><key>StandardErrorPath</key><string>$D/volunteer.log</string>"
                echo '</dict></plist>'; } > "$L"
              launchctl bootout "gui/$(id -u)" "$L" 2>/dev/null || true
              launchctl bootstrap "gui/$(id -u)" "$L"
              echo "== volunteering as a service (stop: launchctl bootout gui/$(id -u) $L); log: $D/volunteer.log"
              exit 0
            fi
            echo "== joining (the first run compiles the model for this ANE: a few minutes)"
            exec "$D/ane-embed" $ARGS

            """
            send(connection, status: 200, body: Data(script.utf8), type: "text/plain")
        case "/join/ane-embed":
            guard let bin = kitBinary, let data = try? Data(contentsOf: bin, options: .alwaysMapped) else { throw HTTPError(404, "no ane-embed binary beside this server") }
            send(connection, status: 200, body: data, type: "application/octet-stream")
        case "/join/assets.tar":
            let dir = assets.deletingLastPathComponent().appendingPathComponent("volunteer-kit")
            let tar = dir.appendingPathComponent("\(assets.lastPathComponent).tar")
            if !FileManager.default.fileExists(atPath: tar.path) {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                let tmp = tar.appendingPathExtension("part")
                let p = Process()
                p.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
                p.arguments = ["-cf", tmp.path, "--exclude", "packages", "-C", assets.deletingLastPathComponent().path, assets.lastPathComponent]
                try p.run(); p.waitUntilExit()
                guard p.terminationStatus == 0 else { throw HTTPError(500, "could not pack the model") }
                try FileManager.default.moveItem(at: tmp, to: tar)
            }
            send(connection, status: 200, body: try Data(contentsOf: tar, options: .alwaysMapped), type: "application/x-tar")
        default:
            throw HTTPError(404, "not found")
        }
    }

    func send(_ connection: NWConnection, status: Int, body: Data, type: String = "application/json") {
        let head = "HTTP/1.1 \(status) \((200..<300).contains(status) ? "OK" : "Error")\r\nContent-Type: \(type)\r\n" +
                   "Content-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        connection.send(content: Data(head.utf8), completion: .contentProcessed { _ in })
        connection.send(content: body, completion: .contentProcessed { _ in connection.cancel() })
    }
}

struct Health: Encodable {
    let status = "ok"
    let model: String
    let vector_space_identity: String
    let engine = "Swift CoreML/ANE"
    let workers: Int
    let stats: Stats.Snapshot
    /// Other models served here. Their identity is under "identity", not "vector_space_identity":
    /// pool peers read the first vector_space_identity in /health, which must stay the pool's.
    let also: [Also]
    /// Vision encoders (ANE Embed 0.2): which text model each feeds, its layouts, and its counters.
    let vision: [VisionInfo]
    let app_version = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "cli"
    let app_build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "cli"

    struct VisionInfo: Encodable {
        let model: String
        let layouts: [String]
        let loaded: [String]
        let default_layout: String
        let images: Int
        let prep_seconds: Double
        let predict_seconds: Double
    }

    struct Also: Encodable {
        let model: String
        let identity: String
        let workers: Int
        let stats: Stats.Snapshot
    }

    init(engine: Engine, extra: [String: Engine] = [:], visions: [String: Vision] = [:]) {
        model = engine.assets.manifest.model_tag
        vector_space_identity = engine.assets.manifest.vector_space_identity
        workers = engine.workerCount
        stats = engine.stats.snapshot()
        also = extra.keys.sorted().map { tag in
            let e = extra[tag]!
            return Also(model: tag, identity: e.assets.manifest.vector_space_identity, workers: e.workerCount, stats: e.stats.snapshot())
        }
        vision = visions.keys.sorted().map { tag in
            let v = visions[tag]!
            return VisionInfo(model: tag, layouts: v.layouts, loaded: v.loadedLayouts, default_layout: v.manifest.default_layout,
                              images: v.images, prep_seconds: v.prepSeconds, predict_seconds: v.predictSeconds)
        }
    }
}
