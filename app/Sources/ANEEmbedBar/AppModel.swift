import ANEEmbedCore
import ANEEmbedHTTP
import Foundation

/// The app's state: settings, the engine and server it owns, and a 1 s poll of the
/// counters (plus ANE watts when a mactop exporter is listening on :9101).
@MainActor
final class AppModel: ObservableObject {
    enum Phase: Equatable { case stopped, loading, serving(String), failed(String) }

    @Published var phase: Phase = .stopped
    @Published var stats = Stats.Snapshot()
    @Published var aneWatts: Double?
    /// The last 60 s of texts/s, one sample per second, oldest first.
    @Published var history: [Double] = Array(repeating: 0, count: 60)
    @Published var peak: Double = 0
    @Published var cpuPercent: Double?
    @Published var memoryMB: Double?
    private var sampler = ResourceSampler()
    @Published var system = SystemMonitor.Reading()
    @Published var workerList: [WorkerPool.Info] = []
    @Published var workerMessage: String?
    private var monitor = SystemMonitor()
    @Published var port: Int = UserDefaults.standard.object(forKey: "port") as? Int ?? 11435 {
        didSet { UserDefaults.standard.set(port, forKey: "port") }
    }
    @Published var workers: Int = UserDefaults.standard.object(forKey: "workers") as? Int ?? 2 {
        didSet { UserDefaults.standard.set(workers, forKey: "workers") }
    }
    @Published var host: String = UserDefaults.standard.string(forKey: "host") ?? "127.0.0.1" {
        didSet { UserDefaults.standard.set(host, forKey: "host") }
    }

    /// On-screen Super Saiyan Chippy while the ANE works (click-through). Off: Settings toggle.
    @Published var saiyanPopup: Bool = UserDefaults.standard.object(forKey: "saiyanPopup") as? Bool ?? true {
        didSet { UserDefaults.standard.set(saiyanPopup, forKey: "saiyanPopup"); updatePopup() }
    }
    private let popup = SaiyanPopup()
    private var popupShown = false

    private var engine: Engine?
    private var v2Engine: Engine?
    @Published var v2Error: String?
    /// Images (0.2): the vision encoder on the ANE beside EmbeddingGemma 2, when its assets are exported.
    @Published var visionStatus: String?
    /// The one line another Mac runs to join this pool (nil: no tailscale address).
    @Published var joinCommand: String?
    private var vision: Vision?
    private var server: EmbedServer?
    private var poll: Timer?
    /// The panel is open. Closed, the model polls every 5 s (enough for the menu bar title)
    /// instead of every second, and the mascot stops animating: idle Chippy was using 14% CPU
    /// on SwiftUI redraws nobody could see.
    @Published var panelOpen = false {
        didSet { if panelOpen != oldValue { schedulePoll(); Task { await refresh() } } }
    }
    /// Held while serving: a menu bar app is otherwise eligible for App Nap, which
    /// throttles its CPU and timers and stretches every request.
    private var activity: NSObjectProtocol?

    init() {
        schedulePoll()
        if UserDefaults.standard.object(forKey: "autostart") as? Bool ?? true { start() }
    }

    /// ANE utilization from IOReport, any process (an LLM, the PM2.5 models, ...), or nil.
    var aneUtilization: Double? { system.aneUtilization }

    /// The ANE is really working, by us or anyone: embedding, or IOReport shows it busy AND moving
    /// data. Utilization alone can read 100% at idle (seen with powermetrics sampling), while an
    /// idle ANE sits at 0-4 GB/s, so bandwidth above 8 GB/s is required too.
    var superSaiyan: Bool {
        guard case .serving = phase else { return false }
        return stats.textsPerSecond >= 1 || ((aneUtilization ?? 0) >= 20 && (system.aneBandwidthGBs ?? 0) > 8)
    }

    /// Menu bar text: texts/s while embedding, else the ANE's utilization when something else
    /// keeps it busy, else just "ANE".
    var title: String {
        switch phase {
        case .serving:
            if stats.textsPerSecond >= 1 { return "\(Int(stats.textsPerSecond))/s" }
            if let u = aneUtilization, u >= 5 {
                let g = system.aneBandwidthGBs.map { " · \(Int($0.rounded())) GB/s" } ?? ""
                return "ANE \(Int(u.rounded()))%\(g)"
            }
            return "ANE"
        case .loading: return "ANE…"
        case .failed: return "ANE!"
        case .stopped: return "ANE"
        }
    }

    /// Menu bar icon: the state at a glance.
    var statusSymbol: String {
        switch phase {
        case .serving:
            if superSaiyan { return "flame.fill" }
            let busy = stats.textsPerSecond >= 1 || (aneUtilization ?? 0) >= 5
            return busy ? "cpu.fill" : "cpu"
        case .loading: return "hourglass"
        case .failed: return "exclamationmark.triangle.fill"
        case .stopped: return "moon.zzz"
        }
    }

    func start() {
        guard phase == .stopped || { if case .failed = phase { return true } else { return false } }() else { return }
        phase = .loading
        let (host, port, workers) = (host, UInt16(port), workers)
        Task {
            do {
                let engine: Engine
                if let existing = self.engine, existing.workerCount == workers { engine = existing }
                else { engine = try await Engine(assets: try Assets(), workers: workers) }
                let allow = host != "127.0.0.1"
                // volunteers (0.3): also listen on this Mac's tailscale address for the join kit and the
                // work endpoints only, so other Macs can join while the embed API stays on loopback
                let join = UserDefaults.standard.object(forKey: "acceptVolunteers") as? Bool ?? true
                let server = try EmbedServer(engine: engine, host: host, port: port, allowNetwork: allow,
                                             joinHosts: join ? EmbedServer.joinAddresses(lan: UserDefaults.standard.object(forKey: "volunteersOnLAN") as? Bool ?? true) : [])
                self.joinCommand = server.joinCommand
                try server.start()
                // EmbeddingGemma 2 beside v1 when its assets are exported, attached once it has compiled
                // for the ANE (v1 serves meanwhile): requests naming its tag go to it; everything else,
                // the pool included, stays on v1 until stored vectors are re-embedded.
                let v2Root = Assets.root(named: "embeddinggemma2-w16")
                if FileManager.default.fileExists(atPath: v2Root.appendingPathComponent("manifest.json").path) {
                    let cached = self.v2Engine
                    Task {
                        do {
                            let v2: Engine
                            if let cached { v2 = cached } else { v2 = try await Engine(assets: try Assets(root: v2Root), workers: 1) }
                            self.v2Engine = v2
                            server.addModel(v2)
                            // images for EmbeddingGemma 2 (0.2): its vision encoder on the ANE, compiled now
                            // so the first image request does not wait for it
                            if FileManager.default.fileExists(atPath: Vision.defaultRoot.appendingPathComponent("manifest.json").path) {
                                let vision = try self.vision ?? Vision()
                                self.vision = vision
                                server.attachVision(vision)
                                let layout = vision.manifest.default_layout
                                self.visionStatus = "images: compiling vision \(layout) for the ANE…"
                                _ = try await vision.model(layout)
                                self.visionStatus = "images: vision \(layout) on the ANE"
                            }
                        } catch { self.v2Error = String(describing: error) }
                    }
                }
                self.engine = engine
                self.server = server
                self.phase = .serving("http://\(host):\(port)")
                self.activity = ProcessInfo.processInfo.beginActivity(
                    options: [.userInitiatedAllowingIdleSystemSleep, .latencyCritical], reason: "Serving embeddings on the ANE")
            } catch {
                self.phase = .failed(String(describing: error))
            }
        }
    }

    func stop() {
        server?.stop()
        server = nil
        if let activity { ProcessInfo.processInfo.endActivity(activity) }
        activity = nil
        phase = .stopped
    }

    func restart() { stop(); start() }

    private func schedulePoll() {
        poll?.invalidate()
        // closed: every 2 s, so the on-screen popup reacts within ~2 s (was 5 s before the popup)
        poll = Timer.scheduledTimer(withTimeInterval: panelOpen || popupShown ? 1 : 2, repeats: true) { [weak self] _ in
            Task { @MainActor in await self?.refresh() }
        }
    }

    func addWorker(url: String, weight: Double) {
        guard let pool = server?.pool else { workerMessage = "start the server first"; return }
        Task {
            do { _ = try await pool.add(url: url, weight: weight); workerMessage = "added \(url)" }
            catch { workerMessage = "\(error)" }
            workerList = pool.list()
        }
    }

    func removeWorker(id: String) {
        guard let pool = server?.pool else { return }
        do { try pool.remove(id: id); workerMessage = nil } catch { workerMessage = "\(error)" }
        workerList = pool.list()
    }

    private func refresh() async {
        if let engine { stats = engine.stats.snapshot(window: 2) }
        history = Array((history + [stats.textsPerSecond]).suffix(60))
        peak = max(peak, stats.textsPerSecond)
        (cpuPercent, memoryMB) = sampler.sample()
        workerList = server?.pool.list() ?? []
        var monitor = self.monitor
        system = await monitor.read()
        self.monitor = monitor
        aneWatts = system.aneWatts
        updatePopup()
    }

    private func updatePopup() {
        let on = saiyanPopup && superSaiyan
        popup.update(on: on, util: aneUtilization, gbs: system.aneBandwidthGBs, textsPerSecond: stats.textsPerSecond)
        if on != popupShown { popupShown = on; schedulePoll() }   // 1 s readout while it is up
    }

    nonisolated static func readAneWatts() async -> Double? {
        guard let url = URL(string: "http://127.0.0.1:9101/metrics") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 0.5
        guard let (data, _) = try? await URLSession.shared.data(for: request) else { return nil }
        for line in String(decoding: data, as: UTF8.self).split(separator: "\n")
        where line.hasPrefix("mactop_power_watts{component=\"ane\"}") {
            return Double(line.split(separator: " ").last ?? "")
        }
        return nil
    }
}
