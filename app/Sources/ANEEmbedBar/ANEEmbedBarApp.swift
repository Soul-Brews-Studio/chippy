import AppKit
import Charts
import SwiftUI

/// Menu bar app: serves embeddinggemma:ane-w8a16 on the ANE at /api/embed and shows
/// what it is doing. No Dock icon (accessory app).
@main
struct ANEEmbedBarApp: App {
    @StateObject private var model = AppModel()

    init() { NSApplication.shared.setActivationPolicy(.accessory) }

    var body: some Scene {
        MenuBarExtra {
            StatusView(model: model)
                .onAppear { model.panelOpen = true }
                .onDisappear { model.panelOpen = false }
        } label: {
            Label(model.title, systemImage: model.statusSymbol).labelStyle(.titleAndIcon).monospacedDigit()
        }
        .menuBarExtraStyle(.window)
    }
}

extension AppModel {
    var isBusy: Bool { if case .serving = phase { return stats.textsPerSecond >= 1 } else { return false } }

    var mood: Mascot.Mood {
        switch phase {
        case .serving: return isBusy ? .busy : .idle
        case .loading: return .loading
        case .failed: return .failed
        case .stopped: return .asleep
        }
    }

    var moodLine: String {
        switch mood {
        case .busy: return "Crunching vectors on the Neural Engine"
        case .idle: return "Awake and waiting for texts"
        case .loading: return "Waking up… loading models"
        case .failed: return "Something went wrong"
        case .asleep: return "Sleeping. Press Start to wake me"
        }
    }
}

struct StatusView: View {
    @ObservedObject var model: AppModel
    @State private var showSettings = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            hero
            NowView(stats: model.stats)
            ResourcesView(system: model.system, appCPU: model.cpuPercent, appMemoryMB: model.memoryMB)
            cards
            WorkersView(model: model)
            RecentRequestsView(requests: Array(model.stats.lastRequests.prefix(4)))
            if case .failed(let why) = model.phase {
                Text(why).font(.caption).foregroundStyle(Theme.peach).lineLimit(4)
            }
            if let vision = model.visionStatus {
                Label(vision, systemImage: "photo").font(.caption).foregroundStyle(Theme.muted).lineLimit(2)
            }
            DisclosureGroup(isExpanded: $showSettings) { settings.padding(.top, 6) } label: {
                Label("Settings", systemImage: "slider.horizontal.3").font(.callout)
            }
            controls
        }
        .padding(16)
        .frame(width: 360)
        .background(LinearGradient(colors: [Color(red: 0.09, green: 0.11, blue: 0.16), Color(red: 0.06, green: 0.07, blue: 0.10)],
                                   startPoint: .top, endPoint: .bottom))
    }

    private var header: some View {
        HStack(spacing: 12) {
            Mascot(mood: model.mood, size: 58, animating: model.panelOpen, superSaiyan: model.superSaiyan)
            VStack(alignment: .leading, spacing: 3) {
                HStack(alignment: .firstTextBaseline, spacing: 6) {
                    Text("Chippy").font(.system(size: 18, weight: .bold, design: .rounded))
                    Text("v" + (Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?")
                         + " (" + (Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?") + ")")
                        .font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.muted)
                }
                Text(model.moodLine).font(.system(size: 11.5)).foregroundStyle(Theme.muted)
                if case .serving(let url) = model.phase {
                    Button { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(url + "/api/embed", forType: .string) } label: {
                        HStack(spacing: 5) {
                            Circle().fill(Theme.mint).frame(width: 6, height: 6)
                            Text(url.replacingOccurrences(of: "http://", with: "")).font(.system(size: 11, design: .monospaced))
                            Image(systemName: "doc.on.doc").font(.system(size: 9))
                        }
                        .foregroundStyle(Theme.mint)
                        .padding(.horizontal, 8).padding(.vertical, 3)
                        .background(Theme.mint.opacity(0.12), in: Capsule())
                    }
                    .buttonStyle(.plain).help("Copy the /api/embed URL")
                }
            }
            Spacer(minLength: 0)
        }
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text(String(format: "%.0f", model.stats.textsPerSecond))
                    .font(.system(size: 40, weight: .heavy, design: .rounded)).monospacedDigit()
                    .contentTransition(.numericText())
                    .animation(.snappy, value: model.stats.textsPerSecond)
                Text("texts/s").font(.system(size: 13, weight: .medium)).foregroundStyle(Theme.muted)
                Spacer()
                Text("peak \(Int(model.peak))").font(.system(size: 11, design: .rounded)).foregroundStyle(Theme.muted)
            }
            Chart(Array(model.history.enumerated()), id: \.offset) { point in
                AreaMark(x: .value("s", point.offset), y: .value("texts/s", point.element))
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(LinearGradient(colors: [Theme.mint.opacity(0.55), Theme.mint.opacity(0.02)], startPoint: .top, endPoint: .bottom))
                LineMark(x: .value("s", point.offset), y: .value("texts/s", point.element))
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(Theme.mint)
                    .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round))
            }
            .chartXAxis(.hidden).chartYAxis(.hidden)
            .chartYScale(domain: 0...max(10, model.history.max() ?? 0))
            .frame(height: 54)
            Text("last 60 seconds").font(.system(size: 10)).foregroundStyle(Theme.muted)
        }
        .padding(12)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 16))
    }

    private var cards: some View {
        let s = model.stats
        let padding = s.slots > 0 ? 1 - Double(s.tokens) / Double(s.slots) : 0
        return Grid(horizontalSpacing: 8, verticalSpacing: 8) {
            GridRow {
                StatCard(symbol: "text.bubble.fill", tint: Theme.sky, value: s.texts.formatted(), caption: "texts embedded")
                StatCard(symbol: "arrow.left.arrow.right", tint: Theme.lemon, value: s.requests.formatted(),
                         caption: s.errors == 0 ? "requests · no errors" : "requests · \(s.errors) errors")
            }
            GridRow {
                StatCard(symbol: "number", tint: Theme.sky, value: s.tokens.formatted(), caption: "tokens")
                StatCard(symbol: "square.stack.3d.up.fill", tint: Theme.peach,
                         value: s.slots > 0 ? String(format: "%.0f%%", 100 * padding) : "–", caption: "padding") {
                    Ring(fraction: padding, tint: Theme.peach)
                }
            }
        }
    }

    private var settings: some View {
        Grid(alignment: .leading, horizontalSpacing: 10, verticalSpacing: 8) {
            GridRow {
                Text("Host").foregroundStyle(Theme.muted)
                TextField("127.0.0.1", text: $model.host).textFieldStyle(.roundedBorder).frame(width: 150)
            }
            GridRow {
                Text("Port").foregroundStyle(Theme.muted)
                TextField("11435", value: $model.port, format: .number.grouping(.never)).textFieldStyle(.roundedBorder).frame(width: 80)
            }
            GridRow {
                Text("Workers").foregroundStyle(Theme.muted)
                Stepper("\(model.workers)", value: $model.workers, in: 1...4)
            }
            GridRow {
                Text("Popup").foregroundStyle(Theme.muted)
                Toggle("Super Saiyan Chippy on screen while the ANE works", isOn: $model.saiyanPopup)
                    .toggleStyle(.switch).controlSize(.mini).font(.system(size: 11))
            }
            Text("A non-loopback host must be this Mac's Tailscale address (100.64.0.0/10). Restart to apply.")
                .font(.system(size: 10)).foregroundStyle(Theme.muted).gridCellColumns(2)
        }
        .font(.callout)
    }

    private var controls: some View {
        let serving: Bool = { if case .serving = model.phase { return true } else { return false } }()
        return HStack(spacing: 8) {
            Button { model.restart() } label: {
                Label(serving ? "Restart" : "Start", systemImage: serving ? "arrow.clockwise" : "play.fill")
            }
            .buttonStyle(.borderedProminent).tint(Theme.mint.opacity(0.8))
            Button { model.stop() } label: { Label("Sleep", systemImage: "moon.zzz.fill") }
                .disabled(!serving)
            Spacer()
            Button { NSApplication.shared.terminate(nil) } label: { Image(systemName: "power") }
                .help("Quit")
        }
        .controlSize(.regular)
    }
}
