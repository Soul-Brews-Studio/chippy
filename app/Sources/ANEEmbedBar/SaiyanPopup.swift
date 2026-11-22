import AppKit
import SwiftUI

/// A floating Chippy that powers up "Super Saiyan" on screen while the ANE is really working,
/// whoever is using it (our embedder, an LLM, the PM2.5 models, ...), and powers down and
/// leaves when it stops. Click-through everywhere: the panel ignores every mouse event, so it
/// never blocks what is behind it. Same panel recipe as the-oracle-voice's porthole popup.
@MainActor
final class SaiyanPopup {
    private let state = PopupState()
    private var panel: NSPanel?
    private var hideTask: Task<Void, Never>?

    func update(on: Bool, util: Double?, gbs: Double?, textsPerSecond: Double) {
        state.util = util; state.gbs = gbs; state.tps = textsPerSecond
        if on {
            hideTask?.cancel()
            hideTask = nil
            show()
            state.on = true
        } else if state.on {
            state.on = false
            // let the power-down play (0.25 s) and the fade finish before the window goes
            hideTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(1.2))
                guard !Task.isCancelled else { return }
                self?.panel?.orderOut(nil)
            }
        }
    }

    private func show() {
        let panel = self.panel ?? make()
        self.panel = panel
        place(panel)
        panel.orderFrontRegardless()
    }

    private func make() -> NSPanel {
        let panel = NSPanel(contentRect: NSRect(x: 0, y: 0, width: 170, height: 200),
                            styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.level = .statusBar
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.ignoresMouseEvents = true          // click-through: never blocks the click behind
        panel.contentView = NSHostingView(rootView: PopupView(state: state))
        return panel
    }

    /// Top-right of the screen, just under the menu bar, near Chippy's own status item.
    private func place(_ panel: NSPanel) {
        guard let screen = NSScreen.main?.visibleFrame else { return }
        let size = panel.frame.size
        panel.setFrameOrigin(NSPoint(x: screen.maxX - size.width - 24, y: screen.maxY - size.height - 8))
    }
}

@MainActor
final class PopupState: ObservableObject {
    @Published var on = false
    @Published var util: Double?
    @Published var gbs: Double?
    @Published var tps: Double = 0
}

private struct PopupView: View {
    @ObservedObject var state: PopupState
    private let gold = Color(red: 1.0, green: 0.92, blue: 0.6)

    private func readout(_ value: String, _ unit: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 2) {
            Text(value).font(.system(size: 22, weight: .heavy, design: .rounded)).monospacedDigit()
                .contentTransition(.numericText())
            Text(unit).font(.system(size: 10, weight: .bold, design: .rounded)).opacity(0.75)
        }
        .foregroundStyle(gold)
    }

    var body: some View {
        VStack(spacing: 6) {
            Mascot(mood: .busy, size: 84, animating: true, superSaiyan: state.on)
                .frame(width: 150, height: 150)
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                readout(state.util.map { String(format: "%.0f", $0) } ?? "–", "%")
                readout(state.gbs.map { String(format: "%.0f", $0) } ?? "–", "GB/s")
            }
            .padding(.horizontal, 10).padding(.vertical, 4)
            .background(.black.opacity(0.6), in: Capsule())
            if state.tps >= 1 {
                Text("\(Int(state.tps)) texts/s")
                    .font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundStyle(gold.opacity(0.8))
            }
        }
        .frame(width: 170, height: 200)
        .animation(.easeOut(duration: 0.3), value: state.util)
        .opacity(state.on ? 1 : 0)
        .animation(.easeOut(duration: state.on ? 0.2 : 0.9), value: state.on)
    }
}
