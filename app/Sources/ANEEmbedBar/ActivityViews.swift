import ANEEmbedCore
import SwiftUI

/// "Now": requests in flight and one light per ANE worker (lit while it runs a call).
struct NowView: View {
    let stats: Stats.Snapshot

    var body: some View {
        HStack(spacing: 14) {
            VStack(alignment: .leading, spacing: 2) {
                Text("\(stats.inFlight)").font(.system(size: 22, weight: .bold, design: .rounded)).monospacedDigit()
                    .foregroundStyle(stats.inFlight > 0 ? Theme.lemon : .primary)
                Text(stats.inFlight == 1 ? "request now" : "requests now").font(.system(size: 10.5)).foregroundStyle(Theme.muted)
            }
            Divider().frame(height: 30)
            VStack(alignment: .leading, spacing: 5) {
                HStack(spacing: 6) {
                    ForEach(Array(stats.busyWorkers.enumerated()), id: \.offset) { _, busy in
                        RoundedRectangle(cornerRadius: 4)
                            .fill(busy ? Theme.mint : Theme.card)
                            .overlay(RoundedRectangle(cornerRadius: 4).stroke(Theme.mint.opacity(0.5), lineWidth: 1))
                            .frame(width: 22, height: 12)
                            .shadow(color: busy ? Theme.mint.opacity(0.8) : .clear, radius: 4)
                            .animation(.easeOut(duration: 0.15), value: busy)
                    }
                }
                Text("\(stats.busyWorkers.filter { $0 }.count) of \(stats.busyWorkers.count) ANE workers busy")
                    .font(.system(size: 10.5)).foregroundStyle(Theme.muted)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 14))
    }
}

/// This Mac (CPU, RAM, GPU, ANE) on one row; this app's own CPU and memory below it.
struct ResourcesView: View {
    let system: SystemMonitor.Reading
    let appCPU: Double?
    let appMemoryMB: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("This Mac").font(.system(size: 11, weight: .semibold)).foregroundStyle(Theme.muted)
            HStack(spacing: 6) {
                gauge("cpu", Theme.sky, pct(system.cpuPercent), watts(system.cpuWatts, "CPU"), fraction: (system.cpuPercent ?? 0) / 100)
                gauge("memorychip", Theme.peach, system.ramUsedGB.map { String(format: "%.0f/%.0f GB", $0, system.ramTotalGB) } ?? "–", "RAM",
                      fraction: (system.ramUsedGB ?? 0) / system.ramTotalGB)
                gauge("square.grid.3x3.fill", Theme.lemon, pct(system.gpuPercent), watts(system.gpuWatts, "GPU"), fraction: (system.gpuPercent ?? 0) / 100)
                gauge("brain", Theme.mint, system.aneUtilization.map { String(format: "%.0f%%", $0) } ?? "–",
                      system.aneBandwidthGBs.map { String(format: "ANE %.1f GB/s", $0) } ?? "ANE",
                      fraction: (system.aneUtilization ?? 0) / 100)
            }
            HStack(spacing: 12) {
                Label("Chippy uses " + (appCPU.map { String(format: "%.0f%% CPU", $0) } ?? "– CPU"), systemImage: "gauge.with.dots.needle.33percent")
                Label(appMemoryMB.map { $0 >= 1024 ? String(format: "%.1f GB", $0 / 1024) : String(format: "%.0f MB", $0) } ?? "–", systemImage: "memorychip")
            }
            .font(.system(size: 10.5)).foregroundStyle(Theme.muted)
        }
        .padding(10)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 14))
    }

    private func pct(_ v: Double?) -> String { v.map { String(format: "%.0f%%", $0) } ?? "–" }
    private func watts(_ w: Double?, _ name: String) -> String { w.map { String(format: "%@ %.1f W", name, $0) } ?? name }

    private func gauge(_ symbol: String, _ tint: Color, _ value: String, _ caption: String, fraction: Double) -> some View {
        VStack(spacing: 4) {
            ZStack {
                Ring(fraction: fraction, tint: tint).scaleEffect(1.4)
                Image(systemName: symbol).font(.system(size: 10, weight: .semibold)).foregroundStyle(tint)
            }
            .frame(height: 32)
            Text(value).font(.system(size: 11.5, weight: .semibold, design: .rounded)).monospacedDigit().lineLimit(1).minimumScaleFactor(0.7)
            Text(caption).font(.system(size: 9)).foregroundStyle(Theme.muted).lineLimit(1).minimumScaleFactor(0.7)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
        .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 10))
    }
}

/// The last requests, newest first: when, how many texts and tokens, how long.
struct RecentRequestsView: View {
    let requests: [Stats.RequestLog]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Recent requests").font(.system(size: 11, weight: .semibold)).foregroundStyle(Theme.muted)
            if requests.isEmpty {
                Text("none yet — POST /api/embed to feed Chippy").font(.system(size: 11)).foregroundStyle(Theme.muted)
            }
            ForEach(requests) { r in
                HStack(spacing: 8) {
                    Text(r.at, format: .dateTime.hour().minute().second()).foregroundStyle(Theme.muted)
                    Text("\(r.texts) texts").frame(width: 70, alignment: .leading)
                    Text("\(r.tokens.formatted()) tok").foregroundStyle(Theme.muted).frame(width: 74, alignment: .leading)
                    Spacer(minLength: 0)
                    Text(r.milliseconds >= 1000 ? String(format: "%.1f s", r.milliseconds / 1000) : String(format: "%.0f ms", r.milliseconds))
                        .foregroundStyle(r.milliseconds > 3000 ? Theme.peach : Theme.mint)
                }
                .font(.system(size: 11, design: .monospaced))
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 14))
    }
}
