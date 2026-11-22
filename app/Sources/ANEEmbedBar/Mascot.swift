import SwiftUI

/// Chippy: the Neural Engine as a little chip with a face. Its mood follows the server:
/// busy = wide eyes, glow and sparkles; idle = blinks now and then; stopped = asleep;
/// loading = eyes looking around; failed = x eyes.
struct Mascot: View {
    enum Mood { case busy, idle, asleep, loading, failed }
    let mood: Mood
    var size: CGFloat = 64
    /// False while the panel is closed: no frames are drawn for an animation nobody sees.
    var animating = true
    /// "Super Saiyan" (from the-oracle-voice's porthole): while the ANE is really working,
    /// a golden aura of flames licks out past the chip and rises, the outline burns
    /// white-gold, and a shock ring bursts out as it powers up (0.35 s up, 0.25 s down).
    var superSaiyan = false
    @State private var powerSince = Date.distantPast
    @State private var powered = false

    var body: some View {
        TimelineView(.animation(minimumInterval: 1 / 30, paused: !animating)) { context in
            let t = context.date.timeIntervalSinceReferenceDate
            let since = context.date.timeIntervalSince(powerSince)
            let power = powered ? min(1, since / 0.35) : max(0, 1 - since / 0.25)
            ZStack {
                if power > 0.001 { aura(t: t, power: power) }
                if mood == .busy {
                    RoundedRectangle(cornerRadius: size * 0.3)
                        .fill(Theme.mint.opacity(0.35 + 0.2 * sin(t * 4)))
                        .blur(radius: size * 0.18)
                        .frame(width: size * 1.05, height: size * 1.05)
                }
                pins
                RoundedRectangle(cornerRadius: size * 0.24)
                    .fill(LinearGradient(colors: [Theme.chipTop, Theme.chipBottom], startPoint: .top, endPoint: .bottom))
                    .overlay(RoundedRectangle(cornerRadius: size * 0.24).stroke(.white.opacity(0.18), lineWidth: 1))
                    .frame(width: size * 0.78, height: size * 0.78)
                    .overlay(RoundedRectangle(cornerRadius: size * 0.24)
                        .stroke(Color(red: 1.0, green: 0.96, blue: 0.7).opacity(0.9 * power), lineWidth: 2.5)
                        .blur(radius: 1.5))
                face(t: t)
                if mood == .busy { sparkles(t: t) }
                if mood == .asleep { sleepy(t: t) }
            }
            .frame(width: size, height: size)
            .offset(y: mood == .busy ? CGFloat(sin(t * 8)) * 1.5 : 0)
        }
        .onChange(of: superSaiyan, initial: true) { _, on in
            guard on != powered else { return }
            powered = on
            powerSince = Date()
        }
    }

    /// Flame tongues around the chip, longer at the top and flickering; a shock ring on power-up.
    private func aura(t: Double, power p: Double) -> some View {
        Canvas { ctx, sz in
            let c = CGPoint(x: sz.width / 2, y: sz.height / 2)
            let rim = size * 0.40
            let tongues = 36
            for i in 0..<tongues {
                let a = Double(i) / Double(tongues) * 2 * .pi
                let up = 0.5 - 0.5 * sin(a)                                  // 1 at the top (y grows down)
                let n = 0.5 + 0.5 * sin(a * 7 + t * 9) * sin(a * 13 - t * 6.5 + 1.3)
                let n2 = 0.5 + 0.5 * sin(a * 23 - t * 14)
                let len = size * (0.07 + 0.17 * n + 0.05 * n2) * (0.55 + 0.65 * up) * p
                let w = 2 * .pi / Double(tongues) * 0.9
                func pt(_ ang: Double, _ r: Double) -> CGPoint { CGPoint(x: c.x + cos(ang) * r, y: c.y + sin(ang) * r) }
                var path = Path()
                path.move(to: pt(a - w, rim))
                path.addQuadCurve(to: pt(a + sin(t * 5 + Double(i)) * 0.06, rim + len), control: pt(a - w * 0.3, rim + len * 0.6))
                path.addQuadCurve(to: pt(a + w, rim), control: pt(a + w * 0.3, rim + len * 0.6))
                path.closeSubpath()
                ctx.fill(path, with: .radialGradient(
                    Gradient(colors: [Color(red: 1.0, green: 0.97, blue: 0.75), Color(red: 1.0, green: 0.55, blue: 0.12).opacity(0.0)]),
                    center: c, startRadius: rim, endRadius: rim + len))
            }
            let burst = rim + size * 0.28 * p
            ctx.stroke(Path(ellipseIn: CGRect(x: c.x - burst, y: c.y - burst, width: burst * 2, height: burst * 2)),
                       with: .color(Color(red: 1.0, green: 0.92, blue: 0.6).opacity(min(1, (1 - p) * p * 3))), lineWidth: 2)
        }
        .frame(width: size * 1.7, height: size * 1.7)
        .blur(radius: 0.8)
        .allowsHitTesting(false)
    }

    private var pins: some View {
        let pin = RoundedRectangle(cornerRadius: 1).fill(Theme.pin)
        return ZStack {
            ForEach(0..<4) { side in
                HStack(spacing: size * 0.09) {
                    ForEach(0..<3) { _ in pin.frame(width: size * 0.06, height: size * 0.12) }
                }
                .offset(y: -size * 0.44)
                .rotationEffect(.degrees(Double(side) * 90))
            }
        }
    }

    private func face(t: Double) -> some View {
        let eyeGap = size * 0.2
        let blinking = mood == .idle && t.truncatingRemainder(dividingBy: 3.2) < 0.14
        let look: CGFloat = mood == .loading ? CGFloat(sin(t * 3)) * size * 0.05 : 0
        return VStack(spacing: size * 0.07) {
            HStack(spacing: eyeGap) { eye(blinking: blinking); eye(blinking: blinking) }
                .offset(x: look)
            mouth
        }
        .offset(y: size * 0.02)
    }

    @ViewBuilder private func eye(blinking: Bool) -> some View {
        switch mood {
        case .failed:
            Image(systemName: "xmark").font(.system(size: size * 0.13, weight: .heavy)).foregroundStyle(Theme.ink)
        case .asleep:
            Capsule().fill(Theme.ink).frame(width: size * 0.13, height: size * 0.035)
        default:
            Capsule().fill(Theme.ink)
                .frame(width: size * 0.1, height: blinking ? size * 0.025 : size * (mood == .busy ? 0.15 : 0.12))
                .overlay(alignment: .topTrailing) {
                    if !blinking { Circle().fill(.white).frame(width: size * 0.035).offset(x: -size * 0.01, y: size * 0.015) }
                }
        }
    }

    @ViewBuilder private var mouth: some View {
        switch mood {
        case .busy: Capsule().fill(Theme.ink).frame(width: size * 0.12, height: size * 0.07)
        case .failed: Capsule().fill(Theme.ink).frame(width: size * 0.12, height: size * 0.025)
        default:
            Circle().trim(from: 0.1, to: 0.4).stroke(Theme.ink, style: StrokeStyle(lineWidth: size * 0.03, lineCap: .round))
                .frame(width: size * 0.16, height: size * 0.16).offset(y: -size * 0.06)
        }
    }

    private func sparkles(t: Double) -> some View {
        ZStack {
            ForEach(0..<3) { i in
                let phase = (t * 0.9 + Double(i) / 3).truncatingRemainder(dividingBy: 1)
                Image(systemName: "sparkle")
                    .font(.system(size: size * 0.16, weight: .bold))
                    .foregroundStyle(Theme.mint)
                    .opacity(1 - phase)
                    .offset(x: size * [0.42, -0.44, 0.3][i], y: -size * (0.25 + phase * 0.3) + size * [0, 0.2, 0.45][i])
            }
        }
    }

    private func sleepy(t: Double) -> some View {
        ZStack {
            ForEach(0..<2) { i in
                let phase = (t * 0.4 + Double(i) / 2).truncatingRemainder(dividingBy: 1)
                Text("z").font(.system(size: size * (0.14 + 0.08 * phase), weight: .heavy, design: .rounded))
                    .foregroundStyle(Theme.muted).opacity(1 - phase)
                    .offset(x: size * (0.35 + 0.15 * phase), y: -size * (0.25 + 0.3 * phase))
            }
        }
    }
}
