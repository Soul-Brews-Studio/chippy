import SwiftUI

enum Theme {
    static let mint = Color(red: 0.42, green: 0.93, blue: 0.75)
    static let sky = Color(red: 0.45, green: 0.72, blue: 1.0)
    static let peach = Color(red: 1.0, green: 0.66, blue: 0.55)
    static let lemon = Color(red: 1.0, green: 0.86, blue: 0.45)
    static let chipTop = Color(red: 0.62, green: 0.98, blue: 0.84)
    static let chipBottom = Color(red: 0.36, green: 0.80, blue: 0.86)
    static let pin = Color(red: 0.55, green: 0.62, blue: 0.72)
    static let ink = Color(red: 0.10, green: 0.14, blue: 0.20)
    static let card = Color.white.opacity(0.06)
    static let muted = Color.white.opacity(0.55)
}

/// A rounded tile with an SF symbol, a big value and a caption.
struct StatCard<Accessory: View>: View {
    let symbol: String
    let tint: Color
    let value: String
    let caption: String
    @ViewBuilder var accessory: () -> Accessory

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: symbol)
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(tint)
                .frame(width: 28, height: 28)
                .background(tint.opacity(0.16), in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 1) {
                Text(value).font(.system(size: 15, weight: .semibold, design: .rounded)).monospacedDigit()
                Text(caption).font(.system(size: 10.5)).foregroundStyle(Theme.muted)
            }
            Spacer(minLength: 0)
            accessory()
        }
        .padding(10)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 14))
    }
}

extension StatCard where Accessory == EmptyView {
    init(symbol: String, tint: Color, value: String, caption: String) {
        self.init(symbol: symbol, tint: tint, value: value, caption: caption) { EmptyView() }
    }
}

/// A small ring: fraction filled in tint.
struct Ring: View {
    let fraction: Double
    let tint: Color
    var body: some View {
        ZStack {
            Circle().stroke(Theme.card, lineWidth: 4)
            Circle().trim(from: 0, to: max(0, min(1, fraction)))
                .stroke(tint, style: StrokeStyle(lineWidth: 4, lineCap: .round))
                .rotationEffect(.degrees(-90))
        }
        .frame(width: 22, height: 22)
    }
}
