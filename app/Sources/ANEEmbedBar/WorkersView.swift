import ANEEmbedHTTP
import AppKit
import SwiftUI

/// The pool: this Mac plus any remote workers, each with its weight and live rate.
/// Adding checks the remote serves the same vector space; removing is one click.
struct WorkersView: View {
    @ObservedObject var model: AppModel
    @State private var url = "http://<worker-host>:11436"
    @State private var weight = 77.0

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Workers").font(.system(size: 11, weight: .semibold)).foregroundStyle(Theme.muted)
                Spacer()
                Text("\(model.workerList.count) in pool").font(.system(size: 10)).foregroundStyle(Theme.muted)
            }
            ForEach(model.workerList, id: \.id) { w in
                HStack(spacing: 8) {
                    Circle().fill(w.lastError == nil ? (w.inFlight > 0 ? Theme.lemon : Theme.mint) : Theme.peach).frame(width: 7, height: 7)
                    Image(systemName: w.url == nil ? "laptopcomputer" : (w.id.hasPrefix("pull:") ? "hand.raised.fill" : "network"))
                        .font(.system(size: 10)).foregroundStyle(Theme.muted)
                    Text(w.url.map { $0.replacingOccurrences(of: "http://", with: "").replacingOccurrences(of: "pull:", with: "") + ($0.hasPrefix("pull:") ? " (volunteer)" : "") } ?? "this Mac")
                        .font(.system(size: 11, design: .monospaced)).lineLimit(1)
                    Spacer(minLength: 4)
                    Text(String(format: "%.0f/s", w.textsPerSecond)).font(.system(size: 11, design: .rounded)).monospacedDigit()
                    Text("w \(Int(w.weight))").font(.system(size: 10)).foregroundStyle(Theme.muted)
                    if w.url != nil {
                        Button { model.removeWorker(id: w.id) } label: { Image(systemName: "xmark.circle.fill") }
                            .buttonStyle(.plain).foregroundStyle(Theme.muted).help("Remove from pool")
                    }
                }
            }
            HStack(spacing: 6) {
                TextField("http://host:11436", text: $url).textFieldStyle(.roundedBorder).font(.system(size: 11))
                TextField("weight", value: $weight, format: .number).textFieldStyle(.roundedBorder).frame(width: 46).font(.system(size: 11))
                Button { model.addWorker(url: url, weight: weight) } label: { Image(systemName: "plus") }
                    .help("Add worker (must serve the same vector space)")
            }
            if let join = model.joinCommand {
                Button {
                    NSPasteboard.general.clearContents(); NSPasteboard.general.setString(join, forType: .string)
                    model.workerMessage = "join command copied: run it in Terminal on the other Mac"
                } label: {
                    Label("Copy join command for another Mac", systemImage: "person.badge.plus").font(.system(size: 11))
                }
                .buttonStyle(.plain).foregroundStyle(Theme.mint).help(join)
            }
            if let message = model.workerMessage {
                Text(message).font(.system(size: 10)).foregroundStyle(Theme.muted).lineLimit(3)
            }
        }
        .padding(10)
        .background(Theme.card, in: RoundedRectangle(cornerRadius: 14))
    }
}
