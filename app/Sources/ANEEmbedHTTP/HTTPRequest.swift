import Foundation

/// A complete HTTP/1.1 request parsed from the bytes received so far, or nil while the
/// headers or the Content-Length body are still incomplete.
struct HTTPRequest {
    let method: String
    let path: String
    let query: [String: String]
    let headers: [String: String]
    let body: Data

    init?(_ buffer: Data) {
        guard let end = buffer.range(of: Data("\r\n\r\n".utf8)) else { return nil }
        let head = String(decoding: buffer[..<end.lowerBound], as: UTF8.self)
        let lines = head.components(separatedBy: "\r\n")
        let parts = lines.first?.split(separator: " ") ?? []
        guard parts.count >= 2 else { return nil }
        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            let pair = line.split(separator: ":", maxSplits: 1)
            if pair.count == 2 { headers[pair[0].lowercased()] = pair[1].trimmingCharacters(in: .whitespaces) }
        }
        let length = Int(headers["content-length"] ?? "0") ?? 0
        let bodyStart = end.upperBound
        guard buffer.count - (bodyStart - buffer.startIndex) >= length else { return nil }
        method = String(parts[0])
        let target = parts[1].split(separator: "?", maxSplits: 1)
        path = String(target.first ?? "")
        var query: [String: String] = [:]
        if target.count == 2 {
            for item in target[1].split(separator: "&") {
                let kv = item.split(separator: "=", maxSplits: 1)
                if kv.count == 2 { query[String(kv[0])] = String(kv[1]).removingPercentEncoding }
            }
        }
        self.query = query
        self.headers = headers
        body = buffer.subdata(in: bodyStart..<(bodyStart + length))
    }
}
