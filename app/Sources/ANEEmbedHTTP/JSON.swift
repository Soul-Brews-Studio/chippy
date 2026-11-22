import Foundation

/// {"model": ..., "embeddings": [[...]]} written by hand: each float32 is widened to a
/// Double and printed in its shortest round-trip form, which is what Python's json gives
/// for the same values (float32 -> float64 via tolist()), so clients parse equal numbers.
public func encodeEmbedResponse(model: String, vectors: [[Float]]) -> Data {
    var out = "{\"model\":\(quote(model)),\"embeddings\":["
    out.reserveCapacity(vectors.count * 768 * 22 + 64)
    for (i, vector) in vectors.enumerated() {
        if i > 0 { out += "," }
        out += "["
        for (j, x) in vector.enumerated() {
            if j > 0 { out += "," }
            out += String(Double(x))
        }
        out += "]"
    }
    out += "]}"
    return Data(out.utf8)
}

func quote(_ s: String) -> String {
    let data = try! JSONSerialization.data(withJSONObject: [s], options: [.fragmentsAllowed])
    let wrapped = String(decoding: data, as: UTF8.self)
    return String(wrapped.dropFirst().dropLast())
}

/// The request body: {"model": "...", "input": "text" | ["text", ...]}, or for images (ANE Embed 0.2)
/// {"model": "...", "images": ["<base64>" | "data:image/...;base64,<base64>", ...], "vision": "v1" | "v2"}.
struct EmbedRequest {
    let model: String?
    let texts: [String]
    let images: [Data]
    let vision: String?
    /// "solo": only this Mac's engine; "cluster" (default): spread over the worker pool.
    let pool: String?

    init(_ body: Data) throws {
        guard let object = try JSONSerialization.jsonObject(with: body) as? [String: Any] else {
            throw HTTPError(400, "body must be a JSON object")
        }
        model = object["model"] as? String
        vision = object["vision"] as? String
        pool = object["pool"] as? String
        if let pool, pool != "solo", pool != "cluster" { throw HTTPError(400, "pool must be \"solo\" or \"cluster\"") }
        if let list = object["images"] {
            guard let many = list as? [Any], !many.isEmpty else { throw HTTPError(400, "images must be a non-empty list") }
            guard object["input"] == nil else { throw HTTPError(400, "send input (texts) or images, not both") }
            images = try many.map { item in
                guard var s = item as? String else { throw HTTPError(400, "images must be base64 strings") }
                if s.hasPrefix("data:"), let comma = s.firstIndex(of: ",") { s = String(s[s.index(after: comma)...]) }
                guard let data = Data(base64Encoded: s, options: .ignoreUnknownCharacters) else { throw HTTPError(400, "an image is not valid base64") }
                return data
            }
            texts = []
            return
        }
        images = []
        switch object["input"] {
        case let one as String: texts = [one]
        case let many as [Any]:
            texts = try many.map { item in
                guard let s = item as? String else { throw HTTPError(400, "input must be strings") }
                return s
            }
        default: throw HTTPError(400, "input is required")
        }
    }
}

public struct HTTPError: Error, CustomStringConvertible {
    public var description: String { message }
    let status: Int
    let message: String
    init(_ status: Int, _ message: String) { self.status = status; self.message = message }
}
