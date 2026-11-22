import CoreML
import Foundation

/// Fixed-shape float16 inputs for one job, as ane_runtime.stage_inputs builds them:
/// embeds[b, t, :] = float16(token_table[id] * scale) (precomputed in embed_scaled.f16),
/// mask[b, t] = 1 for real tokens, everything else zero.
func stageInputs(job: Job, tokenIds: [[Int]], assets: Assets) throws -> (embeds: MLMultiArray, mask: MLMultiArray) {
    let hidden = assets.manifest.hidden
    let batch = max(1, assets.manifest.slots_per_call / job.bucket)
    let embeds = try MLMultiArray(shape: [batch, job.bucket, hidden].map(NSNumber.init), dataType: .float16)
    let mask = try MLMultiArray(shape: [batch, job.bucket].map(NSNumber.init), dataType: .float16)
    let rowBytes = hidden * 2
    let one: UInt16 = 0x3C00   // float16 1.0
    embeds.withUnsafeMutableBytes { dst, _ in
        memset(dst.baseAddress!, 0, dst.count)
        assets.embedScaled.withUnsafeBytes { table in
            for (b, row) in job.rows.enumerated() {
                for (t, id) in tokenIds[row].enumerated() {
                    memcpy(dst.baseAddress! + (b * job.bucket + t) * rowBytes, table.baseAddress! + id * rowBytes, rowBytes)
                }
            }
        }
    }
    mask.withUnsafeMutableBytes { dst, _ in
        memset(dst.baseAddress!, 0, dst.count)
        let m = dst.bindMemory(to: UInt16.self)
        for (b, row) in job.rows.enumerated() {
            for t in 0..<tokenIds[row].count { m[b * job.bucket + t] = one }
        }
    }
    return (embeds, mask)
}

/// The same inputs from precomputed rows (an image's sequence: token-table rows around the vision
/// encoder's soft tokens): embeds[b, t, :] = rows[row][t], mask 1 over each row's length.
func stageRows(job: Job, rows: [[Float16]], assets: Assets) throws -> (embeds: MLMultiArray, mask: MLMultiArray) {
    let hidden = assets.manifest.hidden
    let batch = max(1, assets.manifest.slots_per_call / job.bucket)
    let embeds = try MLMultiArray(shape: [batch, job.bucket, hidden].map(NSNumber.init), dataType: .float16)
    let mask = try MLMultiArray(shape: [batch, job.bucket].map(NSNumber.init), dataType: .float16)
    embeds.withUnsafeMutableBytes { dst, _ in
        memset(dst.baseAddress!, 0, dst.count)
        for (b, row) in job.rows.enumerated() {
            rows[row].withUnsafeBytes { src in
                memcpy(dst.baseAddress! + b * job.bucket * hidden * 2, src.baseAddress!, src.count)
            }
        }
    }
    mask.withUnsafeMutableBytes { dst, _ in
        memset(dst.baseAddress!, 0, dst.count)
        let m = dst.bindMemory(to: UInt16.self)
        for (b, row) in job.rows.enumerated() {
            for t in 0..<(rows[row].count / hidden) { m[b * job.bucket + t] = 0x3C00 }
        }
    }
    return (embeds, mask)
}
