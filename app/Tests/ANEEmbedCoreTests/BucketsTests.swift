import XCTest
@testable import ANEEmbedCore

final class BucketsTests: XCTestCase {
    let buckets = [32, 64, 128, 256, 512, 640, 768, 1024, 2048]

    func testSmallestBucketInRequestOrderChunkedByBatch() throws {
        let jobs = try assignJobs(lengths: [10, 300, 33, 5, 2048], buckets: buckets)
        XCTAssertEqual(jobs, [Job(bucket: 32, rows: [0, 3]), Job(bucket: 64, rows: [2]),
                              Job(bucket: 512, rows: [1]), Job(bucket: 2048, rows: [4])])
    }

    func testBatchLimitSplitsJobs() throws {
        let jobs = try assignJobs(lengths: Array(repeating: 600, count: 7), buckets: buckets)
        XCTAssertEqual(jobs.map(\.rows), [[0, 1, 2], [3, 4, 5], [6]])
    }

    func testRejectsEmptyAndOversizedRows() {
        XCTAssertThrowsError(try assignJobs(lengths: [0], buckets: buckets))
        XCTAssertThrowsError(try assignJobs(lengths: [2049], buckets: buckets))
    }
}
