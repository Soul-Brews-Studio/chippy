import Darwin
import Foundation

/// This process's CPU % (all cores, so 200% = two cores busy) since the last sample, and
/// its memory footprint: the number Activity Monitor shows as "Memory".
struct ResourceSampler {
    private var lastCPU: Double?
    private var lastAt: Date?

    mutating func sample() -> (cpuPercent: Double?, memoryMB: Double?) {
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        let cpu = seconds(usage.ru_utime) + seconds(usage.ru_stime)
        let now = Date()
        var percent: Double?
        if let lastCPU, let lastAt, now > lastAt { percent = 100 * (cpu - lastCPU) / now.timeIntervalSince(lastAt) }
        lastCPU = cpu
        lastAt = now
        return (percent, footprintMB())
    }

    private func seconds(_ t: timeval) -> Double { Double(t.tv_sec) + Double(t.tv_usec) / 1e6 }

    private func footprintMB() -> Double? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : nil
    }
}
