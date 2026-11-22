import ANEEmbedCore
import Darwin
import Foundation
import IOKit

/// Whole-machine CPU, RAM and GPU, read natively (no sudo, no helper): CPU from the
/// per-core tick counters, RAM as active + wired + compressed pages (what Activity
/// Monitor calls "Memory Used"), GPU from IOAccelerator's "Device Utilization %".
/// Power per component comes from a mactop exporter on :9101 when one is running;
/// macOS has no public API for ANE utilisation or watts.
struct SystemMonitor {
    struct Reading {
        var cpuPercent: Double?
        var ramUsedGB: Double?
        var ramTotalGB = Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824
        var gpuPercent: Double?
        var aneWatts: Double?, gpuWatts: Double?, cpuWatts: Double?
        /// From IOReport directly (ANEMonitor): works on macOS 27, where ANE watts read 0 unless powermetrics samples the ANE.
        var aneUtilization: Double?, aneBandwidthGBs: Double?
    }

    private var lastTicks: (busy: UInt64, total: UInt64)?
    private let ane = ANEMonitor()

    mutating func read() async -> Reading {
        var r = Reading()
        r.cpuPercent = cpu()
        r.ramUsedGB = ramUsed()
        r.gpuPercent = Self.gpu()
        if let a = ane?.read() { r.aneUtilization = a.utilizationPercent; r.aneBandwidthGBs = a.bandwidthGBs }
        if let power = await Self.mactopPower() {
            r.aneWatts = power["ane"]; r.gpuWatts = power["gpu"]; r.cpuWatts = power["cpu"]
        }
        return r
    }

    private mutating func cpu() -> Double? {
        var info: processor_info_array_t?
        var count: mach_msg_type_number_t = 0
        var cpus: natural_t = 0
        guard host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &cpus, &info, &count) == KERN_SUCCESS, let info else { return nil }
        defer { vm_deallocate(mach_task_self_, vm_address_t(bitPattern: info), vm_size_t(count) * vm_size_t(MemoryLayout<integer_t>.size)) }
        var busy: UInt64 = 0, total: UInt64 = 0
        for i in 0..<Int(cpus) {
            let base = Int(CPU_STATE_MAX) * i
            let user = UInt64(info[base + Int(CPU_STATE_USER)]), system = UInt64(info[base + Int(CPU_STATE_SYSTEM)])
            let nice = UInt64(info[base + Int(CPU_STATE_NICE)]), idle = UInt64(info[base + Int(CPU_STATE_IDLE)])
            busy += user + system + nice
            total += user + system + nice + idle
        }
        defer { lastTicks = (busy, total) }
        guard let last = lastTicks, total > last.total else { return nil }
        return 100 * Double(busy - last.busy) / Double(total - last.total)
    }

    private func ramUsed() -> Double? {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        guard result == KERN_SUCCESS else { return nil }
        let pages = UInt64(stats.active_count) + UInt64(stats.wire_count) + UInt64(stats.compressor_page_count)
        return Double(pages * UInt64(vm_kernel_page_size)) / 1_073_741_824
    }

    static func gpu() -> Double? {
        var iterator: io_iterator_t = 0
        guard IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator) == KERN_SUCCESS else { return nil }
        defer { IOObjectRelease(iterator) }
        var service = IOIteratorNext(iterator)
        while service != 0 {
            defer { IOObjectRelease(service); service = IOIteratorNext(iterator) }
            if let stats = IORegistryEntryCreateCFProperty(service, "PerformanceStatistics" as CFString, kCFAllocatorDefault, 0)?
                .takeRetainedValue() as? [String: Any],
               let util = stats["Device Utilization %"] as? NSNumber {
                return util.doubleValue
            }
        }
        return nil
    }

    static func mactopPower() async -> [String: Double]? {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:9101/metrics")!)
        request.timeoutInterval = 0.5
        guard let (data, _) = try? await URLSession.shared.data(for: request) else { return nil }
        var out: [String: Double] = [:]
        for line in String(decoding: data, as: UTF8.self).split(separator: "\n") where line.hasPrefix("mactop_power_watts{component=\"") {
            let name = line.dropFirst("mactop_power_watts{component=\"".count).prefix { $0 != "\"" }
            out[String(name)] = Double(line.split(separator: " ").last ?? "")
        }
        return out
    }
}
