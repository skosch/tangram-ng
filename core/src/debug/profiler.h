#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>

namespace Tangram {

// Always-compiled, runtime-toggled instrumentation profiler that writes Chrome Trace Event
// JSON (viewable in ui.perfetto.dev; summarized by scripts/perf/summarize_trace.py).
//
// - When capture is inactive, per-zone overhead is one relaxed atomic load + branch.
// - Zone/counter names must be string literals or pointers returned by Profiler::intern()
//   -- never a temporary std::string::c_str().
// - startCapture/stopCapture/frameMark must be called from the main thread; Zone, counter,
//   counterAdd and threadName may be used from any thread; GpuZone only on the main GL context.
class Profiler {
public:
    static bool isCapturing() { return s_active.load(std::memory_order_relaxed); }

    static void startCapture(const std::string& jsonPath);   // main thread
    static void stopCapture();                                // main thread; writes JSON
    static void frameMark();           // once per presented frame, main thread
    static void counter(const char* name, double value);      // sampled value, this frame
    static void counterAdd(const char* name, double value);   // accumulates within frame, reset at frameMark
    static void meta(const char* key, const std::string& value);  // capture-level metadata
    static void threadName(const char* name);                 // call once per worker thread
    static const char* intern(const std::string& name);       // stable ptr for dynamic names (per-style, per-source)

    // Non-RAII zone pair for bridge code (e.g. FrameInfo::begin/end); prefer PROFILE_SCOPE.
    // Must be balanced (LIFO) on each thread; unbalanced calls are dropped gracefully.
    static void zoneBegin(const char* name);
    static void zoneEnd(const char* name);

    struct Zone {                      // RAII CPU zone
        Zone(const char* name) : m_live(isCapturing()) {
            if (m_live) { m_name = name; m_t0 = timeNs(); }
        }
        ~Zone() {
            if (m_live) { record(m_name, m_t0, timeNs()); }
        }
        const char* m_name; int64_t m_t0; bool m_live;
    };

    struct GpuZone {                   // RAII GL timer-query zone, main GL context only
        GpuZone(const char* name) : m_live(isCapturing()) {
            if (m_live) { m_name = name; m_live = gpuBegin(m_q0); }
        }
        ~GpuZone() {
            if (m_live) { gpuEnd(m_name, m_q0); }
        }
        const char* m_name; uint32_t m_q0 = 0; bool m_live;
    };

private:
    // ns since capture start (steady_clock)
    static int64_t timeNs() {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()
            - s_epochNs.load(std::memory_order_relaxed);
    }
    static void record(const char* name, int64_t t0, int64_t t1);
    static bool gpuBegin(uint32_t& q0);
    static void gpuEnd(const char* name, uint32_t q0);

    friend struct ProfilerInternal;   // profiler.cpp helpers (event cap enforcement)

    static std::atomic<bool> s_active;
    static std::atomic<int64_t> s_epochNs;
};

}

// __LINE__ needs an extra expansion step to be pasted as its numeric value
#define PROFILER_PASTE2(a, b) a##b
#define PROFILER_PASTE(a, b) PROFILER_PASTE2(a, b)
#define PROFILE_SCOPE(name)  Tangram::Profiler::Zone PROFILER_PASTE(_profZone, __LINE__)(name)
#define PROFILE_GPU(name)    Tangram::Profiler::GpuZone PROFILER_PASTE(_profGpu, __LINE__)(name)
