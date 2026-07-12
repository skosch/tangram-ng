#include "debug/profiler.h"

#include "gl.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Tangram {

std::atomic<bool> Profiler::s_active{false};
std::atomic<int64_t> Profiler::s_epochNs{0};

// gives file-local helpers access to Profiler's private state
struct ProfilerInternal {
    static void setActive(bool active) { Profiler::s_active.store(active, std::memory_order_relaxed); }
};

namespace {

constexpr size_t kChunkShift = 14;                      // 16384 events per chunk
constexpr size_t kChunkSize = size_t(1) << kChunkShift;
constexpr size_t kMaxTotalEvents = 4 * 1024 * 1024;     // hard cap: auto-stop capture
constexpr unsigned int kGpuTid = 9999;                  // synthetic thread id for GPU spans

struct SpanEvent { const char* name; int64_t t0, t1; };
struct CounterEvent { const char* name; int64_t t; double value; };

struct ThreadBuffer {
    // Chunked storage: chunks never move once allocated, so the owning thread appends
    // lock-free; only chunk-list growth (rare) and reads at stopCapture take s_mutex.
    // Events are published via a release store of count and read with an acquire load.
    std::vector<std::unique_ptr<SpanEvent[]>> chunks;
    std::atomic<size_t> count{0};
    std::string name;
    uint32_t tid = 0;
    bool named = false;
    // open zones for the non-RAII zoneBegin/zoneEnd API (used only by the owning thread)
    std::vector<std::pair<const char*, int64_t>> openZones;
};

// s_mutex guards the thread registry, chunk-list growth, counters, metadata and intern table.
std::mutex s_mutex;
std::vector<std::unique_ptr<ThreadBuffer>> s_threads;
thread_local ThreadBuffer* t_buffer = nullptr;

std::atomic<size_t> s_totalEvents{0};
std::atomic<bool> s_hitCap{false};
bool s_captureOpen = false;      // main thread only
std::string s_tracePath;

std::vector<CounterEvent> s_counterEvents;
std::map<std::string, double, std::less<>> s_counterAccum;
std::vector<int64_t> s_frameMarks;
std::vector<std::pair<std::string, std::string>> s_metadata;
std::unordered_set<std::string> s_internedNames;

// GPU timer queries: main GL thread only, no locking needed
struct GpuPending { const char* name; uint32_t q0, q1; };
std::deque<GpuPending> s_gpuPending;
std::vector<SpanEvent> s_gpuSpans;
std::vector<uint32_t> s_gpuFreeQueries;
int64_t s_gpuOffsetNs = 0;       // (capture-relative CPU ns) - (GPU timestamp ns)
bool s_gpuCalibrated = false;

ThreadBuffer* getThreadBuffer() {
    ThreadBuffer* tb = t_buffer;
    if (!tb) {
        std::lock_guard<std::mutex> lock(s_mutex);
        s_threads.push_back(std::unique_ptr<ThreadBuffer>(new ThreadBuffer()));
        tb = s_threads.back().get();
        tb->tid = uint32_t(s_threads.size() - 1);
        tb->name = "thread-" + std::to_string(tb->tid);
        t_buffer = tb;
    }
    return tb;
}

void bumpEventCount() {
    size_t total = s_totalEvents.fetch_add(1, std::memory_order_relaxed) + 1;
    if (total == kMaxTotalEvents) {
        ProfilerInternal::setActive(false);
        s_hitCap.store(true, std::memory_order_relaxed);
        LOGW("Profiler: event cap (%d) reached, capture stopped; trace is written at the next frame mark",
             int(kMaxTotalEvents));
    }
}

void appendSpan(ThreadBuffer* tb, const char* name, int64_t t0, int64_t t1) {
    size_t n = tb->count.load(std::memory_order_relaxed);
    if (n >= tb->chunks.size() * kChunkSize) {
        std::lock_guard<std::mutex> lock(s_mutex);
        tb->chunks.emplace_back(new SpanEvent[kChunkSize]);
    }
    SpanEvent& ev = tb->chunks[n >> kChunkShift][n & (kChunkSize - 1)];
    ev.name = name;
    ev.t0 = t0;
    ev.t1 = t1;
    tb->count.store(n + 1, std::memory_order_release);
    bumpEventCount();
}

const char* internLocked(const std::string& name) {
    return s_internedNames.insert(name).first->c_str();
}

uint32_t allocGpuQuery() {
    if (s_gpuFreeQueries.empty()) {
        GLuint ids[32] = { 0 };
        GL::genQueries(32, ids);
        for (GLuint id : ids) {
            if (id) { s_gpuFreeQueries.push_back(id); }
        }
        if (s_gpuFreeQueries.empty()) { return 0; }
    }
    uint32_t q = s_gpuFreeQueries.back();
    s_gpuFreeQueries.pop_back();
    return q;
}

void discardGpuPending() {
    for (auto& p : s_gpuPending) {
        s_gpuFreeQueries.push_back(p.q0);
        s_gpuFreeQueries.push_back(p.q1);
    }
    s_gpuPending.clear();
}

// resolve completed GPU queries (main GL thread); drops samples if the GPU clock was disjoint
void pollGpuQueries() {
    if (s_gpuPending.empty()) { return; }
    bool disjoint = GL::gpuDisjointOccurred();
    while (!s_gpuPending.empty()) {
        GpuPending& p = s_gpuPending.front();
        if (!GL::queryResultAvailable(p.q1)) { break; }
        int64_t g0 = int64_t(GL::queryResult64(p.q0));
        int64_t g1 = int64_t(GL::queryResult64(p.q1));
        if (!disjoint && g1 >= g0) {
            s_gpuSpans.push_back({p.name, g0 + s_gpuOffsetNs, g1 + s_gpuOffsetNs});
            bumpEventCount();
        }
        s_gpuFreeQueries.push_back(p.q0);
        s_gpuFreeQueries.push_back(p.q1);
        s_gpuPending.pop_front();
    }
}

void writeJsonString(FILE* f, const char* s) {
    fputc('"', f);
    for (; *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c < 0x20) { fprintf(f, "\\u%04x", c); }
        else { fputc(c, f); }
    }
    fputc('"', f);
}

// called with s_mutex held; ts/dur in microseconds per the Trace Event format
void writeTraceLocked(FILE* f) {
    fputs("{\"traceEvents\":[", f);
    bool first = true;
    auto sep = [&]() { if (!first) { fputs(",\n", f); } else { fputs("\n", f); first = false; } };

    for (auto& tb : s_threads) {
        sep();
        fprintf(f, "{\"ph\":\"M\",\"pid\":0,\"tid\":%u,\"name\":\"thread_name\",\"args\":{\"name\":", tb->tid);
        writeJsonString(f, tb->name.c_str());
        fputs("}}", f);
    }
    if (!s_gpuSpans.empty()) {
        sep();
        fprintf(f, "{\"ph\":\"M\",\"pid\":0,\"tid\":%u,\"name\":\"thread_name\",\"args\":{\"name\":\"GPU\"}}", kGpuTid);
    }

    auto writeSpan = [&](unsigned int tid, const SpanEvent& ev) {
        sep();
        double dur = (ev.t1 > ev.t0) ? (ev.t1 - ev.t0) / 1000.0 : 0.0;
        fprintf(f, "{\"ph\":\"X\",\"pid\":0,\"tid\":%u,\"name\":", tid);
        writeJsonString(f, ev.name);
        fprintf(f, ",\"ts\":%.3f,\"dur\":%.3f}", ev.t0 / 1000.0, dur);
    };

    for (auto& tb : s_threads) {
        size_t n = tb->count.load(std::memory_order_acquire);
        n = std::min(n, tb->chunks.size() * kChunkSize);
        for (size_t i = 0; i < n; i++) {
            writeSpan(tb->tid, tb->chunks[i >> kChunkShift][i & (kChunkSize - 1)]);
        }
    }
    for (const SpanEvent& ev : s_gpuSpans) { writeSpan(kGpuTid, ev); }

    for (const CounterEvent& ev : s_counterEvents) {
        sep();
        fputs("{\"ph\":\"C\",\"pid\":0,\"name\":", f);
        writeJsonString(f, ev.name);
        fprintf(f, ",\"ts\":%.3f,\"args\":{\"v\":%g}}", ev.t / 1000.0, ev.value);
    }

    for (int64_t t : s_frameMarks) {
        sep();
        fprintf(f, "{\"ph\":\"i\",\"pid\":0,\"tid\":0,\"name\":\"frame\",\"s\":\"g\",\"ts\":%.3f}", t / 1000.0);
    }

    fputs("\n],\"metadata\":{", f);
    first = true;
    for (auto& kv : s_metadata) {
        if (!first) { fputc(',', f); }
        first = false;
        writeJsonString(f, kv.first.c_str());
        fputc(':', f);
        writeJsonString(f, kv.second.c_str());
    }
    fputs("}}\n", f);
}

}  // anonymous namespace

void Profiler::record(const char* name, int64_t t0, int64_t t1) {
    appendSpan(getThreadBuffer(), name, t0, t1);
}

void Profiler::zoneBegin(const char* name) {
    if (!isCapturing()) { return; }
    getThreadBuffer()->openZones.emplace_back(name, timeNs());
}

void Profiler::zoneEnd(const char*) {
    ThreadBuffer* tb = t_buffer;
    if (!tb || tb->openZones.empty()) { return; }  // capture started after zoneBegin
    auto zone = tb->openZones.back();
    tb->openZones.pop_back();
    appendSpan(tb, zone.first, zone.second, timeNs());
}

void Profiler::counter(const char* name, double value) {
    if (!isCapturing()) { return; }
    int64_t t = timeNs();
    std::lock_guard<std::mutex> lock(s_mutex);
    s_counterEvents.push_back({name, t, value});
    bumpEventCount();
}

void Profiler::counterAdd(const char* name, double value) {
    if (!isCapturing()) { return; }
    std::lock_guard<std::mutex> lock(s_mutex);
    auto it = s_counterAccum.find(name);
    if (it != s_counterAccum.end()) { it->second += value; }
    else { s_counterAccum.emplace(name, value); }
}

void Profiler::meta(const char* key, const std::string& value) {
    std::lock_guard<std::mutex> lock(s_mutex);
    for (auto& kv : s_metadata) {
        if (kv.first == key) { kv.second = value; return; }
    }
    s_metadata.emplace_back(key, value);
}

void Profiler::threadName(const char* name) {
    ThreadBuffer* tb = getThreadBuffer();
    std::lock_guard<std::mutex> lock(s_mutex);
    tb->name = name;
    tb->named = true;
}

const char* Profiler::intern(const std::string& name) {
    std::lock_guard<std::mutex> lock(s_mutex);
    return internLocked(name);
}

bool Profiler::gpuBegin(uint32_t& q0) {
    if (!GL::timerQueriesSupported()) { return false; }
    if (!s_gpuCalibrated) {
        int64_t gpuNow = GL::gpuTimestampNs();
        if (gpuNow <= 0) { return false; }
        s_gpuOffsetNs = timeNs() - gpuNow;
        s_gpuCalibrated = true;
    }
    q0 = allocGpuQuery();
    if (!q0) { return false; }
    GL::queryCounter(q0);
    return true;
}

void Profiler::gpuEnd(const char* name, uint32_t q0) {
    uint32_t q1 = allocGpuQuery();
    if (!q1) { s_gpuFreeQueries.push_back(q0); return; }
    GL::queryCounter(q1);
    s_gpuPending.push_back({name, q0, q1});
}

void Profiler::frameMark() {
    if (!isCapturing()) {
        // finalize an auto-stopped capture (event cap hit on some thread mid-frame)
        if (s_hitCap.load(std::memory_order_relaxed) && s_captureOpen) { stopCapture(); }
        return;
    }
    int64_t t = timeNs();
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        s_frameMarks.push_back(t);
        // flush counterAdd accumulators as counter samples for this frame
        for (auto& kv : s_counterAccum) {
            s_counterEvents.push_back({internLocked(kv.first), t, kv.second});
        }
        s_counterAccum.clear();
    }
    pollGpuQueries();
}

void Profiler::startCapture(const std::string& jsonPath) {
    if (s_captureOpen) { stopCapture(); }
    ThreadBuffer* tb = getThreadBuffer();
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        // reset buffers; chunk memory is kept (never freed) so stale in-flight zones from a
        // previous capture can never write out of bounds
        for (auto& t : s_threads) {
            t->count.store(0, std::memory_order_relaxed);
            t->openZones.clear();
        }
        if (!tb->named) { tb->name = "Main"; tb->named = true; }
        s_counterEvents.clear();
        s_counterAccum.clear();
        s_frameMarks.clear();
        s_metadata.clear();
        s_tracePath = jsonPath;
    }
    discardGpuPending();
    s_gpuSpans.clear();
    s_gpuCalibrated = false;
    s_totalEvents.store(0, std::memory_order_relaxed);
    s_hitCap.store(false, std::memory_order_relaxed);
    s_captureOpen = true;
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    s_epochNs.store(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count(),
                    std::memory_order_relaxed);
    s_active.store(true, std::memory_order_release);
    LOG("Profiler: capture started (%s)", jsonPath.c_str());
}

void Profiler::stopCapture() {
    s_active.store(false);
    if (!s_captureOpen) { return; }
    s_captureOpen = false;
    pollGpuQueries();     // resolve whatever is already complete; drop the rest
    discardGpuPending();

    std::lock_guard<std::mutex> lock(s_mutex);
    FILE* f = fopen(s_tracePath.c_str(), "w");
    if (!f) {
        LOGE("Profiler: failed to open trace file '%s' for writing", s_tracePath.c_str());
        return;
    }
    writeTraceLocked(f);
    fclose(f);
    LOG("Profiler: capture stopped, %d events written to %s",
        int(s_totalEvents.load(std::memory_order_relaxed)), s_tracePath.c_str());
}

}
