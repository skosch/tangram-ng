#include "catch.hpp"

#include "debug/profiler.h"
#include "util/json.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

using namespace Tangram;

// Exercises Zone/GpuZone/counter/frameMark/stopCapture without a GL context (mock GL reports
// no timer-query support, so GpuZone must be a silent no-op) and validates the trace JSON.

static std::string tracePath() {
    const char* tmpdir = getenv("TMPDIR");
    return std::string(tmpdir && tmpdir[0] ? tmpdir : "/tmp") + "/tangram-profiler-test-trace.json";
}

static std::string readFile(const std::string& path) {
    std::string data;
    FILE* f = fopen(path.c_str(), "rb");
    REQUIRE(f != nullptr);
    fseek(f, 0, SEEK_END);
    data.resize(ftell(f));
    fseek(f, 0, SEEK_SET);
    REQUIRE(fread(&data[0], 1, data.size(), f) == data.size());
    fclose(f);
    return data;
}

TEST_CASE("Profiler writes a parseable Chrome trace without GL", "[profiler]") {

    std::string path = tracePath();
    remove(path.c_str());

    REQUIRE_FALSE(Profiler::isCapturing());
    { PROFILE_SCOPE("BeforeCapture"); }        // must be a no-op
    Profiler::counter("ignored", 1.0);         // must be a no-op

    Profiler::startCapture(path);
    REQUIRE(Profiler::isCapturing());
    Profiler::meta("scenario", "unit-test");

    std::thread worker([]() {
        Profiler::threadName("TestWorker");
        for (int i = 0; i < 50; i++) { PROFILE_SCOPE("WorkerTask"); }
    });

    const char* dynName = Profiler::intern(std::string("style:") + "test");
    for (int frame = 0; frame < 5; frame++) {
        {
            PROFILE_SCOPE("Update");
            Profiler::Zone dynZone(dynName);
            Profiler::counterAdd("drawCalls", 3);
            Profiler::counterAdd("drawCalls", 2);
        }
        { PROFILE_GPU("Render"); }             // silent no-op without timer queries
        Profiler::zoneBegin("Bridged");        // FrameInfo-style non-RAII pair
        Profiler::zoneEnd("Bridged");
        Profiler::counter("zoom", 10.0 + frame);
        Profiler::frameMark();
    }
    worker.join();

    Profiler::stopCapture();
    REQUIRE_FALSE(Profiler::isCapturing());

    std::string json = readFile(path);
    const char* error = nullptr;
    size_t errorOffset = 0;
    JsonDocument doc = JsonParseBytes(json.data(), json.size(), &error, &errorOffset);
    REQUIRE(error == nullptr);
    REQUIRE(doc.IsObject());
    REQUIRE(doc.HasMember("traceEvents"));
    REQUIRE(doc["traceEvents"].IsArray());

    int spans = 0, frames = 0, counters = 0, threadNames = 0;
    int updateSpans = 0, workerSpans = 0, gpuSpans = 0, bridgedSpans = 0, dynSpans = 0;
    for (auto& ev : doc["traceEvents"].GetArray()) {
        std::string ph = ev["ph"].GetString();
        std::string name = ev.HasMember("name") ? ev["name"].GetString() : "";
        if (ph == "X") {
            spans++;
            REQUIRE(ev["ts"].IsNumber());
            REQUIRE(ev["dur"].IsNumber());
            REQUIRE(ev["dur"].GetDouble() >= 0.0);
            if (name == "Update") { updateSpans++; }
            if (name == "WorkerTask") { workerSpans++; }
            if (name == "Render") { gpuSpans++; }
            if (name == "Bridged") { bridgedSpans++; }
            if (name == "style:test") { dynSpans++; }
        } else if (ph == "i" && name == "frame") {
            frames++;
        } else if (ph == "C") {
            counters++;
        } else if (ph == "M" && name == "thread_name") {
            threadNames++;
        }
    }
    CHECK(updateSpans == 5);
    CHECK(workerSpans == 50);
    CHECK(dynSpans == 5);
    CHECK(bridgedSpans == 5);
    CHECK(gpuSpans == 0);                       // no GL: GpuZone must not record anything
    CHECK(frames == 5);
    CHECK(counters == 10);                      // 5x zoom + 5x flushed drawCalls
    CHECK(threadNames >= 2);                    // main + worker
    CHECK(spans == updateSpans + workerSpans + dynSpans + bridgedSpans);
    CHECK(spans + frames + counters <= 4 * 1024 * 1024);

    REQUIRE(doc.HasMember("metadata"));
    REQUIRE(doc["metadata"].HasMember("scenario"));
    CHECK(std::string(doc["metadata"]["scenario"].GetString()) == "unit-test");

    // no recording once capture is stopped
    { PROFILE_SCOPE("AfterCapture"); }
    Profiler::counter("ignored", 2.0);
}
