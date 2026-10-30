#include <cassert>
#include <cstdio>
#include <utility>
#include <vector>

#include "SyncLed.h"

using namespace sync_led;

namespace {

std::vector<std::pair<uint32_t, bool>> edgesOf(SyncLed &led, uint32_t from_ms, uint32_t span_ms) {
    std::vector<std::pair<uint32_t, bool>> edges;
    bool on = led.isOn();
    edges.push_back({from_ms, on});
    for (uint32_t now_ms = from_ms + 1; now_ms <= from_ms + span_ms; ++now_ms) {
        if (led.advanceTo(now_ms)) {
            on = led.isOn();
            edges.push_back({now_ms, on});
        }
    }
    return edges;
}

}  // namespace

int main() {
    const PulseTiming timing;
    assert(timing.on_ms >= 3 * 8.34 && timing.gap_ms >= 3 * 8.34);
    assert(timing.on_ms >= 3 * 4.77 && timing.gap_ms >= 3 * 4.77);

    {
        SyncLed led;
        assert(!led.isOn() && !led.isSendingSignal());
        assert(led.sendStartSignal(1000));
        assert(led.isOn() && led.isSendingSignal());
        auto edges = edgesOf(led, 1000, 200);
        assert(edges.size() == 3);
        assert(edges[0].first == 1000 && edges[0].second);
        assert(edges[1].first == 1000 + timing.on_ms && !edges[1].second);
        assert(edges[2].first == 1000 + timing.on_ms + timing.gap_ms && edges[2].second);
        assert(!led.isSendingSignal() && led.isOn());
        assert(led.advanceTo(100000) == false);
        assert(led.isOn());
    }

    {
        SyncLed led;
        led.holdOn();
        assert(led.isOn());
        assert(led.sendStopSignal(5000));
        assert(!led.isOn() && led.isSendingSignal());
        auto edges = edgesOf(led, 5000, 200);
        assert(edges.size() == 3);
        assert(edges[0].first == 5000 && !edges[0].second);
        assert(edges[1].first == 5000 + timing.on_ms && edges[1].second);
        assert(edges[2].first == 5000 + timing.on_ms + timing.gap_ms && !edges[2].second);
        assert(!led.isSendingSignal() && !led.isOn());
    }

    {
        SyncLed led;
        assert(led.sendBlink(0));
        assert(led.isOn());
        auto edges = edgesOf(led, 0, 100);
        assert(edges.size() == 2);
        assert(edges[1].first == timing.blink_ms && !edges[1].second);
        assert(!led.isSendingSignal() && !led.isOn());
    }

    {
        SyncLed led;
        led.sendStartSignal(0);
        edgesOf(led, 0, timing.on_ms + 2);
        assert(!led.isOn() && led.isSendingSignal());
        led.holdOff();
        assert(!led.isSendingSignal() && !led.isOn());
        for (uint32_t now_ms = timing.on_ms + 3; now_ms < 500; ++now_ms)
            assert(!led.advanceTo(now_ms));
        assert(!led.isOn());
    }

    {
        SyncLed led;
        led.holdOn();
        assert(!led.sendStartSignal(0));
        assert(!led.isSendingSignal());
        for (uint32_t now_ms = 0; now_ms < 200; ++now_ms) assert(!led.advanceTo(now_ms));
        assert(led.isOn());
        assert(led.sendStopSignal(0));
        assert(!led.isOn() && led.isSendingSignal());
        assert(!led.holdOff());
        assert(!led.isSendingSignal());
    }

    {
        SyncLed led;
        led.sendStartSignal(0xFFFFFFF0u);
        assert(led.advanceTo(0x00000005u) == false);
        assert(led.advanceTo(0x0000000Eu) == true);
        assert(!led.isOn());
    }

    {
        SyncLed led(PulseTiming{5, 7, 9});
        led.sendStartSignal(0);
        auto edges = edgesOf(led, 0, 50);
        assert(edges.size() == 3);
        assert(edges[1].first == 5 && edges[2].first == 12);
        assert(led.timing().on_ms == 5 && led.timing().gap_ms == 7);
    }

    printf("SyncLed: START/STOP are 3-edge pulse pairs at %u/%u ms, blink is one pulse,\n"
           "  an interrupted signal cannot resume, and the waits survive millis() wrap\n"
           "  ok\n",
           unsigned(timing.on_ms), unsigned(timing.gap_ms));
    return 0;
}
