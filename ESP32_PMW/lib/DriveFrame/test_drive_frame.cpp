// Self-check for the shared wire format.
//
//   c++ -std=c++17 -O2 -I lib/DriveFrame/src -o /tmp/tdf lib/DriveFrame/test_drive_frame.cpp && /tmp/tdf
//
// Plain asserts and no framework, which is the convention everywhere else in this tree;
// it is a compiled binary rather than a `demo()` only because the code under test is C++.
// `controller/control/drive_frame.py` mirrors the format for the host, and its own
// `demo()` decodes bytes THIS file emits, so the two implementations are held together
// rather than merely both existing.
#include "drive_frame.h"
#include "frame_link.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
using namespace drive_frame;

static int feed_all(Framer& f, const uint8_t* p, int n, uint32_t t0, int* lines, int* frames) {
    int got = 0;
    for (int i = 0; i < n; ++i) {
        Got g = f.feed(p[i], t0 + (uint32_t)i * 11);   // ~11 us/byte at 921600
        if (g == Got::Line) { (*lines)++; got++; }
        if (g == Got::Frame) { (*frames)++; got++; }
    }
    return got;
}

int main() {
    // round trip, including the extremes of every field
    Drive d{}; d.seq = 40000;
    float a[4] = {0.0f, 33.333f, 99.999f, 100.0f};
    float p[4] = {0.0f, 90.0f, 179.9f, 359.99f};
    for (int i = 0; i < 4; ++i) { d.amp_pct[i] = a[i]; d.phase_deg[i] = p[i]; }
    d.freq_hz = 210.123f;
    uint8_t buf[DRIVE_LEN];
    assert(encode(d, buf) == DRIVE_LEN);
    Drive r{};
    assert(decode(buf, r));
    assert(r.seq == 40000);
    for (int i = 0; i < 4; ++i) {
        assert(std::fabs(r.amp_pct[i] - a[i]) < 0.002f);
        float e = std::fabs(r.phase_deg[i] - p[i]);
        assert(e < 0.01f || std::fabs(e - 360.0f) < 0.01f);
    }
    assert(std::fabs(r.freq_hz - 210.123f) < 0.001f);

    // out-of-range input is clamped at the SENDER, so the wire never carries nonsense
    Drive w{}; w.amp_pct[0] = 150.0f; w.amp_pct[1] = -20.0f; w.phase_deg[0] = -90.0f;
    encode(w, buf); decode(buf, r);
    assert(std::fabs(r.amp_pct[0] - 100.0f) < 0.01f);
    assert(std::fabs(r.amp_pct[1]) < 0.01f);
    assert(std::fabs(r.phase_deg[0] - 270.0f) < 0.01f);

    // every single-bit flip in the payload must be caught
    encode(d, buf);
    int caught = 0;
    for (int i = 1; i < DRIVE_LEN; ++i)
        for (int b = 0; b < 8; ++b) {
            uint8_t save = buf[i];
            buf[i] ^= (uint8_t)(1 << b);
            Drive x{};
            if (!decode(buf, x)) caught++;
            buf[i] = save;
        }
    assert(caught == (DRIVE_LEN - 1) * 8);

    // ---- the framer ----------------------------------------------------------------
    int lines = 0, frames = 0;
    Framer f;
    std::string s = "stop\n";
    feed_all(f, (const uint8_t*)s.data(), (int)s.size(), 0, &lines, &frames);
    assert(lines == 1 && frames == 0 && std::strcmp(f.line(), "stop") == 0);

    // a frame interleaved between two ASCII lines, all in one buffer
    encode(d, buf);
    uint8_t mix[200]; int n = 0;
    const char* c1 = "throttle=80\n";
    memcpy(mix + n, c1, strlen(c1)); n += (int)strlen(c1);
    memcpy(mix + n, buf, DRIVE_LEN); n += DRIVE_LEN;
    const char* c2 = "hover\n";
    memcpy(mix + n, c2, strlen(c2)); n += (int)strlen(c2);
    lines = frames = 0; f.reset();
    feed_all(f, mix, n, 0, &lines, &frames);
    assert(lines == 2 && frames == 1);

    // a truncated frame must not eat the `stop` that follows it. This is the property the
    // whole framing choice turns on: a stop that a corrupted link can swallow is not a
    // stop. The host sends "\nstop\n", and the leading newline is what ends the partial.
    lines = frames = 0; f.reset();
    n = 0;
    memcpy(mix, buf, 12); n = 12;                       // half a frame, then silence
    const char* st = "\nstop\n";
    memcpy(mix + n, st, strlen(st)); n += (int)strlen(st);
    for (int i = 0; i < 12; ++i) f.feed(mix[i], (uint32_t)i * 11);
    // ...the gap: the next byte arrives well after FRAME_GAP_US
    uint32_t t = 12 * 11 + 5000;
    for (int i = 12; i < n; ++i) {
        Got g = f.feed(mix[i], t + (uint32_t)(i - 12) * 11);
        if (g == Got::Line) lines++;
        if (g == Got::Frame) frames++;
    }
    assert(frames == 0 && lines == 1 && std::strcmp(f.line(), "stop") == 0);
    assert(f.n_gap() == 1);

    // an over-long line is flagged, not silently dropped, and does not wedge the parser
    lines = frames = 0; f.reset();
    std::string big(MAX_LINE + 40, 'x'); big += "\n";
    int over = 0;
    for (size_t i = 0; i < big.size(); ++i)
        if (f.feed((uint8_t)big[i], (uint32_t)i * 11) == Got::Overflow) over++;
    assert(over == 1);
    s = "stop\n"; lines = 0;
    feed_all(f, (const uint8_t*)s.data(), (int)s.size(), 0, &lines, &frames);
    assert(lines == 1);

    // a corrupted frame is counted as CRC, not as never-arrived: two different faults
    lines = frames = 0; f.reset();
    // Counters are since-boot and survive `reset()` on purpose -- they are what the ACK
    // reports -- so compare deltas, not absolutes.
    uint16_t crc0 = f.n_crc(), rx0 = f.n_rx();
    encode(d, buf); buf[7] ^= 0x01;
    feed_all(f, buf, DRIVE_LEN, 0, &lines, &frames);
    assert(frames == 0);
    assert((uint16_t)(f.n_crc() - crc0) == 1);
    assert((uint16_t)(f.n_rx() - rx0) == 0);

    // ACK round trip
    Ack ak{}; ak.seq_echo = 1234; ak.n_rx = 5000; ak.n_crc = 3; ak.state = 2;
    uint8_t ab[ACK_LEN];
    assert(encode_ack(ak, ab) == ACK_LEN);
    Ack ar{};
    assert(decode_ack(ab, ar));
    assert(ar.seq_echo == 1234 && ar.n_rx == 5000 && ar.n_crc == 3 && ar.state == 2);
    ab[4] ^= 0x02;
    assert(!decode_ack(ab, ar));

    printf("drive_frame: round trip, clamping, all %d single-bit flips caught,\n"
           "  framer handles interleave, truncation, overflow and CRC;\n"
           "  `stop` survives a desync\n  ok\n", (DRIVE_LEN - 1) * 8);
    return 0;
}
