// The binary DRIVE/ACK wire format, and the ONLY place its layout is written down.
//
// Included by the firmware (`src/main_flight.cpp`) and by the host's native controller
// (`controller/native/`), which is why it is plain C++ with no Arduino and no STL: one
// header, one definition, no copy to drift. `controller/control/drive_frame.py` is a
// Python mirror for tests and tooling, and `test_drive_frame` holds the two together.
//
// WHY BINARY, AND WHY ONLY FOR THIS
// ---------------------------------
// The ASCII commands (`seq=`, `probe=`, `stop`, ...) are rare, human-typable and worth
// reading in a serial monitor. They stay. What does not fit that shape is the periodic
// drive command: four amplitudes, four phases and a frequency at 200 Hz. As ASCII that is
// ~60 bytes to format, parse and echo 200 times a second, with an Arduino `String`
// allocation per field on a device whose heap is the thing that fails first.
//
// See `control/theory.md` 26.

#pragma once

#include <stdint.h>

namespace drive_frame {

// Sync bytes have the high bit set. Every ASCII command the host sends is 7-bit, so a
// 0xA5 at a line boundary CANNOT be text -- which is what makes the two framings
// coexist in one stream without an escape scheme or a mode command.
static const uint8_t SYNC_DRIVE = 0xA5;
static const uint8_t SYNC_ACK   = 0xA6;
static const uint8_t TYPE_DRIVE = 0x01;
static const uint8_t TYPE_ACK   = 0x02;

static const int NCH = 4;
static const int DRIVE_LEN = 26;   // sync,type,seq(2),amp(8),phase(8),freq(4),crc(2)
static const int ACK_LEN   = 11;   // sync,type,seq(2),n_rx(2),n_crc(2),state,crc(2)

// Fixed length, no length field. A corrupted length byte is a desync source and there is
// nothing variable here to describe.

// ---- fixed-point scaling ---------------------------------------------------------------
// Amplitude is carrier duty percent, 0..100, as u16. 655.35 counts per percent puts the
// LSB at 0.0015 %, far below anything the LEDC hardware resolves, so the quantisation is
// never the limiting term and the field never saturates inside its range.
static const float AMP_SCALE = 655.35f;
// Phase is degrees, 0..360, as u16: 65536/360 = 182.0444 counts per degree, 0.0055 deg per
// LSB. u8 was considered and rejected -- 1.41 deg/LSB is finer than the 1.8 deg the
// commutation ISR resolves at a 200 Hz field, but COARSER than the 0.45 deg it resolves at
// 50 Hz, and the whole ramp runs through that band.
static const float PHASE_SCALE = 182.04444f;
// Frequency in millihertz as u32: 0.001 Hz over the whole 0..300 Hz range, and 0 is DC
// idle, which is what `setGlobalFrequency` already means by zero.

struct Drive {
    uint16_t seq;
    float amp_pct[NCH];     // 0..100
    float phase_deg[NCH];   // 0..360
    float freq_hz;          // 0 = DC idle
};

struct Ack {
    uint16_t seq_echo;      // last DRIVE applied
    uint16_t n_rx;          // frames accepted since boot
    uint16_t n_crc;         // frames rejected on CRC or desync
    uint8_t  state;         // firmware State enum
};

// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final xor. Bitwise
// rather than table-driven -- 24 bytes at 200 Hz is 38 k iterations a second on a 240 MHz
// core, and a 512-byte table in a header that two builds include is the worse trade.
inline uint16_t crc16(const uint8_t* p, int n) {
    uint16_t c = 0xFFFF;
    for (int i = 0; i < n; ++i) {
        c ^= (uint16_t)p[i] << 8;
        for (int b = 0; b < 8; ++b)
            c = (c & 0x8000) ? (uint16_t)((c << 1) ^ 0x1021) : (uint16_t)(c << 1);
    }
    return c;
}

inline void put_u16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)(v & 0xFF); p[1] = (uint8_t)(v >> 8); }
inline void put_u32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFF);  p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF); p[3] = (uint8_t)((v >> 24) & 0xFF);
}
inline uint16_t get_u16(const uint8_t* p) { return (uint16_t)(p[0] | ((uint16_t)p[1] << 8)); }
inline uint32_t get_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

inline float clampf_(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Clamp-then-round, so a host that sends 100.001 gets full drive rather than wrapping to
// zero. Values are bounded here and NOT trusted again on the far side.
inline uint16_t enc_amp(float pct)   { return (uint16_t)(clampf_(pct, 0.0f, 100.0f) * AMP_SCALE + 0.5f); }
inline uint16_t enc_phase(float deg) {
    float d = deg;
    while (d < 0.0f) d += 360.0f;
    while (d >= 360.0f) d -= 360.0f;
    uint32_t v = (uint32_t)(d * PHASE_SCALE + 0.5f);
    return (uint16_t)(v > 65535u ? 0u : v);   // 359.9973+ rounds to 65536, i.e. 0 deg
}

//: Serialise into `out` (DRIVE_LEN bytes). Returns DRIVE_LEN.
inline int encode(const Drive& d, uint8_t* out) {
    out[0] = SYNC_DRIVE;
    out[1] = TYPE_DRIVE;
    put_u16(out + 2, d.seq);
    for (int i = 0; i < NCH; ++i) put_u16(out + 4 + 2 * i, enc_amp(d.amp_pct[i]));
    for (int i = 0; i < NCH; ++i) put_u16(out + 12 + 2 * i, enc_phase(d.phase_deg[i]));
    put_u32(out + 20, (uint32_t)(clampf_(d.freq_hz, 0.0f, 1000.0f) * 1000.0f + 0.5f));
    // CRC covers bytes 1..23: everything but the sync byte, which is the thing being
    // resynchronised ON and so cannot also be the thing being checked.
    put_u16(out + 24, crc16(out + 1, 23));
    return DRIVE_LEN;
}

//: Parse `in` (DRIVE_LEN bytes, sync already matched). False on a CRC or type failure.
inline bool decode(const uint8_t* in, Drive& d) {
    if (in[0] != SYNC_DRIVE || in[1] != TYPE_DRIVE) return false;
    if (get_u16(in + 24) != crc16(in + 1, 23)) return false;
    d.seq = get_u16(in + 2);
    for (int i = 0; i < NCH; ++i) d.amp_pct[i] = get_u16(in + 4 + 2 * i) / AMP_SCALE;
    for (int i = 0; i < NCH; ++i) d.phase_deg[i] = get_u16(in + 12 + 2 * i) / PHASE_SCALE;
    d.freq_hz = get_u32(in + 20) / 1000.0f;
    return true;
}

inline int encode_ack(const Ack& a, uint8_t* out) {
    out[0] = SYNC_ACK;
    out[1] = TYPE_ACK;
    put_u16(out + 2, a.seq_echo);
    put_u16(out + 4, a.n_rx);
    put_u16(out + 6, a.n_crc);
    out[8] = a.state;
    put_u16(out + 9, crc16(out + 1, 8));
    return ACK_LEN;
}

inline bool decode_ack(const uint8_t* in, Ack& a) {
    if (in[0] != SYNC_ACK || in[1] != TYPE_ACK) return false;
    if (get_u16(in + 9) != crc16(in + 1, 8)) return false;
    a.seq_echo = get_u16(in + 2);
    a.n_rx = get_u16(in + 4);
    a.n_crc = get_u16(in + 6);
    a.state = in[8];
    return true;
}

}  // namespace drive_frame
