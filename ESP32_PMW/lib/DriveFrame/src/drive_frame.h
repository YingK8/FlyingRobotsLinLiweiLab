#pragma once

#include <stdint.h>

namespace drive_frame {

// Sync bytes have the high bit set, so they cannot appear in 7-bit ASCII commands.
static const uint8_t SYNC_DRIVE = 0xA5;
static const uint8_t SYNC_ACK   = 0xA6;
static const uint8_t TYPE_DRIVE = 0x01;
static const uint8_t TYPE_ACK   = 0x02;

static const int NCH = 4;
static const int DRIVE_LEN = 26;
static const int ACK_LEN   = 11;

static const float AMP_SCALE = 655.35f;
static const float PHASE_SCALE = 182.04444f;

struct Drive {
    uint16_t seq;
    float amp_pct[NCH];
    float phase_deg[NCH];
    float freq_hz;
};

struct Ack {
    uint16_t seq_echo;
    uint16_t n_rx;
    uint16_t n_crc;
    uint8_t state;
};

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

// Clamp then round: 100.001 must give full drive, not wrap to zero.
inline uint16_t enc_amp(float pct)   { return (uint16_t)(clampf_(pct, 0.0f, 100.0f) * AMP_SCALE + 0.5f); }
inline uint16_t enc_phase(float deg) {
    float d = deg;
    while (d < 0.0f) d += 360.0f;
    while (d >= 360.0f) d -= 360.0f;
    uint32_t v = (uint32_t)(d * PHASE_SCALE + 0.5f);
    return (uint16_t)(v > 65535u ? 0u : v);   // 359.9973 deg rounds to 65536, so wrap to zero
}

inline int encode(const Drive& d, uint8_t* out) {
    out[0] = SYNC_DRIVE;
    out[1] = TYPE_DRIVE;
    put_u16(out + 2, d.seq);
    for (int i = 0; i < NCH; ++i) put_u16(out + 4 + 2 * i, enc_amp(d.amp_pct[i]));
    for (int i = 0; i < NCH; ++i) put_u16(out + 12 + 2 * i, enc_phase(d.phase_deg[i]));
    put_u32(out + 20, (uint32_t)(clampf_(d.freq_hz, 0.0f, 1000.0f) * 1000.0f + 0.5f));
    put_u16(out + 24, crc16(out + 1, 23));   // CRC covers everything but the sync byte
    return DRIVE_LEN;
}

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
