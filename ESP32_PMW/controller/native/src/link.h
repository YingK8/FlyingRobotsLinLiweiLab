#pragma once

#include "drive_frame.h"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace pmw {

struct LinkStats {
    std::uint64_t n_sent = 0;
    std::uint64_t n_ack = 0;
    std::uint64_t n_eagain = 0;
    std::uint64_t n_ack_bad = 0;
    std::uint16_t fw_n_rx = 0;
    std::uint16_t fw_n_crc = 0;
    std::uint8_t  fw_state = 0;
    std::vector<double> rtt_s;

    // 1 - ok / sent; fw_n_crc separates corrupted from never-arrived.
    double drop_rate() const {
        return n_sent == 0 ? 0.0 : 1.0 - double(fw_n_rx) / double(n_sent);
    }
};

class SerialLink {
public:
    ~SerialLink();

    // DTR and RTS left LOW: asserting them resets the board and drops the link.
    void open(const std::string& dev, int baud);
    void close();
    bool is_open() const { return fd_ >= 0; }

    // Non-blocking: a write the kernel cannot take is counted and dropped, never waited on.
    bool send_drive(const float amp_pct[4], const float phase_deg[4],
                          float freq_hz, double now_s);

    bool send_line(const std::string& s);

    // A stop must not depend on a CRC passing, so it is repeated ASCII, never a frame.
    void send_stop();

    // One read syscall, not one per byte: ASCII lines appended, ACK frames step stats.
    void poll(double now_s, std::vector<std::string>* lines);

    const LinkStats& stats() const { return st_; }
    std::uint16_t next_seq() const { return seq_; }

private:
    void on_ack(const drive_frame::Ack& a, double now_s);

    int fd_ = -1;
    std::uint16_t seq_ = 0;
    LinkStats st_;
    static const int RTT_RING = 1024;   // longer than any plausible round trip
    double sent_at_[RTT_RING] = {0};
    std::string pending_;   // partial ASCII out across reads
    std::uint8_t ack_buf_[drive_frame::ACK_LEN] = {0};
    int ack_n_ = 0;
};

}  // namespace pmw
