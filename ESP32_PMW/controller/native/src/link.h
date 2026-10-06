// The host end of the serial link: a non-blocking POSIX serial port that speaks the
// binary DRIVE frame and the ASCII commands side by side.
//
// The layout comes from `lib/DriveFrame/src/drive_frame.h`, included directly rather than
// copied -- the firmware and this file are the two ends of one format and there is one
// definition of it. See `control/theory.md` 26.
#pragma once

#include "drive_frame.h"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace pmw {

//: What the link has measured about itself. "Reliable" has to be a measurement.
struct LinkStats {
    std::uint64_t n_sent = 0;        // DRIVE frames written
    std::uint64_t n_ack = 0;         // ACKs parsed
    std::uint64_t n_eagain = 0;      // writes the kernel could not take immediately
    std::uint64_t n_ack_bad = 0;     // ACK-shaped bytes that failed CRC
    std::uint16_t fw_n_rx = 0;       // frames the FIRMWARE says it accepted
    std::uint16_t fw_n_crc = 0;      // frames the firmware rejected
    std::uint8_t  fw_state = 0;
    std::vector<double> rtt_s;       // round trip, host -> board -> host
    //: 1 - (frames the board accepted) / (frames we sent). `fw_n_crc` separates
    //: *corrupted* from *never arrived*: two different faults with two different fixes.
    double drop_rate() const {
        return n_sent == 0 ? 0.0 : 1.0 - double(fw_n_rx) / double(n_sent);
    }
};

class SerialLink {
public:
    ~SerialLink();
    //: Open `dev` at `baud`. Throws with errno's message on failure. DTR and RTS are left
    //: LOW so opening the port does not reset the board -- an ESP32 reset mid-flight
    //: would drop the coils to whatever the bootloader leaves them at.
    void open(const std::string& dev, int baud);
    void close();
    bool is_open() const { return fd_ >= 0; }

    //: One DRIVE frame. Non-blocking: a write the kernel cannot take is COUNTED and
    //: dropped, never waited on. At 26 bytes against a 921600 line this cannot happen in
    //: steady state, and if it does the tick's deadline matters more than the frame --
    //: the next one is 5 ms away and carries the same intent.
    bool send_drive(const float amp_pct[4], const float phase_deg[4], float freq_hz,
                    double now_s);

    //: One ASCII command plus '\n'. For `seq=`, `probe=`, `stop` -- rare, and worth
    //: reading in a serial monitor.
    bool send_line(const std::string& s);

    //: `stop`, sent the way a corrupted link cannot eat: a leading newline terminates any
    //: partial ASCII line, the inter-byte gap terminates any partial frame, and `stop` is
    //: idempotent so repeating it costs nothing. Never a flag inside the binary frame --
    //: a stop that depends on a CRC passing is a stop that a bit error can swallow.
    void send_stop();

    //: Drain everything waiting. ASCII lines are appended to `lines`; ACK frames update
    //: the stats. One read syscall, not one per byte.
    void poll(double now_s, std::vector<std::string>* lines);

    const LinkStats& stats() const { return st_; }
    std::uint16_t next_seq() const { return seq_; }

private:
    void on_ack(const drive_frame::Ack& a, double now_s);

    int fd_ = -1;
    std::uint16_t seq_ = 0;
    LinkStats st_;
    // Send times keyed by the low bits of the sequence number, for the round-trip
    // measurement. 1024 entries is 5 s at 200 Hz, far longer than any plausible RTT.
    static const int RTT_RING = 1024;
    double sent_at_[RTT_RING] = {0};
    std::string pending_;   // partial ASCII line across reads
    std::uint8_t ack_buf_[drive_frame::ACK_LEN] = {0};
    int ack_n_ = 0;
};

}  // namespace pmw
