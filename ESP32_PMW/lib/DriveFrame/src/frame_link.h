// One byte stream carrying two framings: ASCII commands and binary DRIVE frames.
//
// Replaces `SerialComm` on `main_flight`'s hot path only. `SerialComm` is left alone --
// three other mains use it -- but it cannot serve here: it accumulates into an Arduino
// `String` one character at a time, which is a heap allocation per command at 200 Hz, and
// it returns at most one line per call.
//
// See `control/theory.md` 26.

#pragma once

#include <stdint.h>
#include <string.h>

#include "drive_frame.h"

namespace drive_frame {

// The ASCII half. 128 is `SerialComm::MAX_LINE_LEN`, kept so nothing that fits today
// stops fitting. Over-long lines are truncated and FLAGGED rather than dropped whole:
// silently discarding a command the operator typed is the worse failure.
static const int MAX_LINE = 128;

// A frame that stalls mid-flight must not hold the parser out of ASCII forever, or a
// single stray high-bit byte would make `stop` unreachable until 25 more bytes arrive.
// 2 ms is ~184 byte-times at 921600, so a real frame never trips it.
static const uint32_t FRAME_GAP_US = 2000;

enum class Got { None, Line, Frame, Overflow };

//: Byte-at-a-time framer. Feed it bytes; it tells you when something completed.
//:
//: Deliberately not a `Stream` reader: the firmware and the host both have their own way
//: of getting bytes, and the framing logic is the only part worth sharing.
class Framer {
public:
    void reset() { n_ = 0; in_frame_ = false; over_ = false; }

    //: Feed one byte, with a microsecond timestamp for the inter-byte gap check.
    Got feed(uint8_t b, uint32_t now_us) {
        if (in_frame_) {
            // An inter-byte gap this long means the frame's tail is not coming. Drop back
            // to ASCII rather than swallowing whatever arrives next as frame payload.
            if ((uint32_t)(now_us - last_us_) > FRAME_GAP_US) {
                in_frame_ = false;
                n_ = 0;
                n_gap_++;
            } else {
                last_us_ = now_us;
                buf_[n_++] = b;
                if (n_ < DRIVE_LEN) return Got::None;
                in_frame_ = false;
                n_ = 0;
                if (decode(buf_, drive_)) { n_rx_++; return Got::Frame; }
                n_crc_++;
                return Got::None;
            }
        }
        // High bit set: a frame may be starting. Only from an EMPTY accumulator -- a
        // high-bit byte part-way through a line is corruption in a line, not the start of
        // a frame, and treating it as one would eat the next 25 bytes of good ASCII.
        if ((b & 0x80) && n_ == 0) {
            if (b == SYNC_DRIVE) {
                in_frame_ = true;
                last_us_ = now_us;
                buf_[0] = b;
                n_ = 1;
                return Got::None;
            }
            return Got::None;               // some other high byte: noise, ignore
        }
        if (b & 0x80) { n_ = 0; over_ = false; return Got::None; }   // corrupt line, drop it
        if (b == '\n' || b == '\r') {
            if (n_ == 0) return Got::None;  // blank line, or the other half of a CRLF
            line_[n_ < MAX_LINE ? n_ : MAX_LINE - 1] = '\0';
            bool was_over = over_;
            n_ = 0;
            over_ = false;
            return was_over ? Got::Overflow : Got::Line;
        }
        if (n_ < MAX_LINE - 1) line_[n_++] = (char)b;
        else over_ = true;                  // keep counting to the newline, but flag it
        return Got::None;
    }

    const char* line() const { return line_; }
    const Drive& drive() const { return drive_; }
    uint16_t n_rx() const { return n_rx_; }
    uint16_t n_crc() const { return n_crc_; }
    uint16_t n_gap() const { return n_gap_; }

private:
    char line_[MAX_LINE] = {0};
    uint8_t buf_[DRIVE_LEN] = {0};
    Drive drive_ = {};
    int n_ = 0;
    bool in_frame_ = false, over_ = false;
    uint32_t last_us_ = 0;
    uint16_t n_rx_ = 0, n_crc_ = 0, n_gap_ = 0;
};

}  // namespace drive_frame
