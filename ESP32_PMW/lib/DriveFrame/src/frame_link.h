#pragma once

#include <stdint.h>
#include <string.h>

#include "drive_frame.h"

namespace drive_frame {

static const int MAX_LINE = 128;
static const uint32_t FRAME_GAP_US = 2000;

enum class Got { None, Line, Frame, Overflow };

class Framer {
public:
    void reset() { n_ = 0; in_frame_ = false; over_ = false; }

    Got feed(uint8_t b, uint32_t now_us) {
        if (in_frame_) {
            // gap this long means the tail is not coming
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
        // high bit set: a frame may be starting, but only from an empty accumulator
        if ((b & 0x80) && n_ == 0) {
            if (b == SYNC_DRIVE) {
                in_frame_ = true;
                last_us_ = now_us;
                buf_[0] = b;
                n_ = 1;
                return Got::None;
            }
            return Got::None;     // some other high byte: noise, ignore
        }
        if (b & 0x80) { n_ = 0; over_ = false; return Got::None; }   // corrupt out: drop it
        if (b == '\n' || b == '\r') {
            if (n_ == 0) return Got::None;   // blank out, or the other half of CRLF
            line_[n_ < MAX_LINE ? n_ : MAX_LINE - 1] = '\0';
            bool was_over = over_;
            n_ = 0;
            over_ = false;
            return was_over ? Got::Overflow : Got::Line;
        }
        if (n_ < MAX_LINE - 1) line_[n_++] = (char)b;
        else over_ = true;        // keep counting to the newline, but flag it
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
