#pragma once

#include <stdint.h>

namespace sync_led {

struct PulseTiming {
  uint16_t on_ms = 30;
  uint16_t gap_ms = 30;
  uint16_t blink_ms = 30;
};

enum class Signal : uint8_t { None, Start, Stop, Blink };

class SyncLed {
 public:
  explicit SyncLed(const PulseTiming &timing = PulseTiming()) : timing_(timing) {}

  const PulseTiming &timing() const { return timing_; }
  Signal signal() const { return signal_; }
  bool isSendingSignal() const { return signal_ != Signal::None; }
  bool isOn() const { return isOn_; }

  bool sendStartSignal(uint32_t now_ms) { return beginSignal(Signal::Start, true, now_ms); }
  bool sendStopSignal(uint32_t now_ms) { return beginSignal(Signal::Stop, false, now_ms); }
  bool sendBlink(uint32_t now_ms) { return beginSignal(Signal::Blink, true, now_ms); }

  bool holdOn() { return endSignal(true); }
  bool holdOff() { return endSignal(false); }

  bool advanceTo(uint32_t now_ms) {
    if (signal_ == Signal::None) return false;
    const Signal running = signal_;
    const uint32_t segment_ms = (edgesWritten_ == 0) ? timing_.on_ms : timing_.gap_ms;
    if (now_ms - lastEdgeMs_ < segment_ms) return false;
    lastEdgeMs_ = now_ms;
    if (running == Signal::Blink) {
      signal_ = Signal::None;
      return setOn(false);
    }
    if (++edgesWritten_ < 2) return setOn(running == Signal::Stop);
    signal_ = Signal::None;
    return setOn(running == Signal::Start);
  }

 private:
  bool beginSignal(Signal signal, bool on, uint32_t now_ms) {
    if (!setOn(on)) return false;
    signal_ = signal;
    edgesWritten_ = 0;
    lastEdgeMs_ = now_ms;
    return true;
  }

  bool endSignal(bool on) {
    signal_ = Signal::None;
    edgesWritten_ = 0;
    return setOn(on);
  }

  bool setOn(bool on) {
    if (isOn_ == on) return false;
    isOn_ = on;
    return true;
  }

  PulseTiming timing_;
  Signal signal_ = Signal::None;
  uint8_t edgesWritten_ = 0;
  uint32_t lastEdgeMs_ = 0;
  bool isOn_ = false;
};

}  // namespace sync_led
