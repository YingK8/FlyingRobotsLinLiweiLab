#include "SyncLed.h"

namespace sync_led {

bool SyncLed::advanceTo(uint32_t now_ms) {
  if (signal_ == Signal::None) return false;
  const Signal running = signal_;
  const uint32_t segment_ms = (running == Signal::Code) ? timing_.code_ms
                                      : (edgesWritten_ == 0)     ? timing_.on_ms
                                                                 : timing_.gap_ms;
  if (now_ms - lastEdgeMs_ < segment_ms) return false;
  lastEdgeMs_ = now_ms;
  if (running == Signal::Code) {
    // start bit, then the 7 Hamming bits MSB first, then dark
    if (edgesWritten_ == 7) {
      signal_ = Signal::None;
      return setOn(false);
    }
    return setOn((word_ >> (6 - edgesWritten_++)) & 1);
  }
  if (running == Signal::Blink) {
    signal_ = Signal::None;
    return setOn(false);
  }
  if (++edgesWritten_ < 2) return setOn(running == Signal::Stop);
  signal_ = Signal::None;
  return setOn(running == Signal::Start);
}

bool SyncLed::beginSignal(Signal signal, bool on, uint32_t now_ms) {
  if (!setOn(on)) return false;
  signal_ = signal;
  edgesWritten_ = 0;
  lastEdgeMs_ = now_ms;
  return true;
}

bool SyncLed::endSignal(bool on) {
  signal_ = Signal::None;
  edgesWritten_ = 0;
  return setOn(on);
}

bool SyncLed::setOn(bool on) {
  if (isOn_ == on) return false;
  isOn_ = on;
  return true;
}

}  // namespace sync_led
