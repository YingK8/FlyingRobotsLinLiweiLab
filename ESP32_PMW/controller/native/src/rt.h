// Real-time helpers for the control thread: the clock, the scheduling policy, the pacing,
// and a lock-free row queue so logging never touches the tick.
//
// See `control/theory.md` 27.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <mach/mach_time.h>
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>

namespace pmw {

//: THE clock. `std::chrono::steady_clock` and nothing else.
//:
//: On Darwin libc++ this is `mach_absolute_time`, the same counter
//: `clock_gettime_nsec_np(CLOCK_UPTIME_RAW)` reads. `tracker.h::now_s()` already stamps
//: every frame with it, and the filter must propagate from the SHUTTER stamp rather than
//: the tick time (`control/theory.md` 19.6), so one clock is a correctness requirement
//: and not a style preference. `system_clock` is wrong here -- it steps when the machine
//: syncs time, and a step backwards through a filter's `dt` is a divide by a negative.
inline double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

//: Ask the kernel to schedule this thread as real time.
//:
//: What this buys, honestly: it moves the thread onto the real-time run queue ahead of
//: every timeshare thread, which removes the SCHEDULER half of the tens-of-milliseconds
//: outliers the Python loop shows (`theory.md` 19.13 measured dt max 16.70 ms at 500 Hz).
//: What it does not buy: a guarantee. A thread that overruns `computation` is demoted;
//: page faults, malloc arena locks and the USB and WindowServer interrupt paths can still
//: stall it. Expect p99.9 under a millisecond and a worst case of a few, not zero.
//:
//: `computation_s` must come from a MEASUREMENT of the tick body, not a guess. Set it too
//: low and the kernel demotes the thread for overrunning; too high and it is asking for a
//: reservation the scheduler may refuse outright.
inline bool set_realtime(double period_s, double computation_s, double constraint_s) {
    mach_timebase_info_data_t tb;
    if (mach_timebase_info(&tb) != KERN_SUCCESS) return false;
    const double ns_to_abs = double(tb.denom) / double(tb.numer);
    thread_time_constraint_policy_data_t p;
    p.period = (uint32_t)(period_s * 1e9 * ns_to_abs);
    p.computation = (uint32_t)(computation_s * 1e9 * ns_to_abs);
    p.constraint = (uint32_t)(constraint_s * 1e9 * ns_to_abs);
    p.preemptible = 0;
    return thread_policy_set(pthread_mach_thread_np(pthread_self()),
                             THREAD_TIME_CONSTRAINT_POLICY, (thread_policy_t)&p,
                             THREAD_TIME_CONSTRAINT_POLICY_COUNT) == KERN_SUCCESS;
}

//: Sleep to just before `deadline`, then spin.
//:
//: macOS sleep granularity is ~1 ms, which at a 2 ms period is the whole question --
//: `theory.md` 19.13 expected it to force a busy-wait and found the Python loop got away
//: without one. A bounded spin costs ~12 % of one core at 500 Hz and removes the
//: granularity term outright, which is a trade worth making in a process whose entire
//: purpose is the tail of the dt distribution.
inline void pace_until(double deadline, double spin_s = 250e-6) {
    const double t = now_s();
    if (deadline - t > spin_s)
        std::this_thread::sleep_for(std::chrono::duration<double>(deadline - t - spin_s));
    while (now_s() < deadline) { /* spin */ }
}

//: Running dt statistics. The loop used to resync in silence, so a loop that never made
//: its period looked exactly like one that always did.
class ClockStats {
public:
    void note(double dt, double period) {
        dts_.push_back(dt);
        n_++;
        if (dt > 1.5 * period) n_overrun_++;
    }
    std::string summary(double design_hz, double elapsed) const {
        if (dts_.empty()) return "clock: no ticks";
        std::vector<double> v = dts_;
        std::sort(v.begin(), v.end());
        auto q = [&](double f) { return v[(size_t)(f * (v.size() - 1))] * 1e3; };
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "clock: %llu ticks, design %.0f Hz, achieved %.1f Hz | dt ms "
                      "med %.2f p95 %.2f p99.9 %.2f max %.2f | %llu overrun (%.1f%%)",
                      (unsigned long long)n_, design_hz, elapsed > 0 ? n_ / elapsed : 0.0,
                      q(0.5), q(0.95), q(0.999), v.back() * 1e3,
                      (unsigned long long)n_overrun_, 100.0 * n_overrun_ / n_);
        return buf;
    }
    std::uint64_t n() const { return n_; }
    std::uint64_t n_overrun() const { return n_overrun_; }

private:
    std::vector<double> dts_;
    std::uint64_t n_ = 0, n_overrun_ = 0;
};

//: One CSV row, as plain data. NaN means "not measured" and formats to BLANK, never 0 --
//: a 0.00 in this schema reads as measured-and-level, which is the opposite of unknown.
struct Row {
    double t;
    int state, armed, lost;
    double f_hz, x, y, z, tilt, tilt_az, acc_tx, acc_ty, cmd_tx, cmd_ty, mag, az;
    double i[4];
    char spin[12];
};

//: Single-producer / single-consumer ring. The control thread only ever bumps `head`; the
//: writer thread only ever bumps `tail`. No lock, no allocation, and a FULL ring drops
//: the row and counts it -- a dropped log line is always better than a missed deadline.
template <int N>
class RowRing {
public:
    bool push(const Row& r) {
        const auto h = head_.load(std::memory_order_relaxed);
        const auto next = (h + 1) % N;
        if (next == tail_.load(std::memory_order_acquire)) {
            n_dropped_++;
            return false;
        }
        buf_[h] = r;
        head_.store(next, std::memory_order_release);
        return true;
    }
    bool pop(Row& out) {
        const auto t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) return false;
        out = buf_[t];
        tail_.store((t + 1) % N, std::memory_order_release);
        return true;
    }
    std::uint64_t n_dropped() const { return n_dropped_; }

private:
    Row buf_[N];
    std::atomic<int> head_{0}, tail_{0};
    std::uint64_t n_dropped_ = 0;
};

}  // namespace pmw
