// `pmw_fly`: the whole live loop in one process -- capture, pose, filter, control, coils.
//
// Nothing here decides a number. Every constant arrives in the config file written by
// `controller/control/native_config.py`, and an unknown or missing key is fatal at
// startup rather than defaulted, so a value cannot silently exist twice.
//
// ARMING. This binary REFUSES to energise without `--armed-token`, which only
// `controller/control/fly_native.py` issues, and it issues it only after
// `ai/thermal/coil_thermal.wait_until_safe()` has returned. That is not ceremony: the
// coils reach 80 C after four ramps, there is no temperature sensor, and the thermal
// model is a Python module this process deliberately does not reimplement. One writer to
// the stamp file, forever, and it is the supervisor. See `CLAUDE.md` "Safety".
//
// The supervisor also holds this process's stdin. EOF there means the supervisor is gone,
// and the coils are cut -- a host-side liveness backstop for free. It cannot cover the
// USB cable; nothing here can.
//
// See `control/theory.md` 27.

#include "control.h"
#include "link.h"
#include "rt.h"
#include "tracker.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace pmw;

namespace {

// ---- config ---------------------------------------------------------------------------
// A flat `key value` text file, not JSON: the whole point of this file is that it has no
// dependencies the firmware side does not, and a JSON parser is a library or 300 lines to
// carry for a dict of scalars. `native_config.py` writes it and owns every value.
struct Cfg {
    std::map<std::string, std::string> kv;

    void load(const std::string& path) {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot read config " + path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream is(line);
            std::string k;
            is >> k;
            std::string rest;
            std::getline(is, rest);
            size_t b = rest.find_first_not_of(" \t");
            kv[k] = (b == std::string::npos) ? "" : rest.substr(b);
        }
    }
    // Throws, never defaults. The rule `pmw::Config` sets, for the same reason.
    const std::string& str(const char* k) const {
        auto it = kv.find(k);
        if (it == kv.end()) throw std::runtime_error(std::string("config missing '") + k + "'");
        return it->second;
    }
    double num(const char* k) const { return std::stod(str(k)); }
    int integer(const char* k) const { return std::stoi(str(k)); }
    std::vector<double> vec(const char* k) const {
        std::istringstream is(str(k));
        std::vector<double> v;
        double x;
        while (is >> x) v.push_back(x);
        return v;
    }
};

ControlConfig control_cfg(const Cfg& c) {
    ControlConfig k{};
    k.accel_mm_s2 = c.num("accel_mm_s2");
    k.p0_pos = c.num("p0_pos");
    k.p0_vel = c.num("p0_vel");
    k.gate_sigma = c.num("gate_sigma");
    k.max_gated = c.integer("max_gated");
    k.sigma_normal = c.num("sigma_normal");
    k.max_coast_s = c.num("max_coast_s");
    k.gravity = c.num("gravity");
    k.tau_vel_s = c.num("tau_vel_s");
    k.ts = c.num("ts");
    k.f_hover = c.num("f_hover");
    k.g = c.num("g");
    k.k_lat = c.num("k_lat");
    k.mag_max = c.num("mag_max");
    k.freq_min = c.num("freq_min");
    k.freq_max = c.num("freq_max");
    k.freq_slew_hz_per_s = c.num("freq_slew_hz_per_s");
    k.vel_cutoff_hz = c.num("vel_cutoff_hz");
    k.suspect_mm = c.num("suspect_mm");
    auto K = c.vec("K");
    if (K.size() != 12) throw std::runtime_error("config K must be 12 numbers (2x6)");
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 6; ++j) k.K(i, j) = K[i * 6 + j];
    return k;
}

const char* KEYS[] = {"accel_mm_s2", "p0_pos", "p0_vel", "gate_sigma", "max_gated",
                      "sigma_normal", "max_coast_s", "gravity", "tau_vel_s", "ts",
                      "f_hover", "g", "k_lat", "mag_max", "freq_min", "freq_max",
                      "freq_slew_hz_per_s", "vel_cutoff_hz", "suspect_mm", "K"};


// A pose source with no camera: the C++ twin of `hover_controller_runner.stub_ticks` and
// of `fly(dry_run=True)`. It exists for the same reason they do -- the whole chain below
// the camera (gate, predictor, control law, wire encode, CSV, safety ladder) is testable
// without a rig, and shipping it untested because the rig is elsewhere is how untested
// code comes to look tested.
//
// `drop_every`/`drop_len` punch dropouts of a chosen length so the coast -> stale -> land
// ladder is exercised deliberately rather than whenever the robot happens to be lost, and
// `outlier_every` injects a jump the innovation gate must reject.
//
// **`--sim-outlier-every 1` is a BIAS, not a stream of outliers**, and the gate correctly
// ignores it: every pose carries the same offset, so after the first there is no
// innovation to reject and the filter is simply tracking a shifted trajectory. That is
// the right answer -- a constant offset is a calibration error, not an outlier -- and it
// is worth knowing before reading `0 gated` as a failure. The `MAX_GATED` escape, which
// needs a SUSTAINED excursion after normal tracking, is exercised in
// `native_parity.py --stage filter` against both implementations at once.
class SimPose {
public:
    SimPose(double rate_hz, int drop_every, int drop_len, int outlier_every)
        : dt_(1.0 / rate_hz), drop_every_(drop_every), drop_len_(drop_len),
          outlier_every_(outlier_every) {}

    //: The pose valid at `now`, or nullptr during a dropout. `seq` advances only when a
    //: pose is produced, which is what the control loop keys "new fix" on.
    const Vector3d* at(double now, std::uint64_t* seq, double* stamp) {
        // Seed the schedule from the first call. `steady_clock` counts from boot, so a
        // `next_` starting at 0 is decades in the past and every tick looks due -- the
        // source then fires at the CONTROL rate instead of its own, and the coast path
        // never runs. Measured while writing this: 5 coast ticks in 6001 at a nominal
        // 100 Hz source against a 500 Hz loop, where ~4800 were expected.
        if (next_ == 0.0) next_ = now;
        if (now < next_) return nullptr;
        next_ += dt_;
        n_++;
        // A dropout every `drop_every_` poses, lasting `drop_len_` of them.
        if (drop_every_ > 0 && (n_ % drop_every_) < (std::uint64_t)drop_len_) {
            n_lost_++;
            return nullptr;
        }
        const double t = n_ * dt_;
        // A slow lateral drift and a gentle climb: something the constant-velocity model
        // tracks, so a rejected update is visibly the gate and not the trajectory.
        xyz_ = Vector3d(2.0 * std::sin(0.35 * t), 1.5 * std::cos(0.28 * t),
                        60.0 + 3.0 * std::sin(0.2 * t));
        if (outlier_every_ > 0 && n_ % outlier_every_ == 0) {
            xyz_ += Vector3d(30.0, -25.0, 45.0);
            n_outlier_++;
        }
        *seq = ++pose_seq_;
        // The SHUTTER stamp, deliberately behind `now`: a real fix is already this old by
        // the time the loop sees it, and the filter must propagate from here, not from the
        // tick (19.6). A source that stamps `now` would hide a whole class of error.
        *stamp = now - 0.010;
        return &xyz_;
    }
    std::uint64_t n_lost() const { return n_lost_; }
    std::uint64_t n_outlier() const { return n_outlier_; }

private:
    Vector3d xyz_ = Vector3d::Zero();
    double dt_, next_ = 0.0;
    int drop_every_, drop_len_, outlier_every_;
    std::uint64_t n_ = 0, pose_seq_ = 0, n_lost_ = 0, n_outlier_ = 0;
};

std::atomic<bool> g_stop{false};

// stdin is the supervisor's liveness channel. EOF means it is gone; a line is a command.
void watch_stdin(std::vector<std::string>* cmds, std::mutex* m) {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::lock_guard<std::mutex> lk(*m);
        cmds->push_back(line);
    }
    g_stop.store(true);   // EOF: the supervisor died, and the coils must not outlive it
}

}  // namespace

int main(int argc, char** argv) {
    std::string cfg_path, port, token, csv_path;
    bool print_keys = false;
    double duration = 0.0, sim_hz = 0.0;
    int sim_drop_every = 0, sim_drop_len = 1, sim_outlier_every = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if (a == "--config") cfg_path = next();
        else if (a == "--port") port = next();
        else if (a == "--armed-token") token = next();
        else if (a == "--csv") csv_path = next();
        else if (a == "--seconds") duration = std::stod(next());
        else if (a == "--print-keys") print_keys = true;
        else if (a == "--sim-pose") sim_hz = std::stod(next());
        else if (a == "--sim-drop-every") sim_drop_every = std::stoi(next());
        else if (a == "--sim-drop-len") sim_drop_len = std::stoi(next());
        else if (a == "--sim-outlier-every") sim_outlier_every = std::stoi(next());
        else { std::fprintf(stderr, "unknown argument '%s'\n", a.c_str()); return 2; }
    }
    if (print_keys) {
        for (const char* k : KEYS) std::printf("%s\n", k);
        return 0;
    }
    if (cfg_path.empty()) { std::fprintf(stderr, "--config is required\n"); return 2; }

    Cfg cfg;
    try {
        cfg.load(cfg_path);
        (void)control_cfg(cfg);          // validate every key before anything opens
    } catch (const std::exception& e) {
        std::fprintf(stderr, "config: %s\n", e.what());
        return 2;
    }

    // THE ARMING GATE. Without the supervisor's token this process will not open the
    // serial port at all, so there is no path by which running `pmw_fly` by hand can
    // energise a coil with the thermal model unaware of it.
    if (token.empty()) {
        std::fprintf(stderr,
            "REFUSING to arm: no --armed-token.\n"
            "The coils are driven only through controller/control/fly_native.py, which\n"
            "holds the thermal model (ai/thermal/coil_thermal.py) and is the one writer\n"
            "of the energised-seconds stamp. Coils reach 80 C after four ramps and there\n"
            "is no temperature sensor. Run fly_native.py; do not run this by hand.\n");
        return 3;
    }

    const ControlConfig cc = control_cfg(cfg);
    std::printf("pmw_fly: config ok, %d keys\n", (int)(sizeof(KEYS) / sizeof(KEYS[0])));
    std::fflush(stdout);

    std::vector<std::string> cmds;
    std::mutex cmd_m;
    std::thread stdin_thread(watch_stdin, &cmds, &cmd_m);
    stdin_thread.detach();

    SerialLink link;
    if (!port.empty()) {
        try {
            link.open(port, (int)cfg.num("baud"));
        } catch (const std::exception& e) {
            std::fprintf(stderr, "serial: %s\n", e.what());
            return 4;
        }
        std::printf("serial: %s at %d\n", port.c_str(), (int)cfg.num("baud"));
    } else {
        std::printf("serial: none (--port omitted) -- computing but commanding nothing\n");
    }

    HoverController ctrl_x, ctrl_y;
    ctrl_x.init(cc);
    ctrl_y.init(cc);
    StatePredictor pred;
    pred.init(cc);
    ConstantVelocity filt(cc.accel_mm_s2, cc.p0_pos, cc.p0_vel);

    RowRing<8192> ring;
    std::atomic<bool> writing{true};
    std::thread writer;
    if (!csv_path.empty()) {
        writer = std::thread([&] {
            std::ofstream f(csv_path);
            f << "t,state,f_hz,x_mm,y_mm,z_mm,tilt_deg,tilt_az_deg,acc_tilt_x,acc_tilt_y,"
                 "cmd_tilt_x,cmd_tilt_y,mag,az,armed,spin,lost,i_a,i_b,i_c,i_d\n";
            Row r;
            auto num = [&](double v) {
                // NaN -> blank. The 21-column schema's rule, carried across the port:
                // a 0.00 here would read as measured-and-level.
                if (std::isnan(v)) return std::string();
                char b[32];
                std::snprintf(b, sizeof(b), "%.4f", v);
                return std::string(b);
            };
            while (writing.load() || ring.pop(r)) {
                bool got = ring.pop(r);
                if (!got) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    continue;
                }
                f << num(r.t) << ',' << r.state << ',' << num(r.f_hz) << ',' << num(r.x)
                  << ',' << num(r.y) << ',' << num(r.z) << ',' << num(r.tilt) << ','
                  << num(r.tilt_az) << ',' << num(r.acc_tx) << ',' << num(r.acc_ty) << ','
                  << num(r.cmd_tx) << ',' << num(r.cmd_ty) << ',' << num(r.mag) << ','
                  << num(r.az) << ',' << r.armed << ',' << r.spin << ',' << r.lost;
                for (int i = 0; i < 4; ++i) f << ',' << num(r.i[i]);
                f << '\n';
            }
        });
    }

    const double ts = cc.ts;
    // `computation` from a measurement, not a guess. Seeded from the config and refined
    // below once the tick body has been timed; a thread that overruns it is demoted.
    if (!set_realtime(ts, cfg.num("rt_computation_s"), cfg.num("rt_constraint_s")))
        std::fprintf(stderr, "warning: could not set THREAD_TIME_CONSTRAINT_POLICY -- "
                             "the loop will run, with the scheduler's tail\n");

    ClockStats clk;
    const double t0 = now_s();
    double next = t0, last = t0;
    double t_fix = t0;              // WALL time of the last fix, not frame time
    double f_cmd = cc.f_hover, f_hat = cc.f_hover;
    std::uint64_t n_gate_rej = 0, n_coast = 0, n_stale = 0, last_seq = 0;
    double t_meas = t0;
    const bool rows_on = !csv_path.empty();
    std::unique_ptr<SimPose> sim;
    if (sim_hz > 0.0)
        sim.reset(new SimPose(sim_hz, sim_drop_every, sim_drop_len, sim_outlier_every));
    std::vector<std::string> lines;

    while (!g_stop.load()) {
        pace_until(next);
        const double now = now_s();
        clk.note(now - last, ts);
        last = now;
        next += ts;
        if (next < now) next = now + ts;   // fell behind: resync, and it was counted

        if (link.is_open()) {
            lines.clear();
            link.poll(now, &lines);
        }
        {
            std::lock_guard<std::mutex> lk(cmd_m);
            for (auto& c : cmds) {
                if (c == "stop") { g_stop.store(true); }
                else if (!c.empty() && link.is_open()) link.send_line(c);
            }
            cmds.clear();
        }

        // ---- pose -> filter -> predictor -> control ---------------------------------
        bool fresh = false;
        double dt_meas = 0.0;
        if (sim) {
            std::uint64_t pseq = 0;
            double stamp = 0.0;
            const Vector3d* z = sim->at(now, &pseq, &stamp);
            // A pose is new only if its SEQUENCE advanced and its stamp is later than the
            // last fix's. A repeat, or one older than the filter, carries no information
            // about now and must not drag the state.
            if (z && pseq > last_seq && stamp > t_meas) {
                last_seq = pseq;
                // Propagate to the SHUTTER stamp, update there, then forward to now.
                // 19.6's t_pred rule: the next coast then spans the fix's full pipeline
                // age, which is the latency compensation, and it is free here because the
                // filter and the pose source share one clock.
                filt.predict(std::max(stamp - t_meas, 0.0));
                // R inflation from the frame's own quality. One term, because
                // `discrepancy_mm` is the only quality signal in millimetres and so the
                // only one commensurate with R without a scale nobody has measured. The
                // hard gate stays underneath: inflation alone would let a quarter-turn
                // branch flip through with merely a larger R. (Simulated source: quality
                // is nominal, so the factor is 1 and the gate is what acts.)
                const double s_inf = 1.0;
                Matrix3d R = Matrix3d::Zero();
                R(0, 0) = R(1, 1) = 0.05 * 0.05 * s_inf;
                R(2, 2) = 0.35 * 0.35 * s_inf;
                const bool took = filt.update(*z, R, cc.gate_sigma, cc.max_gated);
                if (!took) n_gate_rej++;
                dt_meas = stamp - t_meas;
                t_meas = stamp;
                t_fix = now;                      // WALL time, for the lost-land guard
                fresh = true;
                // Raw position, filtered VELOCITY. `filter.py` measures filtered position
                // as 1.4% WORSE than raw -- depth error autocorrelates at r = 0.966 -- so
                // the filter is a rejector and a rate source, never a smoother.
                pred.update(took ? *z : filt.value(), filt.rate(), stamp);
            }
        }
        if (!fresh) {
            pred.predict(Command{f_cmd, f_hat}, now - pred.t());
            n_coast++;
            if (pred.stale()) n_stale++;
        }

        const Vector3d p = pred.xyz_mm();
        const Vector2d ref_p(0.0, cfg.num("hover_z_m")), ref_v(0.0, 0.0), ref_a(0.0, 0.0);
        const Vector2d ux = ctrl_x.step(p(0) * 1e-3, p(2) * 1e-3, dt_meas, fresh,
                                        ref_p, ref_v, ref_a);
        const Vector2d uy = ctrl_y.step(p(1) * 1e-3, p(2) * 1e-3, dt_meas, fresh,
                                        ref_p, ref_v, ref_a);
        f_cmd = ux(1);

        if (rows_on) {
            Row r{};
            r.t = now - t0;
            r.state = link.stats().fw_state;
            r.armed = 0;
            r.lost = (int)(sim ? sim->n_lost() : 0);
            r.f_hz = f_cmd;
            r.x = p(0); r.y = p(1); r.z = p(2);
            r.tilt = r.tilt_az = r.acc_tx = r.acc_ty = NAN;   // no attitude source here
            r.cmd_tx = ux(0); r.cmd_ty = uy(0);
            r.mag = std::hypot(ux(0), uy(0));
            r.az = std::fmod(std::atan2(uy(0), ux(0)) * 180.0 / M_PI + 360.0, 360.0);
            for (int i = 0; i < 4; ++i) r.i[i] = NAN;
            std::snprintf(r.spin, sizeof(r.spin), "%s", "");
            ring.push(r);
        }

        // THE LADDER'S LAST RUNG. Wall clock, not frame time: the hazard includes the
        // pose producer stalling, and a stalled producer freezes frame time -- so a
        // frame-time deadline never expires in exactly the case it exists for.
        if (now - t_fix > cfg.num("lost_land_s") && link.is_open()) {
            std::printf("TRACKING LOST for %.2fs with the coils driven -- stopping\n",
                        now - t_fix);
            link.send_stop();
            break;
        }

        if (duration > 0.0 && now - t0 >= duration) break;
    }

    if (link.is_open()) link.send_stop();
    writing.store(false);
    if (writer.joinable()) writer.join();

    const double elapsed = now_s() - t0;
    std::printf("%s\n", clk.summary(1.0 / ts, elapsed).c_str());
    if (sim)
        std::printf("pose: %llu coast tick(s), %llu stale, %llu gated, %llu dropout(s), "
                    "%llu outlier(s) injected\n",
                    (unsigned long long)n_coast, (unsigned long long)n_stale,
                    (unsigned long long)n_gate_rej, (unsigned long long)sim->n_lost(),
                    (unsigned long long)sim->n_outlier());
    if (ring.n_dropped())
        std::printf("csv: %llu row(s) dropped -- the ring was full\n",
                    (unsigned long long)ring.n_dropped());
    if (link.is_open()) {
        const LinkStats& s = link.stats();
        std::vector<double> r = s.rtt_s;
        std::sort(r.begin(), r.end());
        auto q = [&](double f) { return r.empty() ? 0.0 : r[(size_t)(f * (r.size() - 1))] * 1e3; };
        std::printf("link: sent %llu, board accepted %u, crc/desync %u, drop %.3f%% | "
                    "rtt ms med %.2f p95 %.2f max %.2f (n=%zu, UPPER BOUND on one way)\n",
                    (unsigned long long)s.n_sent, s.fw_n_rx, s.fw_n_crc,
                    100.0 * s.drop_rate(), q(0.5), q(0.95), r.empty() ? 0.0 : r.back() * 1e3,
                    r.size());
    }
    return 0;
}
