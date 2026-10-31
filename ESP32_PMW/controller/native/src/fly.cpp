
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

// A pose source with no camera, so the whole chain below the camera stays testable.
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
    // Throws, never defaults: a missing key must be fatal at startup.
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


class SimPose {
public:
    SimPose(double rate_hz, int drop_every, int drop_len, int outlier_every)
        : dt_(1.0 / rate_hz), drop_every_(drop_every), drop_len_(drop_len),
          outlier_every_(outlier_every) {}

    const Vector3d* at(double now, std::uint64_t* seq, double* stamp) {
        // seed the schedule from the first call, or every tick looks due
        if (next_ == 0.0) next_ = now;
        if (now < next_) return nullptr;
        next_ += dt_;
        n_++;
        if (drop_every_ > 0 && (n_ % drop_every_) < (std::uint64_t)drop_len_) {
            n_lost_++;
            return nullptr;
        }
        const double t = n_ * dt_;
        xyz_ = Vector3d(2.0 * std::sin(0.35 * t), 1.5 * std::cos(0.28 * t),
                        60.0 + 3.0 * std::sin(0.2 * t));
        if (outlier_every_ > 0 && n_ % outlier_every_ == 0) {
            xyz_ += Vector3d(30.0, -25.0, 45.0);
            n_outlier_++;
        }
        // the shutter stamp sits behind now, as a real fix does
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

// stdin is the supervisor's liveness channel: EOF means it is gone.
void watch_stdin(std::vector<std::string>* cmds, std::mutex* m) {
    std::string line;
    while (std::getline(std::cin, line)) {
        std::lock_guard<std::mutex> lk(*m);
        cmds->push_back(line);
    }
    g_stop.store(true);
}

}

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
        (void)control_cfg(cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "config: %s\n", e.what());
        return 2;
    }

    // arming gate: without the supervisor's token, do not even open the port
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
                // NaN as blank: a 0.00 would read as measured-and-level
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
    if (!set_realtime(ts, cfg.num("rt_computation_s"), cfg.num("rt_constraint_s")))
        std::fprintf(stderr, "warning: could not set THREAD_TIME_CONSTRAINT_POLICY -- "
                             "the loop will run, with the scheduler's tail\n");

    ClockStats clk;
    const double t0 = now_s();
    double next = t0, last = t0;
    double t_fix = t0;
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
        if (next < now) next = now + ts;

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

        bool fresh = false;
        double dt_meas = 0.0;
        if (sim) {
            std::uint64_t pseq = 0;
            double stamp = 0.0;
            const Vector3d* z = sim->at(now, &pseq, &stamp);
            // a pose is new only if its seq advanced and its stamp is later
            if (z && pseq > last_seq && stamp > t_meas) {
                last_seq = pseq;
                // propagate to the shutter stamp, update there, then forward to now
                filt.predict(std::max(stamp - t_meas, 0.0));
                // inflation alone lets a branch flip through; the hard gate stays underneath
                const double s_inf = 1.0;
                Matrix3d R = Matrix3d::Zero();
                R(0, 0) = R(1, 1) = 0.05 * 0.05 * s_inf;
                R(2, 2) = 0.35 * 0.35 * s_inf;
                const bool took = filt.update(*z, R, cc.gate_sigma, cc.max_gated);
                if (!took) n_gate_rej++;
                dt_meas = stamp - t_meas;
                t_meas = stamp;
                t_fix = now;
                fresh = true;
                // raw position, filtered velocity: the filter is a rejector, not a smoother
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
            r.tilt = r.tilt_az = r.acc_tx = r.acc_ty = NAN;
            r.cmd_tx = ux(0); r.cmd_ty = uy(0);
            r.mag = std::hypot(ux(0), uy(0));
            r.az = std::fmod(std::atan2(uy(0), ux(0)) * 180.0 / M_PI + 360.0, 360.0);
            for (int i = 0; i < 4; ++i) r.i[i] = NAN;
            std::snprintf(r.spin, sizeof(r.spin), "%s", "");
            ring.push(r);
        }

        // wall clock, not frame time: a stalled producer freezes frame time
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
