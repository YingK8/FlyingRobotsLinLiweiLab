#!/usr/bin/env python3
"""Every number the native controller needs, in one dict, from the Python that owns it.

Same contract `pose/stereo_native.native_config()` has with `pmw::Config`, extended to the
control core: **no tuning constant is written down in C++.** Each one lives in exactly one
Python module -- `pose/filter.py`, `control/predictor.py`, `control/constants.py`, or the
generated `hover_controller.json` -- and arrives here by reference, so there is no second
copy to drift.

The C++ loader THROWS on a missing key rather than defaulting. That is the half that makes
this work: a number silently defaulting on one side is a number with two values, and the
failure would be a controller flying gains nobody chose.

    uv run python controller/control/native_config.py

See `control/theory.md` 27.
"""

from __future__ import annotations

import json
from pathlib import Path

from controller.control import constants as C
from controller.control import predictor as PR
from controller.control import simulate_hover as SH
from controller.pose import filter as F

HERE = Path(__file__).resolve().parent
GAINS = HERE / "hover_controller.json"

#: Every key `pmw::ControlConfig` requires. Kept here so `demo()` can assert the emitted
#: dict covers exactly this set -- a key added to the C++ struct and forgotten here is a
#: build that throws at startup, which is the loud failure; a key emitted and NOT read is
#: dead weight, which is the quiet one. Both are caught by comparing against
#: `pmw_pose.control_config_keys()`.
REQUIRED = (
    "accel_mm_s2", "p0_pos", "p0_vel", "gate_sigma", "max_gated", "sigma_normal",
    "max_coast_s", "gravity", "tau_vel_s",
    "ts", "f_hover", "g", "k_lat",
    "mag_max", "freq_min", "freq_max", "freq_slew_hz_per_s",
    "vel_cutoff_hz", "suspect_mm", "K",
)

#: Keys the BINARY needs on top of the control core's -- runtime rather than control law,
#: so they are not in `ControlConfig` and not in the parity comparison.
RUNTIME = ("baud", "lost_land_s", "rt_computation_s", "rt_constraint_s",
           "hover_z_m")


def control_config(gains: dict | None = None, noise=None) -> dict:
    """The control-core config the native binary and the parity harness both consume."""

    if gains is None:
        gains = json.loads(GAINS.read_text())
    if noise is None:
        from controller.pose.noise import NoiseModel
        noise = NoiseModel.load()
    lim, par, des = gains["limits"], gains["params"], gains["design"]
    return {
        # pose/filter.py -- the position channel only. The normal channel uses
        # `accel_normal` and its own p0; it is not in the control path and is not ported.
        "accel_mm_s2": F.ACCEL_MM_S2,
        "p0_pos": 100.0,          # `_ConstantVelocity(accel, p0_pos=100.0, p0_vel=1e4)`
        "p0_vel": 1e4,
        "gate_sigma": F.GATE_SIGMA,
        "max_gated": F.MAX_GATED,
        "sigma_normal": noise.sigma_normal,
        # control/predictor.py
        "max_coast_s": PR.MAX_COAST_S,
        "gravity": PR.GRAVITY,
        "tau_vel_s": PR.TAU_VEL_S,
        # the generated gain file -- never hand-copied, always read
        "ts": des["ts"],
        "f_hover": par["f_hover"],
        "g": par["g"],
        "k_lat": par["k_lat"],
        "mag_max": lim["mag_max"],
        "freq_min": lim["freq_min"],
        "freq_max": lim["freq_max"],
        "freq_slew_hz_per_s": lim["freq_slew_hz_per_s"],
        "K": [list(map(float, row)) for row in gains["K"]],
        # simulate_hover.VelocityEstimator's default cutoff, and stereo's suspect scale
        # (the only per-frame quality signal in millimetres, hence the only one
        # commensurate with R -- see control/theory.md 27).
        "vel_cutoff_hz": 5.0,
        "suspect_mm": _suspect_mm(),
    }


def runtime_config() -> dict:
    """The binary's runtime knobs. Not control law, so not part of `ControlConfig`."""

    ts = json.loads(GAINS.read_text())["design"]["ts"]
    return {
        #: MUST match `src/constants.h`'s SERIAL_BAUD. There is no handshake and no ack on
        #: the ASCII half, so a mismatch is silent: the firmware never parses a command
        #: and the coils hold their last value.
        "baud": C.SERIAL_BAUD if hasattr(C, "SERIAL_BAUD") else 921600,
        #: WALL-clock seconds without a fix before the coils are cut. Wall, not frame
        #: time: the hazard includes the pose producer stalling, and a stalled producer
        #: freezes frame time -- so a frame-time deadline never expires in exactly the
        #: case it exists for.
        "lost_land_s": 1.0,
        #: THREAD_TIME_CONSTRAINT_POLICY. `computation` must come from a measurement of
        #: the tick body: too low and the kernel demotes the thread for overrunning, too
        #: high and it is a reservation the scheduler may refuse. Seeded at an eighth of
        #: the period and re-measured by `pmw_fly`'s own dt report.
        "rt_computation_s": ts / 8.0,
        "rt_constraint_s": ts / 4.0,
        #: The altitude setpoint the loop trims about, in METRES (the loop is SI inside;
        #: only the pose and the CSV are millimetres). `RunConfig.hover_z_mm` is the
        #: Python twin and defaults to None, meaning "use the viser slider".
        "hover_z_m": 0.060,
    }


def write(path, gains: dict | None = None) -> dict:
    """Emit the flat `key value` file `pmw_fly` reads. Returns what was written.

    Flat text rather than JSON because the binary's whole claim is that it carries no
    dependency the firmware side does not, and a JSON parser is a library or 300 lines to
    hold a dict of scalars. The C++ loader throws on a key it needs and cannot find.
    """

    cfg = control_config(gains)
    cfg.update(runtime_config())
    lines = []
    for k, v in cfg.items():
        if k == "K":
            lines.append("K " + " ".join(f"{x:.17g}" for row in v for x in row))
        else:
            lines.append(f"{k} {v!r}" if isinstance(v, str) else f"{k} {float(v):.17g}")
    Path(path).write_text(
        "# written by controller/control/native_config.write -- do not hand-edit.\n"
        "# Every value's home is the Python module named in native_config.control_config.\n"
        + "\n".join(lines) + "\n")
    return cfg


def _suspect_mm() -> float:
    """`StereoPoseEstimator._suspect_mm`, which is `MAX_DISCREPANCY_MM`.

    The scale at which a frame's two views disagree enough to be worth doubting. Reused by
    the native filter as the denominator of the measurement-noise inflation, because
    `discrepancy_mm` is the only per-frame quality signal in millimetres and so the only
    one commensurate with R without a fitted conversion nobody has measured.
    """

    from controller.pose import stereo
    return float(stereo.MAX_DISCREPANCY_MM)


def demo():
    cfg = control_config()

    # 1. Exactly the required keys: nothing missing, nothing extra. A key in the C++
    #    struct and not here is a startup throw; a key here and not in the struct is
    #    dead weight nobody will delete.
    assert set(cfg) == set(REQUIRED), (set(cfg) ^ set(REQUIRED))

    # 2. Every value came from its owning module, not from a literal typed here. Spot
    #    check the ones most likely to be duplicated by a future edit.
    assert cfg["gate_sigma"] == F.GATE_SIGMA and cfg["max_gated"] == F.MAX_GATED
    assert cfg["max_coast_s"] == PR.MAX_COAST_S and cfg["tau_vel_s"] == PR.TAU_VEL_S
    g = json.loads(GAINS.read_text())
    assert cfg["K"] == [list(map(float, r)) for r in g["K"]]
    assert cfg["ts"] == g["design"]["ts"]

    # 3. Shapes and signs the C++ asserts on, checked here where the message is readable.
    assert len(cfg["K"]) == 2 and all(len(r) == 6 for r in cfg["K"]), cfg["K"]
    assert cfg["ts"] > 0 and cfg["f_hover"] > 0 and cfg["freq_min"] < cfg["freq_max"]
    assert cfg["max_gated"] >= 1, "a gate with no escape can lock itself on"

    # 4. THE LOADER MUST THROW ON A MISSING KEY. This is the property that makes one home
    #    per number actually hold; without it a typo becomes a silent default.
    try:
        import pmw_pose
    except ImportError:
        print("native_config: pmw_pose not built -- skipped the loader check")
    else:
        pmw_pose.StatePredictor(cfg)          # a full dict is accepted
        for k in REQUIRED:
            short = {kk: vv for kk, vv in cfg.items() if kk != k}
            try:
                pmw_pose.StatePredictor(short)
            except Exception as e:
                assert k in str(e), f"dropping {k!r} threw, but did not name it: {e}"
            else:
                raise AssertionError(f"dropping {k!r} was accepted -- it silently defaulted")
        print(f"native_config: loader throws on each of {len(REQUIRED)} missing keys")

    # 5. The flat file round-trips, and carries every key the binary asks for. Comparing
    #    against `--print-keys` is what stops the two lists drifting apart in silence.
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile("w+", suffix=".cfg", delete=False) as fh:
        path = fh.name
    written = write(path)
    text = Path(path).read_text()
    assert all(f"\n{k} " in "\n" + text for k in REQUIRED + RUNTIME), text[:400]
    binary = Path(__file__).resolve().parents[2] / "build" / "fly" / "pmw_fly"
    if binary.exists():
        want = subprocess.run([str(binary), "--print-keys"], capture_output=True,
                              text=True).stdout.split()
        missing = [k for k in want if k not in written]
        assert not missing, f"pmw_fly needs keys native_config does not emit: {missing}"
        print(f"native_config: flat file covers all {len(want)} keys pmw_fly asks for")
    else:
        print("native_config: pmw_fly not built -- skipped the flat-file key check")

    print(f"native_config: {len(cfg)} keys, all sourced from their owning module\n  ok")


if __name__ == "__main__":
    demo()
