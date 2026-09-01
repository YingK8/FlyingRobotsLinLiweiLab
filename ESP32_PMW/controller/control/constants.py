"""
Every number the flight path shares. One file, so a value cannot drift between two copies.

A constant belongs here when more than one module reads it. A number used in exactly one
place stays where it is used. Each entry says where it came from -- a measurement, a
derivation in `theory.md`, or plainly that it is a guess.

Nothing here is imported by firmware. `src/main_flight.cpp` carries its own copies of the
ramp defaults, and they are overridden by the host on every real flight.
"""

from __future__ import annotations

# --------------------------------------------------------------------- frequencies
#
# These are four different questions, which is why they are four different numbers.
# `theory.md` 18.12 works through why they do not have to agree.

# The MATLAB model this plant was ported from linearised here. Read only by
# `hover_model.summary()`, which asserts the port still reproduces MATLAB's numbers.
# Not a rig measurement and not a target -- do not fly it.
F_HOVER_MATLAB_REF_HZ = 140.0

# Where the LQR is linearised. The gains in `hover_controller.json` are only valid near
# this point, and `runner._anchor()` re-trims them at runtime to the frequency the ramp
# actually reached.
F_HOVER_DESIGN_HZ = 160.0

# Where `ZTracker` starts identifying from, before it adapts. MEASURED 2026-08-29:
# liftoff at ~180-185 Hz, lift peaking near 190-210.
F_HOVER_TRACK_HZ = 190.0

# MEASURED 2026-08-29. Ramps to 240 and 230 Hz both showed lift peaking near 180-210 Hz
# and then collapsing in the 220-240 bin -- z fell to -2.5 mm and stayed there, which is
# a rotor that lost sync. Replaced a seed guess of 190, which was the coil's electrical
# match and not a step-out at all.
F_STEPOUT_HZ = 225.0

# MEASURED. Coil current peaks here and falls above it, so torque margin is worse past
# resonance -- but lift goes as f^2 and wins anyway until step-out. Optimise for lift.
F_RESONANCE_HZ = 174.0

# ---------------------------------------------------------------- the takeoff ramp
#
# HOVER ACHIEVED with these values 2026-08-29. They are the RunConfig defaults.

RAMP_START_HZ = 2.0    # well under the ~4.9 Hz capture crossing, theory.md 18.3
RAMP_TARGET_HZ = 210.0  # 160 was the old target and does not lift, theory.md 18.8
RAMP_S = 30.0          # seconds, start to target
RAMP_K = 2.0           # EASE sharpness. k=1 is linear and fails to capture.

# A ramp longer than this is refused, not clamped: a segment profile goes to the firmware
# verbatim, so the cap cannot be applied for you. Coils reach 80 C after four ramps.
MAX_RAMP_S = 55.0

# `seq=ramp:<from>:<to>:<ms>:<mode>:<k>` -- the mode field, mirroring
# `lib/PwmSequencer` TaskMode. The firmware owns the curve shapes; this is only the
# wire encoding of which one to use.
RAMP_POLYNOMIAL, RAMP_EASE, RAMP_EXPONENTIAL = 0, 1, 2

# ---------------------------------------------------------------------- the coils
#
# TWO measurements of the same hardware that do not agree, kept apart and named rather
# than averaged into a number neither of them supports. The disagreement is open.
#
#   channel   R=15 ohm, L=6.7 mH   the driver's view, measured per channel; the spread
#                                  (12.6-18 ohm, 6.64-6.85 mH) is real, the four
#                                  channels differ. Used for driver limits.
#   series    R=6.9 ohm, L=1.4 mH  the resonant channel with its series capacitor.
#                                  L confirmed by the operator 2026-08-29; C fitted the
#                                  same day from |I|(f) over a 0.2-60 Hz ramp.
#
# The series pair predicts resonance at 1/(2*pi*sqrt(LC)) = 213 Hz, but the rig measures
# 174 Hz, so C is the suspect: it was fitted in the one band where it is the only thing
# visible. `takeoff_report._fit_rlc` refits L and C from each flight's logged currents;
# nothing writes the refit back yet.

COIL_CHANNEL_R_OHM = 15.0
COIL_CHANNEL_L_H = 6.7e-3

COIL_SERIES_R_OHM = 6.9
COIL_SERIES_L_H = 1.4e-3
COIL_SERIES_C_F = 400e-6

V_RAIL = 12.0        # V, net VM1
V_RAIL_MAX = 20.0    # V, driver ceiling before overvoltage risk
I_CONTINUOUS = 2.0   # A per coil
I_TRIP = 10.0        # A, the firmware overcurrent latch

# ------------------------------------------------------------------ the flight log
#
# `runner` writes these columns and `takeoff_report` reads them. Both import this name,
# so the schema cannot drift between writer and reader.

# `tilt_deg` / `tilt_az_deg` are the rotor normal: tilt away from the datum axis and the
# azimuth that tilt points along (estimator.Pose.theta_deg / phi_deg). Added 2026-09-01 --
# the pipeline had measured them every frame since the start and thrown them away, so
# "did the rotation axis move during the ramp?" could not be answered from any run.
# theory.md 18.14 had to infer a 4.6 deg thrust tilt from accelerations instead.
CSV_COLUMNS = ("t,state,f_hz,x_mm,y_mm,z_mm,tilt_deg,tilt_az_deg,mag,az,armed,spin,lost,"
               "i_a,i_b,i_c,i_d").split(",")


def demo():
    assert F_HOVER_TRACK_HZ < F_STEPOUT_HZ, "the tracker must start below step-out"
    assert RAMP_START_HZ < RAMP_TARGET_HZ <= F_STEPOUT_HZ, "the ramp must stop short of step-out"
    assert RAMP_S <= MAX_RAMP_S, "the default ramp must not trip its own thermal cap"

    # The series pair is what predicts resonance; check the disagreement is still the one
    # documented above and has not quietly become something else.
    import math
    predicted = 1.0 / (2 * math.pi * math.sqrt(COIL_SERIES_L_H * COIL_SERIES_C_F))
    assert 205.0 < predicted < 220.0, predicted
    assert predicted > F_RESONANCE_HZ, "series LC still predicts high against the measured peak"

    assert len(CSV_COLUMNS) == 17 and CSV_COLUMNS[0] == "t", CSV_COLUMNS
    assert CSV_COLUMNS[6:8] == ["tilt_deg", "tilt_az_deg"], CSV_COLUMNS
    print(f"constants: ramp {RAMP_START_HZ}->{RAMP_TARGET_HZ} Hz in {RAMP_S}s, "
          f"step-out {F_STEPOUT_HZ} Hz, series LC predicts {predicted:.0f} Hz "
          f"vs {F_RESONANCE_HZ:.0f} measured\n  ok")


if __name__ == "__main__":
    demo()
