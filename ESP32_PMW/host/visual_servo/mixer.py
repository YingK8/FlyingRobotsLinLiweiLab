"""tilt command -> per-coil carrier duties (v2 lateral actuation).

The lateral controller outputs a 2-D tilt command in CAMERA axes. Two stages turn
it into the four carrier duties streamed to the firmware as A<ch>,<duty>:

  1. AXIS/SIGN calibration (LATERAL_CAL, a 2x2): rotate/flip the camera-frame tilt
     into the ROBOT tilt axes. A wrong sign here makes the loop DIVERGE, so run
     `calibrate.py lateral` and paste the matrix into config before closing the
     loop. Identity is only correct if the camera happens to be axis-aligned.

  2. GEOMETRIC mixer: each coil sits at a corner of the 2x2 array; to tilt the
     thrust toward a direction we WEAKEN the carriers on that side (duty < nominal).
     Weakening-only (duty clamped to <= nominal) matches the "mostly weakening"
     actuation and can never command more than full-on.

        duty[i] = clamp(nominal - gain * dot(coil_dir[i], tilt_robot), 0, nominal)

Coil order matches the firmware A<ch> index: 0=A, 1=B, 2=C, 3=D.
"""
from config import CARRIER_NOMINAL, MIXER_GAIN, LATERAL_CAL

# Canonical 2x2 layout (see canonical-gpio-pin-map): A=(0,0) B=(1,1) C=(0,1) D=(1,0).
# Unit direction from the array centre (0.5, 0.5) to each coil, in ROBOT axes.
# These are a geometric PRIOR; LATERAL_CAL corrects the camera<->robot rotation/sign.
_R2 = 0.7071067811865476  # 1/sqrt(2)
COIL_DIRS = (
    (-_R2, -_R2),  # A
    ( _R2,  _R2),  # B
    (-_R2,  _R2),  # C
    ( _R2, -_R2),  # D
)


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def apply_cal(tx_cam, ty_cam, cal=LATERAL_CAL):
    """Map a camera-frame tilt -> robot-frame tilt via the 2x2 calibration matrix."""
    (a, b), (c, d) = cal
    return a * tx_cam + b * ty_cam, c * tx_cam + d * ty_cam


def tilt_to_duties(tx_cam, ty_cam, nominal=CARRIER_NOMINAL, gain=MIXER_GAIN,
                   cal=LATERAL_CAL):
    """Return [dutyA, dutyB, dutyC, dutyD] (%) for a camera-frame tilt command."""
    tx, ty = apply_cal(tx_cam, ty_cam, cal)
    return [
        _clamp(nominal - gain * (dx * tx + dy * ty), 0.0, nominal)
        for dx, dy in COIL_DIRS
    ]


if __name__ == "__main__":
    # Quick sanity print: pure +x and +y tilts under the default (identity) cal.
    for label, (tx, ty) in [("+x", (1.0, 0.0)), ("+y", (0.0, 1.0)),
                            ("level", (0.0, 0.0))]:
        print(f"tilt {label:>5}: duties =",
              [f"{d:5.1f}" for d in tilt_to_duties(tx, ty)])
