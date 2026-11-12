# VectorField

Gives the coil settings for a rotating field at a point. Input: a point `r`, an orientation
(`R` in SO(3) or a unit quaternion `q`), a field radius `b` and the drive frequency `f`. Output:
per-channel carrier duty, per-channel phase, and a saturation `scale`. The axis is `R z`. The
field at theta = 0 lies along `R x`. Identity = +z, which sends exactly `SPIN_PHASES`.

```cpp
static VectorField vectorField(SPIN_PHASES);
const double point[3] = {0, 0, 0.015}, noRotation[4] = {1, 0, 0, 0};
CoilSettings settings = vectorField.settingsFor(point, noRotation, 2e-3, ctl.getFrequency());
// settings.dutyPercent[4], settings.phaseDegrees[4], settings.fieldFraction
```

Nothing is refused with an error. `b` is signed: `-b` is the same field half a cycle later, so every
phase moves 180 deg. `b = 0` gives zero duty with `fieldFraction = 1`, which turns the coils off. An
unusable orientation (`q = 0`, `R` not a rotation), a non-finite `b`, `f <= 0` or a singular point
gives zero duty with `fieldFraction = 0`. The firmware reads that as "keep the previous drive". See
`notes/vectoring.tex` Algorithm 1.

`main_flight` exposes it as `vec=<x_mm>:<y_mm>:<z_mm>:<qw>:<qx>:<qy>:<qz>:<b_mT>` (FLIGHT only).
Coordinates are in the array frame: coil plane at z = 0, robot on the +z side.

The derivation, the calibration knobs (`channelGain`, `phaseOffsetDegrees`) and the rig check that still has to
run are in `controller/control/theory.md` 28. **Keep the twist about the axis continuous between
commands** (28.5): a twist jump is a phase step under a locked rotor.

The RLC and rail constants in `coilArray` mirror `controller/control/constants.py`; change both.

Self-check, built on the host because the library uses `<math.h>` only:

```bash
c++ -std=c++17 -Ilib/VectorField/src lib/VectorField/src/VectorField.cpp \
    lib/VectorField/check/self_check.cpp -o /tmp/vf && /tmp/vf
```
