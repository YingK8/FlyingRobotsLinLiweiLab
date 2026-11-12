# Current-sense calibration and drive characterisation of a four-coil VNH5019 array

## Abstract

We characterise a four-coil electromagnetic drive built from two dual VNH5019
half-bridge boards and calibrate the on-chip current-sense (MultiSense) output of each
channel against load current. Each VNH5019 reports load current as a mirror current
proportional to the true output current; a shunt resistor to ground converts this to a
voltage that the ESP32 ADC and an external oscilloscope both read. We determine the
per-channel sense gain from a supply-driven current sweep over the 6.0 to 7.2 A
operating band and recover the device sense ratio K. Because the sweep is confined to a
narrow, high-current interval, a two-parameter (gain and offset) fit is ill-conditioned
and returns a physically impossible negative zero-current offset; we therefore pin the
offset at the independently measured drive-off value and estimate the gain from a fit
through the origin. The resulting effective sense ratios agree across the four channels
to within 1.1 % (coefficient of variation), consistent with the part-to-part tolerance
of the VNH5019. We close with a power-balance model separating driver-IC dissipation
from coil dissipation and assess when the IC term may be treated as a constant residual.

## 1. Introduction

The array drives four air-core coils, labelled A, B, C, and D, to produce a rotating
magnetic field near a design resonance of 190 Hz. Two dual VNH5019 boards supply the
coils: board 1 drives channels A and C, board 2 drives channels B and D. Each board
energises one opposed coaxial coil pair, so axis 1 corresponds to A and C and axis 2 to
B and D. Quantitative control of the field requires knowledge of the coil current, which
we obtain from each driver's current-sense output rather than from an inline meter. This
document records the drive protocol, the calibration procedure that converts each
sense voltage to amperes, and the reasoning behind the fit methodology. An earlier
investigation of the frequency response, grounding topology, and 190 Hz matching is
retained in `PROGRESS.md`.

## 2. Materials and methods

### 2.1 Coil array and drive electronics

Firmware on an ESP32 (`../ESP32_PMW/src/main_test_current.cpp`) generates one commutation
channel per coil. Two VNH5019 boards act as the power stage. The measured shunt
resistances on the sense outputs are 2538, 2540, 2550, and 2545 Ω for channels A, B, C,
and D respectively (four-wire ohmmeter).

### 2.2 Drive waveform

Each coil is commutated at a fixed drive frequency of 190 Hz with initial electrical
phases of 0, 180, 90, and 270 degrees for A, B, C, and D, giving a counter-clockwise
rotation sequence A, C, B, D. Amplitude is set by a 15 kHz carrier pulse-width
modulation; only the carrier duty cycle is varied during an experiment, while the
190 Hz commutation is held constant.

### 2.3 Instrumentation

An oscilloscope (PicoScope) records the four sense-output voltages at 1 ms sample
spacing. Records are stored as comma-separated files whose second line carries the
column units and is skipped on load. The ESP32 shares a common ground with the supply
and lacks USB isolation, so the microcontroller cannot be addressed while the stage is
energised; each protocol therefore runs once from reset and de-energises the coils on
completion.

### 2.4 Duty-ramp protocol

The duty-ramp protocol energises one board at a time and holds the inactive board at
zero. Within a board, each coil is ramped alone and then the opposed pair is ramped
together. The sequence comprises six segments:

```
A | C | A+C | B | D | B+D  ->  graceful ramp-down
```

Each segment lasts 10 s: a 2 s off-gap followed by a linear increase of the carrier duty
from 0 to 100 % over 8 s. A 2 s controlled ramp-down at the end returns all coils to
zero, giving a total record length near 62 s. Comparing a coil's peak current when
ramped alone against its peak when ramped with its same-board partner isolates the
within-axis mutual coupling and the per-board supply sag. The calibrated result is in
`figures/dutyramp/` and the raw sense voltages, annotated with the active-channel
schedule, are in the companion `*_raw_*` figures.

One data-integrity note applies to the 20260628_7_03pm capture: the oscilloscope inputs
for channels B and D were interchanged relative to the firmware channel index. The
firmware energises B during the 30 to 40 s segment and D during the 40 to 50 s segment,
whereas the recorded columns show the reverse. The raw plots retain the columns as
captured; the swap does not affect the calibration, which treats each channel
independently.

### 2.5 Calibration sweep protocol

For each channel we stepped the bench supply to sweep the coil current across the
operating band and recorded, at each step, the load current in amperes and the
sense-output voltage in millivolts. The load current serves as the reference; the supply
voltage is not used in the fit, since the sense output depends on current alone. The
resulting file, `../ESP32_PMW/src/Tue30JunKCalibration.csv`, holds two columns per
channel and spans roughly 6.0 to 7.2 A.

## 3. Current-sense calibration

### 3.1 Sense model and definition of K

The VNH5019 sense output sources a current equal to the load current divided by the
device sense ratio K. This mirror current develops a voltage across the shunt resistor
R_sense:

```
V_CS = offset + I_load * R_sense / K
```

We express the calibration through a gain g in amperes per volt and an effective sense
ratio:

```
g      = K / R_sense              I_load = g * (V_CS - offset)
K_eff  = R_sense * g              (recovers the device ratio K)
```

Here g is the quantity consumed by the plotting code, and K_eff is the dimensionless
device ratio expected to match across channels of the same part. Because the sense
output measures a ratio of currents, it does not depend on the power dissipated in the
driver; Section 4 develops this point.

### 3.2 Fitting methodology

The calibration sweep covers only the 6.0 to 7.2 A operating band. Over so narrow an
interval a joint fit of slope and intercept is poorly conditioned: the two parameters
trade against each other, and the least-squares solution places the zero-current
intercept at a negative value between -16 and -32 mV across the four channels. A negative
intercept is unphysical, because the sense pin sources current into R_sense and cannot
drive V_CS below zero. We therefore separate the two parameters. The zero-current offset
is a distinct quantity of order 0 mV, obtained from the drive-off baseline, so we pin it
at zero and estimate the gain from a fit through the origin. The procedure is:

1. Fit V_CS against I by ordinary least squares.
2. Flag points whose residual exceeds three scaled median absolute deviations and refit
   on the inliers.
3. Estimate the gain from the through-origin slope of the inliers.

We report the free two-parameter fit alongside the pinned-offset gain for transparency
(`calibrate_k.py`, figure `figures/calibration/kcal_fit.png`).

### 3.3 Results

| Channel | g (A/V) | K_eff | Free-fit g (A/V) | Free offset (mV) | R² |
|---------|---------|-------|------------------|------------------|-----|
| A | 15.10 | 38 331 | 14.07 | -32 | 0.921 |
| B | 15.10 | 38 352 | 14.13 | -29 | 0.984 |
| C | 15.09 | 38 476 | 14.34 | -23 | 0.973 |
| D | 15.43 | 39 257 | 14.86 | -16 | 0.975 |

Across the four channels the effective sense ratio averages 38 604 with a standard
deviation of 440, a coefficient of variation of 1.1 %. The pinned-offset gains agree
with a prior independent calibration (`../ESP32_PMW/src/k-calibration.csv`) to within
about 1 %. Peak calibrated currents in the 20260628_7_03pm record are 9.9, 9.5, 8.6, and
8.6 A for A, B, C, and D.

Channel D reads about 2 % lower on the sense output than the other three: its
voltage-to-current ratio is 64.8 mV/A against 66.1 to 66.3 mV/A for A, B, and C. The
shift is systematic across the entire D data set rather than the effect of a single
point; one genuine outlier at 6.58 A was removed by the residual test. We attribute the
difference to part-to-part variation in the VNH5019 sense ratio, whose datasheet
tolerance exceeds 2 %. The difference is acceptable because each channel carries its own
gain, so D's value of 15.43 A/V introduces no error into D's current estimate; a shared
gain would be required for the spread to matter. The spread also reassigns between
sessions, since the prior calibration placed channel C highest, which places it at the
measurement and tolerance floor.

## 4. Power model

The supply power divides between the driver IC and the coil:

```
P_supply = V_supply * I_supply = P_IC + P_coil
P_coil   = I^2 * R_coil                          (ohmic; air coil, R_coil ~= 0.95 ohm)
P_IC     = I^2 * R_DS(on)   (conduction)
         + V_supply * I_q   (quiescent, ~ constant)
         + P_switching      (proportional to carrier frequency, ~ constant)
```

The VNH5019 on-resistance is near 37 mΩ typical and rises with junction temperature;
the quiescent draw is a few milliamperes. At the 6 to 7 A operating point the coil
dissipates roughly 35 to 47 W, the IC conduction loss is 1.3 to 1.8 W, and the quiescent
and switching terms together contribute 0.1 to 0.3 W.

### 4.1 Is the IC power a constant residual that may be ignored for a linearity model?

For the current-sense calibration the answer is yes, because the sense output measures
coil current as a ratio and is independent of IC dissipation. The relation between V_CS
and I_coil is therefore linear regardless of the power the IC burns, and the only
constant term is the sense-circuit zero-current offset of Section 3.2, not a power term.

For a model of supply power or supply current against coil current the answer is a
qualified no. The dominant IC term, the conduction loss I^2 R_DS(on), scales with the
square of the current and is not constant. Over the narrow 6 to 7 A band its variation is
about 1 % of the total power, so treating the whole IC contribution as a constant offset
is an acceptable first-order approximation locally. The approximation fails on
extrapolation: across a wide current range the conduction term grows quadratically, and
near zero current the constant quiescent term dominates while both I^2 R terms vanish. In
short, a constant residual is adequate as a small correction inside the operating band
but should not be assumed outside it or where better than 1 % accuracy is required.

## 5. Repository structure

```
pico/
  scripts/
    calibrate_k.py          canonical K-calibration (Tue30Jun sweep)
    plot_duty_ramp.py       calibrated current, per channel / board / all
    plot_duty_ramp_raw.py   raw sense voltage with the active-channel schedule
  data/dutyramp/            current duty-ramp capture (CSV)
  figures/calibration/      kcal_fit.png
  figures/dutyramp/         calibrated and raw duty-ramp figures
  PROGRESS.md               earlier investigation log
  old/                      superseded scripts, data, and figures
    scripts/  data/  figures/  debug/  tilt/
```

The calibration source files (`Tue30JunKCalibration.csv`, and the prior
`k-calibration.csv`) reside under `../ESP32_PMW/src/` with the firmware that collected
them. Regenerate every current figure from the `pico/` directory with:

```
python3 scripts/calibrate_k.py
python3 scripts/plot_duty_ramp.py     data/dutyramp/20260628_7_03pm.csv
python3 scripts/plot_duty_ramp_raw.py data/dutyramp/20260628_7_03pm.csv
```

## References

1. STMicroelectronics, VNH5019A-E automotive fully integrated H-bridge motor driver,
   datasheet, June 2017.
2. Pololu Corporation, VNH5019 motor driver carrier, product documentation.
