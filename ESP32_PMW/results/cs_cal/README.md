# cs_cal: ESP32 current-sense calibration, DC into a resistive load

Made by `controller/control/cs_cal.py record` (two-probe mode: scope on OUTA/OUTB, CS from
ESP32 telemetry). Every run folder holds `points.csv` (one row per pulse), `serial.jsonl`
(raw telemetry) and `blocks.npz` (raw scope blocks). Raw: never overwrite.

Load 2.031 ohm, R_s A=2532 B=2530 C=2542 D=2540 ohm (DMM). Telemetry used SENS
{15.26, 15.28, 15.57, 15.34} throughout.

## Layout

| where | what |
|---|---|
| `2026-11-03_020615_A`, `2026-11-04_085506_B`, `2026-11-03_021436_C`, `2026-11-03_023837_D` | **the calibration**: full 6-10 V sweeps, idle channels read exactly 0.00 A |
| `bd_crosstalk/` | full or partial sweeps taken after the 2026-11-03 splitter short: driving D lifts B's reading and vice versa (B idle reads 0.8-10.7 A while D drives). Not a calibration of anything |
| `aborted/` | 0-3 driven points: no-drive runs (V_load < 0.1 V), and the first B/D-crosstalk runs. `090711_D` is the 9.7 A reading that tripped servo's old 10 A limit |
| `cs_cal.png` | `analyse` on the four calibration runs (rebuild below) |
| `cs_D_runs.png`, `cs_esp_vs_true.png` | older figures, 2026-11-03; they include `bd_crosstalk/` D runs. Superseded |

The 2026-10-30 four-probe runs were removed from the tree; recover from git, commit 04151e0.
Their channel mapping is wrong (driving A reads on `esp_B`), so they are not evidence for SENS.

```bash
uv run python controller/control/cs_cal.py analyse results/cs_cal/2026-* \
    --rs A=2532,B=2530,C=2542,D=2540 --rload 2.031
```

## Result (2026-11-04)

The firmware's current reads **1.74-1.88x high** on every channel over 2.4-4.4 A.

Affine fit, I_true = a * I_esp + b, both legs (P0, P180) pooled:

| ch | a | b [A] | SENS' = a*SENS | max resid | gain-only SENS (through 0) | its max resid |
|---|---|---|---|---|---|---|
| A | 0.450 | +0.75 | 6.87 | 21 mA | 8.76 | 296 mA |
| B | 0.443 / 0.450 | +0.82 / +0.60 | 6.77 / 6.87 | 34 / 19 mA | 8.76 (P0) | 309 mA |
| C | 0.429 | +0.74 | 6.68 | 36 mA | 8.50 | 307 mA |
| D | 0.425 | +0.75 | 6.53 | 73 mA | 8.25 | 307 mA |

B is per leg (P0 / P180): its P180 10 V point reads 0.5 A high (r 2.08 against 1.87-1.89
either side), and the B row excludes it. With it, B P180 is a=0.404, resid 163 mA.

- **The intercept is real at ~0.75 A, the same on all four.** A gain-only SENS cannot fit
  it: ~0.3 A worst-case error inside 2.4-4.4 A, and worse outside. An affine fit is good
  to 20-70 mA.
- **The intercept is an extrapolation.** No point is below 2.4 A (6 V is the lowest
  setpoint the VNH5019 UVLO allows into 2 ohm). Whether CS is linear down to 0 or has a
  knee (VNH5019 CS offset, or the ESP32 ADC's dead zone near 0 mV at 11 dB) is NOT
  measured. A bigger load resistor reaches lower current at the same bus voltage.
- **Legs agree** to 1-2% on A, C, D. B's P180 sense mirror sits ~20 mV above P0.
- **K = R_s / slope ~ 17 000** (A, B P0, C, D); datasheet 4670-10110. This is through the
  ESP32 ADC, so a gain error in the ADC path and a high VNH5019 K cannot be told apart
  from these runs. The scope CS probe would separate them.
- **No faults, no trips** in the four runs (`trip=0` on every line).

**Applied 2026-11-04** in `src/drive_common.h`: SENS {6.87, 6.81, 6.68, 6.53} and
CS_DEAD_MV {109, 105, 110, 114} (pooled fit, B without its outlier). `current_sense.cpp`
adds CS_DEAD_MV to every raw sample more than 5 mV above the boot zero, before the 50 ms
filter. Not yet checked on the bench: a re-run of `record` should now read a ~ 1, b ~ 0.
Telemetry amps logged before this date are on the old scale (1.74-1.88x high).
