# Four-Coil RLC Driver — Characterization & Current-Sense Calibration

**Date:** 2026-06-18
**Setup:** 4 air/RLC coils on channels **A, B, C, D**, driven by **VNH5019** motor-driver
half-bridges. Channels **A & C** on one board, **B & D** on another. Design resonant
frequency **f₀ ≈ 190 Hz**. Each channel has a CS (MultiSense) current-sense output with a
shunt resistor (~2.53 kΩ) to ground.

This document tracks the full investigation: frequency-response sweeps → grounding study →
190 Hz matching → carrier-ramp drive test → VNH5019 current-sense calibration → calibrated
current plots.

---

## 1. Frequency-sweep response (initial data)

Each channel measured as a dense spectrum (dBu vs frequency). Raw traces are noisy
(noise floor ~−55 to −65 dBu); analysis uses a rolling-median smoothed envelope.

**Key observation (original `20260617.csv`, parent folder):**
- **A & C** (board 1): higher gain, but a **double-peak with a notch at ~330–350 Hz**
  (notch depth A ≈ 4 dB, C ≈ 2 dB) — and the two channels did **not** match each other.
- **B & D** (board 2): clean single-resonance humps, well matched, slightly lower gain.

A simple series-RLC current response is a single bandpass peak — it **cannot** produce a
notch. A notch + second peak is the signature of **two coupled resonators**. Since *both*
board-1 channels showed it and board-2 showed none, the defect was **systemic to board 1**,
not a single dead component.

---

## 2. Grounding investigation (root cause of the notch)

**Topology that produced the notch:** two power supplies (one per board); all ground
bananas landed on **one** supply (star hub), the two supply grounds linked by a single thin
**male–male jumper**, and the ESP32 USB also powered from the hub supply. The board on the
*other* supply returned its entire current through that one jumper.

**Mechanism:** the jumper's series **R + L** is a **shared return impedance** for both
channels on that board → **common-impedance coupling** → coupled-resonator mode splitting →
the notch. The board tied directly to the hub had a low-impedance return and stayed clean.

### Grounding configurations compared

Files: `left_ground.csv`, `right_ground_17.csv`, `seperate_ground.csv`
(all in **kHz** units; `plot_grounds.py` normalizes to Hz).
Graph: **`ground_compare.png`** (4-grid, one panel per channel, three configs overlaid).

| Config | Peak freq (A/B/C/D) | Peak level | Channel match | Notch? |
|---|---|---|---|---|
| all GND **right** | 246 / 239 / 269 / 212 Hz | −44 to −47 dBu (highest) | ±2.5 dB | none |
| all GND **left**  | 112 / 77 / 133 / 111 Hz | −51 to −52 dBu | ±1.0 dB | none |
| **separate PSU**  | 112 / 106 / 161 / 107 Hz | −49 to −50 dBu | **±0.9 dB** | none |

**Findings:**
- Consolidating the ground **removed the notch** on A & C — confirming the
  common-impedance-coupling hypothesis (not a coil fault).
- Grounding shifts the **resonant frequency by ~130 Hz** and gain by ~5–6 dB → the ground
  return impedance is effectively in series with the RLC and detunes it.
- **Separate grounds** gives the **tightest channel-to-channel match (±0.9 dB)**.

**Decision: use separate grounds** (each board on its own supply, grounds still linked).

---

## 3. 190 Hz matching (separate grounds)

Files: `seperate_ground.csv`, `seperate_ground-40sec.csv`
Script: `plot_separate.py` (auto-detects Hz/kHz units; dashed line at f₀ = 190 Hz)
Graphs: `separate_all.png`, `separate_grid.png`,
`seperate_ground-40sec_all.png`, `seperate_ground-40sec_grid.png`

Response value **at 190 Hz** (40-sec slow sweep, cleanest):

| Channel | @ 190 Hz |
|---|---|
| A | −52.0 dBu |
| C | −52.5 dBu |
| D | −53.0 dBu |
| B | −53.1 dBu |

Spread only **1.1 dB** at f₀ — well matched. (190 Hz sits on the downslope of the broad
~110 Hz envelope hump, but all four channels track together.)

---

## 4. Carrier-ramp drive test (time-domain)

Carrier PWM (15 kHz) duty ramped **linearly 0 → 100%** while driving the 190 Hz signal.
Files: `seperate_ground_10sec_linear_carrier_ramp_15kHz_ABCD.csv`,
`…_A.csv`. Script: `plot_ramp.py`.
**Raw graphs (CS voltage vs time):**
`ramp_ABCD_all.png`, `ramp_ABCD_grid.png`, `ramp_A_only.png`.

The measured `Channel X (V)` is the **CS sense voltage**; envelope (rolling-max) tracks the
duty ramp. Output rises linearly then plateaus at 100% duty.

**Peak CS voltage at 100% duty (envelope):** C 0.65 V · A 0.49 V · D 0.37 V · B 0.36 V.
Driven spread (~5 dB, C ≈ 1.8× B/D) is **larger** than the small-signal sweep suggested —
the driven test is the relevant one for equalization.
B & D raw traces show a small **negative offset** (~−0.04 V) worth investigating.

---

## 5. VNH5019 current-sense calibration (K gain)

The MultiSense (CS) pin sources `I_SENSE = I_load / K`; the shunt converts it to voltage:

```
V_CS = I_SENSE · R_shunt = (I_load / K) · R_shunt
  ⇒   I_load = K · V_CS / R_shunt          (calibration)
  ⇒   K      = I_load · R_shunt / V_CS
```

**Datasheet (Table 10):** K₀ = **7110 typ** (range **4670–10110**), plus temp/part drift.
That predicts sensitivity `K/R ≈ 7110/2540 ≈ 2.8 A/V`, i.e. **V_CS ≈ 1.8 V at 5 A**.

### Single-point measurement (each channel isolated, 190 Hz, 100% duty)

Two input typos corrected: **B shunt 2530 kΩ → 2.530 kΩ**; **D supply 524.7 A → 5.247 A**.

| Ch | I_supply | R_shunt | V_CS | I_SENSE | Sensitivity (A/V) | Back-calc K |
|----|---------|---------|------|---------|------|------|
| A | 5.466 A | 2.532 kΩ | 380.0 mV | 150 µA | **14.38** | 36,400 ⚠ outlier |
| B | 3.618 A | 2.530 kΩ | 404.9 mV | 160 µA | 8.94 | 22,600 (low-conf) |
| C | 5.234 A | 2.543 kΩ | 529.0 mV | 208 µA | 9.89 | 25,160 ✅ |
| D | 5.247 A | 2.540 kΩ | 531.5 mV | 209 µA | 9.87 | 25,080 ✅ |

**C and D agree to 0.3%** (K_eff ≈ 25,100). A is a hard outlier (most supply current, lowest
CS). A single global K would give **±25% current error**, so per-channel cal is required.

### Why back-calc K (~25,000) ≫ datasheet (≤10,110)

The CS only mirrors the **high-side MOSFET current of the conducting leg**; during
freewheeling / the opposite half-cycle it reads ~0. The DC supply ammeter integrates the
**full** average, so `V_CS / I_supply` is artificially low and apparent K is inflated ~3.5×.
**This is a measurement-methodology artifact**, not a blown sense ratio — so the
K_eff ≈ 25,100 is valid **only for this exact 190 Hz / 100%-duty waveform**.

### Two-parameter (gain + offset) fit — `calibrate.py`

The carrier ramp sweeps current 0→max, so it yields the **zero-current offset** (CS voltage
with drive off, `t < 1 s`) and a **linearity check** (CS envelope vs duty). Combined with the
single-point 100% anchor:

```
offset = CS at zero current (from ramp baseline)
gain g = I_100 / (V_100 − offset)        [A/V]
I_load(V_CS) = g · (V_CS − offset)
```

| Ch | **Offset** | Gain (A/V, offset-corr) | K_eff | Linearity R² | Gain (no-offset) |
|----|-----------|------|-------|------|------|
| A | **0.0 mV** | 14.38 | 36,400 ⚠ | 0.979 | 14.38 |
| B | **−40.4 mV** | 8.12 | 20,550 | 0.969 | 8.94 |
| C | **0.0 mV** | 9.89 | 25,160 | 0.989 | 9.89 |
| D | **−40.8 mV** | 9.17 | 23,290 | 0.968 | 9.87 |

Graph: **`calibration_fit.png`** (4-grid, CS vs current with fit line + offset).

**Findings:**
- **B and D share an identical ~−40 mV CS offset** — they're on the **same board**, so this is
  a **systematic board-2 offset** (not random). It must be subtracted before scaling.
- **A & C have zero offset** (board 1).
- **CS is linear** over the full range (R² ≈ 0.97–0.99) → a gain+offset model is valid.
- **A's outlier survives offset correction** (offset = 0, gain still 14.38). So A's inflated
  K is a **genuine gain anomaly, not an offset artifact** — flag A's sense path for a hardware
  check (it draws the most supply current yet the least CS signal).

**Caveat on the offset-corrected K:** the offset comes from the *ramp* capture while V_100
comes from the *single-point* capture; subtracting one from the other assumes both share the
same CS/ADC offset. Notably C & D agreed to 0.3% **without** offset correction, so this
transfer is not certain. The definitive fix is a **single synchronized capture** logging CS
and a series **current-probe** reading together across the ramp (removes the avg-vs-peak and
cross-capture ambiguities, and pulls K back toward the 4670–10110 datasheet window).

### Target metric: PEAK coil current (for magnetic field)

Magnetic field amplitude `B ∝ N·I_peak`, so the wanted quantity is **peak** coil current
(also what sets saturation) — not average or RMS. Calibration model:

```
I_peak(V_CS) = g_peak · (V_CS,peak − offset)
```

- Extract the **peak** of the CS envelope (15 kHz carrier removed → 190 Hz crest). The
  `env()` rolling-max in the scripts already does this.
- Anchor `g_peak` to a **series current-probe PEAK** reading, **not** DC supply current
  (supply current is an average → wrong scale for peak; it's the main source of the 3.5× K
  inflation).

Script: **`peak_calibration.py`** (ready; needs probe data).
- **Mode A (two-point):** fill `PROBE_PEAK_100` with peak coil current at 100% duty per
  channel; gain = `I_peak / (V_CS,peak − offset)`. Ramp already supplies offset and the peak
  CS plateau (A 0.476, B 0.365, C 0.651, D 0.332 V).
- **Mode B (multipoint):** capture one synchronized CSV `Time, CS(V), Probe(A)` while ramping
  duty; the script low-passes CS, takes peak envelopes, and fits `g_peak + offset` with R².

### Next steps
1. **Capture peak current with a series probe** (Mode A one point, or Mode B full sweep) →
   true `g_peak`, datasheet-comparable K, correct absolute peak Amps.
2. **Re-measure A** — its gain anomaly is real; **C & D remain the trusted reference pair.**
3. Subtract the **board-2 (B/D) −40 mV offset** in firmware regardless.
4. Equalize channels on **peak current at 190 Hz** (duty trim) once `g_peak` is anchored.

---

## 6. Calibrated current plots (compensated gain → Amps)

**Two versions produced:**

*(a) Single-point gain (no offset)* — `plot_current.py`, `I = SENS · V_CS`:
`current_ABCD_all.png`, `current_ABCD_grid.png`, `current_A_only.png`.

*(b) Two-parameter, offset-corrected* — `calibrate.py`, `I = g·(V_CS − offset)`:
`current_ABCD_all_offsetcorr.png` (recommended for B/D, which carry the −40 mV offset).

```
I_A = 14.38·(V_CS − 0)        I_B = 8.12·(V_CS + 0.0404)
I_C =  9.89·(V_CS − 0)        I_D = 9.17·(V_CS + 0.0408)        (A, V in volts)
```

**Peak calibrated current at 100% duty (offset-corrected):**

| Channel | Peak current | Confidence |
|---|---|---|
| A | 7.03 A | low (gain anomaly) |
| C | 6.44 A | trusted |
| D | 3.75 A | trusted (offset-corr) |
| B | 3.29 A | trusted (offset-corr) |

> These currents inherit the methodology caveat in §5 — they are consistent **relative**
> values for the 190 Hz/100%-duty waveform; absolute accuracy awaits a probe-based / DC
> multi-point recalibration.

---

## 7. File index

**Data (CSV):** `left_ground`, `right_ground_17`, `seperate_ground`,
`seperate_ground-40sec`, `seperate_ground_10sec_linear_carrier_ramp_15kHz_{A,ABCD}`.

**Scripts:**
- `plot_sweep.py` — single-file sweep (raw + smoothed, unit-aware).
- `plot_grounds.py` — 4-grid grounding-config comparison.
- `plot_separate.py` — separate-ground sweep, 190 Hz marker + value (CSV as arg).
- `plot_ramp.py` — raw carrier-ramp CS voltage vs time.
- `plot_current.py` — calibrated current (single-point gain, CS → A) ramp plots.
- `calibrate.py` — **two-parameter (gain + offset) calibration**, linearity check,
  offset-corrected current plot.
- `plot_boards.py` — per-board A&C / B&D overlay.

**Graphs:** raw sweeps & ramps (`*_sweep.png`, `ramp_*.png`, `ground_compare.png`,
`separate_*`, `seperate_ground-40sec_*`); calibrated current (`current_*`);
calibration fit (`calibration_fit.png`); offset-corrected current
(`current_ABCD_all_offsetcorr.png`).

---

## 8. Summary

1. The A/C **notch** was a **grounding artifact** (common-impedance coupling through a thin
   inter-supply ground jumper), **not** a coil fault — confirmed by it vanishing under
   consolidated grounding.
2. **Separate grounds** chosen: best channel match (±0.9 dB small-signal, 1.1 dB at 190 Hz).
3. VNH5019 CS two-parameter calibration: **B & D carry a systematic board-2 −40 mV offset**
   (subtract it); CS is **linear** (R²≈0.97–0.99). **C & D trustworthy**; **A is a genuine
   gain anomaly** (survives offset correction) → hardware check.
4. Apparent K is ~3.5× datasheet due to AC-waveform sensing (CS mirrors one conducting leg) —
   a **synchronized CS + current-probe multi-point capture** is the remaining step for
   absolute, datasheet-comparable K.
5. Calibrated current plots produced (single-point and offset-corrected); absolute values
   pending the synchronized recalibration.
