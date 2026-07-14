# PicoScope 5443D capture & analysis tool

`picoscope_capture.py` captures 1–4 analog channels from a **PicoScope 5443D MSO**
(5000D series, `ps5000a` driver) and classifies each waveform as **sine / square /
triangle**, measures the fundamental (~190 Hz) and THD, and — when you give it a
voltage and a current channel — measures the **V–I phase** that tells you whether a
coil branch is at resonance.

Built to debug the ESP32 → VNH5019 H-bridge → series-RLC coil chain.

---

## ⚠ GROUNDING SAFETY — READ FIRST (this hazard already destroyed an inverter)

- The bench supplies **float**; the PicoScope BNC grounds are **all common** and
  **earth-referenced**. Clipping an earthed scope ground onto a live floating node
  has already **blown an inverter** on this hardware.
- **Rules:**
  - Use a **differential probe** for any bridge-output / coil / 12 V node. Never clip
    scope ground to a 12 V rail or an `OUTA`/`OUTB` node.
  - Never connect **two channel grounds to two different floating nodes** — they are
    common, so you will short through the scope.
  - Measure **current** with a current probe (isolated) **or** the voltage across a
    small series **sense resistor read with the differential probe** (floating).
  - ESP32 GPIO (3.3 V logic) is safe on an analog channel **only if** the ESP32 GND is
    your established earth reference.

---

## Why the bridge output looks "square" (it's supposed to)

The H-bridge **output** (`OUTA`/`OUTB`) is **always a ~190 Hz bipolar square wave**.
At 100% carrier there is no 15 kHz ripple; below 100% the 15 kHz carrier rides on top.
The **sinusoid is the COIL CURRENT** (and the voltage across L or across C alone),
**not** the bridge output voltage. Seeing a square at the bridge output is **expected,
not a fault** — to see the sine, measure the **current**.

---

## Install

The 5443D needs the official **`picosdk`** wrapper **and** the **native PicoSDK
runtime** (installed separately). Do **not** install `pypicosdk` (6000E/3000E only).

```bash
# From the repo root. Reuse the analysis venv if present, else create it.
test -d writeup/.venv || python3 -m venv writeup/.venv
source writeup/.venv/bin/activate
pip install -r ESP32_PMW/tools/requirements.txt

# Then install the native PicoSDK runtime from https://www.picotech.com/downloads
#   (the picosdk wrapper dlopen()s libps5000a at import time).
```

`--self-test` and `--dry-run` run without the scope or the native runtime, so you can
validate the analysis anywhere.

---

## Quick start

Default — capture just the bridge output (the node you're probing). Replace the
`SCALE` with your differential-probe attenuation (e.g. `20` for ÷20):

```bash
python ESP32_PMW/tools/picoscope/picoscope_capture.py --preset envelope \
    --ch A:5v:DC:20:V:bridge_out
```

**The real test** — add the sense-resistor current channel (`SCALE = 1/Rsense`, e.g.
`10` for a 0.1 Ω sense) and ask for the V–I phase:

```bash
python ESP32_PMW/tools/picoscope/picoscope_capture.py --preset envelope \
    --ch A:5v:DC:20:V:bridge_out \
    --ch B:500mv:DC:10:A:coil_current \
    --vi bridge_out,coil_current
```

Inspect the 15 kHz carrier instead of the 190 Hz envelope:

```bash
python ESP32_PMW/tools/picoscope/picoscope_capture.py --preset carrier --ch A:5v:DC:20:V:bridge_out
```

`--ch` format: `CHANNEL:RANGE:COUPLING:SCALE:UNIT:LABEL` (only `CHANNEL:RANGE`
required). `RANGE` accepts shorthand (`50mv`,`100mv`,…,`2v`,`5v`,`10v`) or the full
`PS5000A_*` key. Presets: `envelope` (~200 kS/s, 100 ms), `carrier` (~5 MS/s, 5 ms),
`custom` (give `--sample-rate`/`--samples`).

---

## Outputs

- `<prefix>_results.csv` — `# key, value` metadata comments then `time_s,<labels>`
  columns in physical units (repo CSV convention).
- `<prefix>.png` — one time-domain trace per channel (with verdict + f0 in the title)
  plus a shared FFT subplot.
- Text summary — per-channel verdict / f0 / THD / RMS, the V–I phase + retune
  direction, and the decision tree below.

---

## Firmware-vs-Hardware decision tree

```
You are probing the BRIDGE OUTPUT (OUTA/OUTB).

1. Bridge output = clean ~190 Hz SQUARE (verdict SQUARE, f0~190, symmetric, no DC)
   -> EXPECTED & HEALTHY. Not the fault. The sinusoid is the CURRENT. Add the
      sense-resistor current channel (--ch B ... coil_current --vi bridge_out,coil_current).
      The CURRENT should read SINE at resonance.

2. Bridge output is NOT a clean 190 Hz square (wrong freq / DC / asymmetric / distorted)
   -> FIRMWARE/DRIVE issue: carrier duty, phase/direction signal, or channel mapping.
      Capture the ESP32 carrier + phase GPIOs (3.3 V logic) and confirm carrier=15 kHz,
      phase toggles at the drive frequency. Firmware map: A=carrier33/pwm32, B=26/25,
      C=14/27, D=13/23.

3. Current channel added -- what does the CURRENT look like?
   * SINE, in phase with bridge output (|phase|<10 deg), ~190 Hz -> healthy tank. Done.
   * TRIANGLE  -> series cap J3/J4 MISSING or SHORTED (current = integral of square).
   * SINE but leading/lagging -> OFF-RESONANCE; retune per the printed phase direction.
```

---

## Verification runbook

1. `--self-test` — synthetic sine/square/triangle classify correctly and a 30° V–I
   pair is recovered (no scope needed).
2. `--list-units` — confirms the unit opens and USB power is handled.
3. **Known signal** — drive a 190 Hz sine then square from the 5443D AWG (or a function
   generator) into channel A; expect verdicts SINE then SQUARE (THD ≈ 48%). Validates
   the full acquisition+analysis chain against ground truth.
4. **DUT (differential probe)** — capture the bridge output (expect clean 190 Hz
   SQUARE → firmware drive is fine), then add the sense-resistor current channel
   (expect SINE) and walk the decision tree with the printed verdicts and V–I phase.

---

## Running experiments (`run_experiment.py`)

`run_experiment.py` is the single build+flash+run driver for all three firmware
variants, selected by `--fw`:

```bash
# JSON-driven sweeps/experiments (main_experiment.cpp) -- any task_sequences/*.json,
# including tilt.json, runs here with no per-experiment script needed:
uv run python tools/run_experiment.py --fw experiment --json task_sequences/tilt.json

# Same, but with closed-loop PI compensation layered on top of the JSON
# schedule's carrier-duty commands (see "PI compensation" below):
uv run python tools/run_experiment.py --fw experiment --json task_sequences/tilt.json \
    --pi-compensate --pi-profile task_sequences/pi_profile_tilt.json

# RatioCurrentController profile (main_pi_profile.cpp):
uv run python tools/run_experiment.py --fw pi_profile \
    --profile task_sequences/pi_profile_tilt.json

# Onboard current-balance PI controller (main_current_pid.cpp), gains live-tunable:
uv run python tools/run_experiment.py --fw current_pid --dir cw
```

### PI compensation on `--fw experiment` (opt-in)

`main_experiment.cpp` is normally fully open-loop -- the JSON schedule's commanded
carrier duty goes straight to the hardware. `--pi-compensate --pi-profile <file>`
(the same ratio/gains schema as `--fw pi_profile`'s `--profile`, e.g.
`task_sequences/pi_profile_tilt.json`) engages `RatioCurrentController` on top of
it instead: each tick, channel A's JSON-commanded carrier duty becomes the shared
reference, every channel's target is `ratio[channel] * reference` (same convention
as the profile's `ratios` field, where A=1.0 is the reference channel), and the PI
loop's output *replaces* the schedule's raw commanded duty as the actual actuator
value. The schedule still drives everything else (frequency ramp, timing, labels,
direction) -- only carrier duty is closed-loop when this is on. Toggle it live from
the TUI's `:` command line with `pi=on` / `pi=off` at any point in the run.

All three firmwares share the same manual-start contract (`lib/ExperimentPhase`):
ARMING (3s, self-calibrates ADC zero) -> WAITING (coils latched off, waiting for an
explicit start command) -> the run -> STOPPED/DONE. Nothing auto-starts off a timer,
and `s` (e-stop) works from every phase.

By default, `run_experiment.py` opens a curses TUI (`experiment_tui.py`) once serial
is open: a scrolling telemetry stream on the left, an ASCII current-matrix on the
right (same channel/quadrant layout and current->mV conversion as
`coil_current_matrix.py`, just drawn with terminal colors instead of a matplotlib
window), and a status bar with REC/RUN indicator dots.

- `r` -- run/stop toggle (sends the start command while WAITING; sends e-stop while running)
- `s` -- immediate e-stop, any state
- `w` -- toggle recording the telemetry stream to `recording_<n>.csv`
- `:` -- command-line mode, for typing `kp=2.5`/`dir=cw`/etc. (`current_pid`/`pi_profile` live tuning)
- `q` -- quit (sends e-stop first)

Pass `--auto-start` for scripted/unattended callers (used internally by
`run_coupling_sweep.py` and `pid_autotune.py`): it sends the start command itself
the instant WAITING is seen, skipping the TUI's human gate. `--no-tui` runs a plain
`print()`-based capture loop instead (requires `--auto-start`, since there's no
other way to send the start command without the TUI).

`coil_current_matrix.py`'s own CLI (`--port` / `--log-file`) is still available for
standalone/offline use, e.g. replaying an already-saved log:

```bash
uv run python tools/coil_current_matrix.py --log-file state_space_cw_2A.log
```

---

## Limitations

- **Analog channels only.** The MSO digital pods (D0–D15) need a different API
  (`ps5000aSetDigitalPort`); capture ESP32 logic on analog channels instead.
- The bridge output is **differential/floating** → use a differential probe
  (single-ended A−B math re-introduces the grounding hazard).
- The 15 kHz carrier **aliases** into the `envelope` capture; inspect the carrier only
  with `--preset carrier`. The THD harmonic search is band-limited to compensate.
- Sample depth scales with resolution × channel count; the tool honors the
  device-reported max and errors (rather than truncating) if you ask for too many
  samples.
