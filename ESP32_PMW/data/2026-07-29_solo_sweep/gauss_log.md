# Solo frequency sweep — gaussmeter log (2026-07-29)

Firmware `env:solo_sweep` + `spiffs_data/solo_sweep.json`. One coil energized at
a time, 70 % carrier, 7 frequency points × 6 s, 20 s de-energized gap before each
channel for repositioning the probe.

## What this answers

1. **Where is each channel's own series resonance?** `f0 = 1/(2π√(LC))` sits at
   the peak of that channel's `I_cs(f)`. Four different `f0` values explain the
   3× strength spread at a fixed 200 Hz (C = 1.55 A, D = 4.75 A) — i.e. you are
   driving each channel at a different distance from its own resonance. No LCR
   meter needed.
2. **Does B track I_cs?** CS senses only high-side conducting current (unipolar,
   ~0 during freewheel), so it is blind to circulating/reactive current. B is
   proportional to *total* coil current. If a channel's B peaks at a frequency
   where its `I_cs` does not, that offset is the hidden current — the direct
   answer to "where did the residual go".

## Run procedure

```bash
~/.platformio/penv/bin/pio run -e solo_sweep -t uploadfs   # once, uploads solo_sweep.json
~/.platformio/penv/bin/pio run -e solo_sweep -t upload
~/.platformio/penv/bin/pio device monitor -e solo_sweep | tee data/2026-07-29_solo_sweep/serial.log
```

Total run ≈ 4.1 min, starts on boot (no arming).

**Probe placement dominates the measurement.** B falls off fast with distance, so
use a fixed spacer, mark the spot, keep the probe axis normal to the coil face,
and put it in the *same* place over each coil. Inconsistent placement makes the
four channels incomparable and wastes the run.

### Filming the meter

The meter is analog-readout only, so film the display and transcribe afterward.

- **Get the ESP32 indicator LED (GPIO2) in the same frame as the meter display.**
  It toggles on every schedule step, so each toggle in the video is a step
  boundary — that is how the video aligns to the serial log without a clapper.
- The 20 s gaps read as B ≈ 0 on the meter: unmistakable channel separators.
- Each 6 s hold is a flat plateau. 7 plateaus per channel, in ascending
  frequency order (100 → 220 Hz).
- Read the plateau *middle*, not the transition edge.

## Transcription table

Fill `B` from the video. `I_cs` comes free from `serial.log` (the `freq=` field
plus the live channel's current), so leave it blank if you'd rather I parse it.

| f (Hz) | A: B | A: I_cs | B: B | B: I_cs | C: B | C: I_cs | D: B | D: I_cs |
|---|---|---|---|---|---|---|---|---|
| 100 | | | | | | | | |
| 120 | | | | | | | | |
| 140 | | | | | | | | |
| 155 | | | | | | | | |
| 170 | | | | | | | | |
| 190 | | | | | | | | |
| 220 | | | | | | | | |

Units: state the gaussmeter's units and range setting here → `______`

## Before you trust a reading

**The drive is AC at 100–220 Hz.** A DC-only Hall gaussmeter will average a
symmetric AC field to ≈ 0 and show jitter around zero, not the field magnitude.
Check the meter has an **AC/RMS mode or peak-hold** before filming a full run.

Quick pre-check: start the run, park the probe over coil A, and watch the
`A_155HZ` step. A clear, steady non-zero deflection → the meter reads AC, carry
on. Jitter around zero → the meter cannot see this field, and only the `I_cs(f)`
half of the experiment is valid (still worth the run: it alone resolves the
per-channel resonance question).

## Notes / observations

<!-- rail voltage, PSU current, anything that tripped, probe jig description -->
