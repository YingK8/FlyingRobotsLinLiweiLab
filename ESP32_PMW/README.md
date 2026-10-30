# ESP32 PWM Coil Controller

Two firmware entry points share the restored `lib/` and `spiffs_data/`:

| env | source | behavior |
|---|---|---|
| `esp32dev` | [`src/main.cpp`](src/main.cpp) | thin JSON-driven 10 Hz horizontal rotating field; payload `horizontal_10hz.json` |
| `pull_impulse` | [`src/main_pull_impulse.cpp`](src/main_pull_impulse.cpp) | JSON-driven pulsed-gradient conveyor: release, kick, coast, weak guide |

## Horizontal rotating field (`esp32dev`)

The behavior lives in [`spiffs_data/horizontal_10hz.json`](spiffs_data/horizontal_10hz.json); the main is only the boot/load/loop wrapper. Pure phase control; every coil runs at the same **50 % carrier duty**:

| coil | A | B | C | D |
|---|---|---|---|---|
| phase | 0° | 90° | 180° | 90° |

Each PH pin is a 10 Hz square wave at 50 % duty (zero DC). With equal
amplitudes the centre field is

```
Bx = b (B - A) = -b (sin wt + cos wt)
By = b (D - C) =  b (cos wt - sin wt)
Bz = bz (A + B + C + D) = -2 bz sin wt
```

`Bx` and `By` are in quadrature with equal amplitude, so the horizontal field
rotates at 10 Hz. **The vertical component vanishes only at the vertical-null
height** (`bz = 0`, about 24.9 mm on this geometry). At 10 mm, `Bz` is actually
larger than the horizontal components; move the working plane to `h0` for a
purely horizontal field. At `h0` every phase set gives `Bz = 0`; this one puts
`Bx` and `By` in quadrature, so the horizontal field is circular. Off `h0`, no
equal-amplitude phase set can both cancel `Bz` and stay circular — unequal
carrier duties would be required.

If B and C are physically dead, A = 0° and D = 90° are still 90° apart in
space and in phase, so they alone produce a horizontal rotating field at `h0`.
The requested pattern therefore degrades gracefully with the B/C drivers off.

B and C are commanded at the same 50 % duty as A and D. If they remain off on
the bench, check the EN/PH wiring or that driver channel; the firmware is
driving all four.

## Pulsed-gradient conveyor (`pull_impulse`)

The field magnitude of a driven coil is largest **at that coil**, so any steady
unipolar or biased-pair drive has its attractor at a coil. A biased pair does not
create a moving trap between coils: at a 50/50 bias the global maximum jumps from
one coil to the other and the centre is a saddle. The release mechanism is
therefore to **turn the source coil fully off**, then pulse the next coil.

`spiffs_data/pull_impulse.json` runs three laps of the clockwise order
**A → D → B → C → A**. Each transition is:

| phase | field command | duration | purpose |
|---|---|---|---|
| `KICK` | target coil 0 → **30 %** → 0 | 50 / 250 / 50 ms | impulse away from the source |
| `COAST` | all coils 0 | 400 ms | let momentum carry the robot |
| `GUIDE` | target coil 0 → **10 %** → 0 | 100 / 700 / 100 ms | weak pull, avoids slamming into the coil |
| `SETTLE` | all off | 200 ms | stop before the next release |

All coils are driven at the **50 Hz series resonance** with the symmetric **50 %
commutation duty**, so the current has no DC component and the series capacitor
passes it. Only carrier amplitudes change; phases stay forward.

Tuning, in order:

1. Robot does not leave the source coil → raise `KICK` (30 → 40–60 %) or the
   250 ms hold.
2. Robot overshoots the target → lower `GUIDE` (10 → 5 %) or shorten the 700 ms
   hold.
3. Robot stops halfway → raise `GUIDE` or lengthen the hold.
4. Change `order` / the labels in `pull_impulse.json` to use a different coil
   sequence.

## Build and run

```bash
cd ESP32_PMW

# upload all JSON payloads once:
pio run -e pull_impulse -t uploadfs

# the impulse conveyor:
pio run -e pull_impulse -t upload
pio device monitor -e pull_impulse

# the single-file 10 Hz rotating field:
pio run -e esp32dev -t upload
```

The `pull_impulse` firmware prints the usual telemetry line and the label of
each sequence step (`KICK_C1_A_TO_D`, `COAST_...`, `GUIDE_D`, `SETTLE_D`, ...).

## Restored payloads

`spiffs_data/` contains the gradient set (`gradient_polarity`,
`lateral_gradient_pull`, `gradient_crossfade`, `gradient_bipolar`,
`gradient_staircase`), the horizontal set (`gradient_horizontal`,
`gradient_envelope`, `gradient_mode`), the rotating set (`rotating_field`,
`rotating_axis`), plus `swim.json`, `carrier_ramp.json`, and the new
`pull_impulse.json`.

## Safety

- 50 Hz resonance current is set by coil resistance; the series cap is rated
  ~2 A RMS. The 30 % kick is a starting point, not a calibrated current.
- No overcurrent protection is wired (`ADC_PINS = GPIO_NUM_NC` on this rig).
- No reset button (`RESET_BUTTON_PIN = GPIO_NUM_NC`); power-cycle to stop.
