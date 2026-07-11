# Two-Channel Magnetic Coil PWM Driver — PCB Design Documentation

> Schematic: `pcb_schematics.pdf` (`PWM_amp.kicad_sch`, KiCad EDA 9.0.2)
> Netlist (authoritative for connectivity): `hw_references/PWM_amp.txt` (KiCad netlist, exported 2026-06-07)
> Authors: Wei Yue, Divij Muthu, Tofic Esses, Kevin Ying, Nikita Lukhanin — UC Berkeley, Liwei Lin Lab
> Sheet: Rev 1, dated 2026-03-04, sheet 1/1, size A4
>
> **All pin numbers and net connections below were verified against the netlist.**

---

## 1. Overview

This board is the **power stage ("amplifier")** between the ESP32 logic controller (the
firmware in this repo) and the magnetic coils of the aerial-robotics platform. It converts
low-level logic signals into high-current bipolar coil drive.

### Design requirements (from the on-sheet note)

| Parameter | Specification |
|---|---|
| Continuous current per coil | **2 A** |
| Nominal coil voltage | **12 V** |
| Cluster architecture | 4-coil cluster |
| Aggregate continuous current per driver unit | **8 A** |

### Scope

The sheet implements **two independent, identical channels**. Each is a full H-bridge coil
driver built on an ST **VNH5019A-E**. Each channel drives **one coil** differentially. To
service a full **4-coil cluster (8 A aggregate)**, **two of these boards** are used.

| | Channel 1 | Channel 2 |
|---|---|---|
| H-bridge | **U2** | **U1** |
| Inverter | **U4** | **U3** |
| Protection FET | **Q1** | **Q2** |
| Signal header | **J1** | **J2** |
| Coil output | **J10** (XT30) | **J5** (XT30) |
| Series-cap mount | **J4** | **J3** |
| Bulk / bypass | C5(100µF), C7(0.1µF), C4(1µF) | C3(100µF), C8(0.1µF), C6(1µF) |
| +3V3 decoupling | C1(0.1µF) | C2(0.1µF) |

---

## 2. Signal flow (verified)

```
ESP32 (per channel)
   │  phase pin ─────────────► J1.1 = INB ─┬──► U2.10  (VNH5019 INB,  direction leg B)
   │                                       └──► U4.2   (inverter IN)
   │                                            U4.4 ──► U2.4  (VNH5019 INA, direction leg A)
   │  carrier pin ───────────► J1.3 = PWM ─────► U2.6   (VNH5019 PWM, amplitude/enable chop)
   │  enable ────────────────► J1.2 ──┬─R6(1k)─► U2.5  (EN/DIAGA)
   │                                  └─R5(1k)─► U2.9  (EN/DIAGB)      (each pulled to +3V3 via 4.7k)
   │  current sense ◄────────── J1.4 ──R4─────── U2.8  (CS, with R7 1k to GND)
   │  +3V3 ──────────────────► J1.5
   │  GND  ──────────────────► J1.6
```

This is the **locked-antiphase** drive: `INA = NOT INB` (set by the inverter) controls
**direction**, while the **carrier on the `PWM` pin** chops the bridge for **amplitude/current**.
Net result: coil voltage centred around 0 and swinging between +VM and −VM.

### Mapping to the firmware

| Firmware (`main_*.cpp`) | Header pin | Board net | Effect |
|---|---|---|---|
| `*_PWM_PIN` (phase square wave) | J1/J2 **pin 1** | `INB` → also into inverter → `INA` | Direction / phase (**inverter is on this line**) |
| `*_CARRIER_PIN` (carrier) | J1/J2 **pin 3** | VNH5019 `PWM` (pin 6), **direct** | Amplitude / current (**no inverter**) |

This is the hardware basis for the firmware note *"the carrier goes directly into the H-bridge;
the inverter is on the PWM pin."*

---

## 3. Signal header pinout (J1 / J2, Conn_01x06) — RESOLVED

| Pin | Net | Connects to | Direction (from ESP32) |
|---|---|---|---|
| **1** | `INB1`/`INB2` | VNH5019 `INB` (pin 10) + inverter input | **Out** — phase/direction square wave |
| **2** | `NET-(J-PIN_2)` | EN/DIAGA via 1k + EN/DIAGB via 1k | **Out** (drive low = disable); also fault readback |
| **3** | `NET-(J-PIN_3)` | VNH5019 `PWM` (pin 6) | **Out** — carrier (amplitude) |
| **4** | `NET-(J-PIN_4)` | VNH5019 `CS` (pin 8) via series R | **In** — analog current-sense (to ADC) |
| **5** | `+3V_1` | VNH5019 `VCC` (pin 7) + inverter VCC | **Power in** — 3.3 V logic supply |
| **6** | `GND1` | logic ground | **Ground** |

> Cross-check with `README.md` channel map: the firmware's per-channel "PWM pin" = header pin 1
> (phase/`INB`) and "Carrier pin" = header pin 3 (`PWM`). The ESP32 has 4 channels; each board
> handles 2, so a full cluster uses two boards (channel→coil assignment is set by the harness).

---

## 4. Power input and conditioning

| Ref | Part | Net(s) | Function |
|---|---|---|---|
| **J49** | XT60PW-M | `J49.1`=`VM1`, `J49.2`/`SH1`/`SH2`=`PGND1` | 12 V power input; shield to power ground. |
| **D1** | SMCJ18A | `VM1` → `PGND1` | TVS, 18 V standoff — clamps supply transients (inductive load / load-dump). |
| **C4 / C6** | 1 µF | `VM1` → `PGND1` | Input-side bulk/decoupling (battery side, before the FET). |

### Reverse-battery protection FET (verified topology)

| Ref | Part | Pin | Net | Role |
|---|---|---|---|---|
| **Q1 / Q2** | **FDD8447L** (N-ch, TO-252) | `.3` source | `VM1` | high-side series device |
| | | `.2` drain | `VBAT` rail (`NET-(Q1-D)`) | feeds the bridge supply |
| | | `.1` gate | VNH5019 `CP` pin 11 (`NET-(Q1-G)`) | **gate driven by the IC's internal charge pump** |

The VNH5019's charge pump (`CP`, pin 11) drives the external N-FET gate above `VM1`, turning the
FET fully on in normal polarity; on reverse battery the FET stays off → **reverse-polarity
protection** (ST-recommended topology). VNH5019 pin 12 connects to `VM1` directly as the
battery-side supply/sense reference.

### Bulk capacitance (on the post-FET `VBAT` rail)

| Ref | Value | Net | Function |
|---|---|---|---|
| **C5 / C3** | **100 µF** (EEH-ZC electrolytic) | `VBAT` (`NET-(Q-D)`) | Bulk reservoir for H-bridge switching transients. |
| **C7 / C8** | 0.1 µF | `VBAT` | HF bypass at the IC supply pins. |

> **Grounds:** `PGND1` (power return: bridge `GND`, bulk caps, D1, J49) and `GND1` (logic: inverter
> GND, +3V3 caps, CS pulldowns, header pin 6) are **separate nets in the netlist** — see §8, item 2.

---

## 5. H-bridge driver: VNH5019A-E (verified pinout)

`U2` (Ch1) / `U1` (Ch2), MultiPowerSO-30. Pin numbers below are from the netlist.

| Pin(s) | Net | Function |
|---|---|---|
| **4** | `INA` | Direction leg A — driven by the inverter output. |
| **10** | `INB` | Direction leg B — driven by header pin 1. |
| **6** | `PWM` | Carrier/chop input — driven by header pin 3. |
| **5** | `EN/DIAGA` | Open-drain enable+fault, leg A. 4.7k pull-up to +3V3, 1k series to header pin 2. |
| **9** | `EN/DIAGB` | Open-drain enable+fault, leg B. 4.7k pull-up to +3V3, 1k series to header pin 2. |
| **8** | `CS` | Analog current-sense source. 1k to `GND1` (sense load), series R to header pin 4. |
| **7** | `VCC` | 3.3 V logic supply (`+3V_1`). |
| **12** | `VBAT`/sense | Battery-side supply (`VM1`, before protection FET). |
| **3, 13, 23, 31** | bridge supply | Post-FET `VBAT` rail (bulk-decoupled). |
| **11** | `CP` | Charge pump → drives external protection-FET gate. |
| **1, 25, 30, 33** | `OUTA` | Bridge output A → coil + (`J10.P`/`J5.P`). |
| **15, 16, 21, 32** | `OUTB` | Bridge output B → series cap → coil return. |
| **18, 19, 20, 26, 27, 28** | `GNDA`/`GNDB` | Power ground (`PGND1`). |
| **2, 14, 17, 22, 24, 29** | NC | Unconnected. |

### Control / bias resistor network (verified)

| Ch1 | Ch2 | Value | Net path | Role |
|---|---|---|---|---|
| R6 | R8 | 1 kΩ | header pin2 → EN/DIAG (A/B) | Series isolation on enable line |
| R5 | R1 | 1 kΩ | header pin2 → EN/DIAG (B/A) | Series isolation on enable line |
| R2 | R11 | 4.7 kΩ | +3V3 → EN/DIAGA | Pull-up (open-drain) |
| R3 | R10 | 4.7 kΩ | +3V3 → EN/DIAGB | Pull-up (open-drain) |
| R7 | R9 | 1 kΩ | CS → GND1 | Current-sense load (I→V) |
| **R4** | R12 | **R12 = 4.7 kΩ; R4 value not in netlist BOM** | CS → header pin4 | Series to ADC — **see §8 item 3** |

Default state: with header pin 2 floating/high, EN/DIAG are pulled to 3.3 V → **enabled**; the
ESP32 drives pin 2 low to disable, and either open-drain leg can still pull its own node low for
fault readback.

---

## 6. Outputs and the series coil capacitor (verified)

The coil connects across the XT30 (`J10` Ch1 / `J5` Ch2). The "capacitor array" header
(`J4` Ch1 / `J3` Ch2) is wired **in series in the bridge→coil loop**, not as decoupling:

```
OUTA (U2.1,25,30,33) ──► J10.P ──► [ COIL ] ──► J10.N ─┐
                                                       │
                                            J4.2 ──────┘
                                            J4.1 ◄── OUTB (U2.15,16,21,32)
              external capacitor array mounts across J4.1–J4.2 (in series)
```

So the current path is **OUTA → coil → (series cap on J4) → OUTB**. This series capacitor:

- **Blocks DC**, AC-coupling the coil → keeps net magnetization centred (no DC core bias),
  complementing the locked-antiphase bipolar drive.
- Forms a **series-resonant LC** with the coil inductance — relevant because the firmware sweeps
  drive frequency; the cap value sets the resonant point.
- Must be rated for the full coil current (≈2 A RMS) and the drive-voltage swing.

Channel 2 is identical: `OUTA`→`J5.P`→coil→`J5.N`=`J3.2`, series cap on `J3`, `J3.1`=`OUTB`.

---

## 7. Locked-antiphase inverter (U3 / U4) and the "broken signal" problem

| Ref | Part | Pin → Net | Notes |
|---|---|---|---|
| **U4 / U3** | **NC7SVU04P5X** (SC-70-5) | `.2`=IN (`INB`), `.4`=OUT (`INA`), `.5`=VCC (`+3V_1`), `.3`=GND (`GND1`), `.1`=NC | Single-gate inverter making `INA = NOT INB`. |

### ⚠ CONFIRMED root cause of the inverter "breaking / broken signals": split ground

**`GND1` (logic) and `PGND1` (power) are not tied on the board.** They meet only through an
indirect, high-impedance path (ESP32 GND → ESP32 board → bench supply → `J49`). Measured DC
offset between them: **~400 mV** with the bridge active (AC ground-bounce during switching is
worse). The netlist confirms they are separate nets with no net-tie (§8 item 2).

Why this breaks/destroys the inverter:

- The inverter output `INA` swings 0–3.3 V **relative to `GND1`**; the VNH5019 reads `INA`
  against **`PGND1`**. A 400 mV+ reference offset shifts/corrupts the logic level the bridge
  sees → "broken/noisy signals."
- **Timeline match:** before the carrier-polarity fix the bridge was off (carrier parked LOW),
  no coil current flowed in `PGND1`, the grounds were nearly equal, and the inverter looked
  "clean." Once the bridge actually drove current (after the fix), `PGND1` bounced against
  `GND1` and the inverter failed.
- **Repeated death:** ground bounce pushes the inverter pins below its local GND / above local
  VCC, forward-biasing its ESD/clamp diodes with the bridge's high available current behind
  them → latch-up / diode burnout, killing each replacement part.
- **Why a sister board looked fine:** same design, but its bridge wasn't sourcing coil current,
  so the offset never appeared.

**Fix (rework):** bond `GND1` to `PGND1` with **one solid, low-impedance jumper at a single
point** (star ground) near the power entry / VNH5019 ground — e.g. `GND1` at J1.6 / inverter GND
to `PGND1` at the VNH5019 `GND` pins / bulk-cap negative / `J49.2`. Fat and short (~0 Ω). Do NOT
tie at multiple points and do NOT route coil return current through the logic ground. Re-measure
`GND1`↔`PGND1` ≈ 0 Ω with the bridge driving. **Fix in copper next spin: add a `GND1`↔`PGND1`
net-tie at the bridge ground.**

**Note on the inverter part:** the unbuffered `NC7SVU04` (the `U` in `SVU`) is *not* the root
cause — it runs clean on a board with a proper ground. The open-drain `SN74LVC1G06` is the wrong
replacement (open-drain → needs a pull-up, weak/noisy node). Re-fit the original `NC7SVU04`, or a
buffered drop-in (`NC7SZ04` / `74LVC1G04` / `SN74LVC1G14`). The `1G14` (Schmitt, push-pull) adds
noise margin but is not required once the ground is fixed.

---

## 8. Anomalies & items to verify (from netlist cross-check)

1. **`CS_DIS` is not connected.** The schematic labels a `CS_DIS` pin ("pull high to disable,
   low to enable"), but **no `CS_DIS` net exists in the netlist** — it appears to be left
   unconnected/floating. Confirm the VNH5019A-E `CS_DIS` behaves acceptably floating, or tie it
   to a defined level. (Several VNH5019 pads are NC: U#.2/14/17/22/24/29.)
2. **`GND1` and `PGND1` are separate nets with no tie — CONFIRMED ROOT CAUSE of the inverter
   failures (see §7).** Measured ~400 mV offset between them with the bridge active; the only
   path is indirect (through the ESP32/supply). Fix in rework with a single-point star jumper;
   fix in copper next spin with a `GND1`↔`PGND1` net-tie at the bridge ground.
3. **`R4` has no value in the netlist BOM**, while its Channel-2 counterpart **`R12` = 4.7 kΩ**.
   `R4` is the Channel-1 `CS`→header series resistor. Confirm `R4`'s intended value (likely
   4.7 kΩ to match R12) and that it is actually populated.
4. **Inverter part choice** (§7) — confirm whether the unbuffered `NC7SVU04` was intentional.
5. **Charge-pump capacitor on `CP` (pin 11)** — the net `NET-(Q-G)` ties `CP` directly to the FET
   gate; verify whether the VNH5019 also needs a dedicated charge-pump capacitor on `CP` per its
   datasheet, and that the FET gate is adequately driven/protected (gate-source clamp).

---

## 9. Full bill of materials (verified against netlist)

### Shared / power

| Ref | Value / Part | Footprint | Function |
|---|---|---|---|
| J49 | XT60PW-M | XT60PW-M | 12 V input |
| D1 | SMCJ18A | SMC (DO-214AB) | TVS, 18 V standoff |
| H1,H2,H3,H8 | MountingHole | — | mechanical (not in netlist) |

### Per channel (Ch1 ref / Ch2 ref)

| Ch1 | Ch2 | Value / Part | Footprint | Function |
|---|---|---|---|---|
| U2 | U1 | VNH5019A-E | MultiPowerSO-30 | Full H-bridge driver |
| U4 | U3 | NC7SVU04P5X (**unbuffered** — see §7) | SC-70-5 | Locked-antiphase inverter |
| Q1 | Q2 | **FDD8447L** (N-ch) | TO-252-2 | Reverse-polarity protection FET |
| C5 | C3 | 100 µF (EEH-ZC) | radial | Bulk cap (VBAT) |
| C7 | C8 | 0.1 µF | 0805 | HF bypass (VBAT) |
| C4 | C6 | 1 µF | 0805 | Input decoupling (VM1) |
| C1 | C2 | 0.1 µF | 0805 | +3V3 decoupling |
| R2 | R11 | 4.7 kΩ | 0805 | EN/DIAGA pull-up |
| R3 | R10 | 4.7 kΩ | 0805 | EN/DIAGB pull-up |
| R6 | R8 | 1 kΩ | 0805 | EN series (A) |
| R5 | R1 | 1 kΩ | 0805 | EN series (B) |
| R7 | R9 | 1 kΩ | 0805 | CS load to GND |
| **R4** | R12 | **R12=4.7 kΩ; R4 value TBD** | 0805 | CS series to header (see §8) |
| J1 | J2 | Conn_01x06 | 1x06 2.54 mm | Signal header |
| J10 | J5 | XT30UPB-M | XT30UPB-M | Coil output |
| J4 | J3 | Conn_01x02 | 1x02 2.54 mm | **Series coil-capacitor mount** (see §6) |

---

## 10. Operational notes & cautions

- **Enable default = on:** header pin 2 floating → EN/DIAG pulled high → channel enabled. Drive
  low to disable. The same line reads back open-drain fault flags.
- **Current sense:** read header pin 4; voltage developed across the 1 kΩ (R7/R9) at the IC.
- **Series coil cap (J3/J4):** must be installed for the bridge to deliver current to the coil —
  it is in the main current path, not optional decoupling. Size it for the sweep frequency band
  and ≥2 A RMS.
- **Inductive load:** D1 (TVS) + bulk caps protect the rail; the VNH5019 body diodes handle
  freewheel. Keep C5/C3 close to the IC supply pins.
- **Two boards per cluster:** each board = 2 coils (2 A each) → two boards for the 8 A / 4-coil spec.
- **Before bring-up:** resolve §8 items 1–3 (CS_DIS, GND1↔PGND1 tie, R4 value) and strongly
  consider the §7 inverter swap to a buffered part.
