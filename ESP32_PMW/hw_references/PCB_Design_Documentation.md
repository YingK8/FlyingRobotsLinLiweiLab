# Two-Channel Magnetic Coil PWM Driver — PCB Design Documentation

> Source: `pcb_schematics.pdf` (`PWM_amp.kicad_sch`, KiCad EDA 9.0.2)
> Authors: Wei Yue, Divij Muthu, Tofic Esses, Kevin Ying, Nikita Lukhanin — UC Berkeley, Liwei Lin Lab
> Sheet revision: **Rev 1**, dated **2026-03-04**, sheet 1/1, size A4

---

## 1. Overview

This board is the **power stage** ("amplifier") that sits between the ESP32 logic controller
(the firmware in this repository) and the magnetic coils of the aerial-robotics platform. It
takes low-level logic signals from the microcontroller and drives high-current coil loads.

### Design requirements (from the on-sheet design note)

| Parameter | Specification |
|---|---|
| Continuous current per coil | **2 A** |
| Nominal coil voltage | **12 V** |
| Cluster architecture | 4-coil cluster |
| Aggregate continuous current per driver unit | **8 A** |

The coils generate a magnetic field to actuate (spin/tilt) the platform. The current and
field-strength targets above set the sizing of the power devices, bulk capacitance, and the
copper/connector current ratings.

### Board scope

The sheet implements **two independent, identical channels** (a "two-channel" driver). Each
channel is a complete full-bridge (H-bridge) coil driver built around an ST **VNH5019A-E**
motor-driver IC. Each channel drives one coil differentially across its `OUTA`/`OUTB` pair.
To service a full **four-coil cluster (8 A aggregate)**, two of these boards are used.

---

## 2. System block diagram

```
                 ┌──────────── ESP32 (this repo's firmware) ───────────┐
                 │  per channel: PWM/phase pin  +  carrier pin         │
                 └───────────────┬──────────────────────┬──────────────┘
                                 │ INB (phase/dir)      │ PWM (carrier/amplitude)
                                 ▼                     ▼
   12 V (VM1) ──► [TVS + filter + MOSFET + bulk caps] ──► VBAT
   XT60 (J49)                                           │
                                 ┌──── inverter ───┐    │
                         INB ──►│ NC7SVU04P5X     │──►│ INA  (locked-antiphase)
                                 └─────────────────┘    │
                                                        ▼
                                            ┌────────────────────┐
                               INA, INB ──►│                    │
                                 PWM ─────►│   VNH5019A-E       │── OUTA ─┐
                        EN/DIAGA,B ◄─────►│  full H-bridge     │         ├──► Coil
                             CS, CS_DIS ──►│                    │── OUTB ─┘  (XT30)
                                            └────────────────────┘
```

Two of the channel block (inverter + VNH5019 + power conditioning) are placed on the sheet:
**Channel 1** (upper) and **Channel 2** (lower).

---

## 3. Power input and conditioning

| Ref | Part | Function |
|---|---|---|
| **J49** | XT60PW-M | Main 12 V power input (`VM1`). High-current XT60 connector with `P`, `N`, `SHIELD` pins. |
| **D1** | SMCJ18A | Unidirectional **TVS / transient-suppression diode** across `VM1`→`PGND1`. 18 V standoff protects the rail from inductive/load-dump transients (important with inductive coil loads). |
| **PWR_FLAG** ×3 | — | KiCad ERC power-flag markers on `VM1`, `GND1`, and `+3V_1` (no physical part). |
| **+3V_1** | — | Logic/control supply net (3.3 V logic reference for the inverters and VNH5019 `VCC`). Sourced from the ESP32 side via the signal header. |

### Per-channel power conditioning (mirrored on both channels)

| Ref (Ch1 / Ch2) | Part | Value | Function |
|---|---|---|---|
| **Q1 / Q2** | FDB8447L | N-ch MOSFET | Series power device in the `VM1`→`VBAT` path (reverse-polarity / load-switch protection of the motor-supply rail). Gate `G` referenced to the supply rail. |
| **C4 / C6** | Ceramic | 1 µF | Local supply decoupling near the MOSFET/gate. |
| **C5 / C3** | Electrolytic | **100 µF** | **Bulk capacitor** on the `VBAT` rail — supplies switching transient current to the H-bridge and damps rail sag during high-di/dt coil switching. (Sheet note: "Bulk Capacitor (Idk why)".) |
| **C7 / C8** | Ceramic | 0.1 µF | High-frequency bypass on the `VBAT` rail (paired with the bulk cap). |
| **C1 / C2** | Ceramic | 0.1 µF | Decoupling on the `+3V_1` logic rail at the signal header. |

> **Grounds:** the sheet distinguishes `PGND1` (power ground, high-current return for the
> H-bridge and bulk caps) from `GND1` (signal/logic ground for the inverter and control
> resistors). Keeping them separate and joining at a single point is standard practice to keep
> high-current return noise out of the logic reference.

---

## 4. The H-bridge driver: VNH5019A-E

Each channel uses one **ST VNH5019A-E** monolithic full-bridge motor driver (`U2` for
Channel 1, `U1` for Channel 2). It integrates two high-side and two low-side MOSFETs plus
charge pump, logic, and current sensing.

### Pin/net usage

| VNH5019 pin | Net (Ch1 / Ch2) | Role |
|---|---|---|
| `INA` | INA1 / INA2 | Bridge leg A direction input (driven by the inverter output). |
| `INB` | INB1 / INB2 | Bridge leg B direction input (driven from the signal header). |
| `PWM` | PWM (carrier) | PWM gate input — chops both legs on/off for amplitude/current control. Driven by the ESP32 **carrier** signal. |
| `EN/DIAGA` | via R6,R2 / R8,R11 | **Open-drain** enable + fault-diagnostic, leg A. Needs pull-up; pulled low internally on fault. |
| `EN/DIAGB` | via R5 / R1 | **Open-drain** enable + fault-diagnostic, leg B. |
| `CS` | via R7 / R9 | Analog **current-sense** output (proportional to bridge current). Routed to the header for ADC monitoring. |
| `CS_DIS` | via R4 / R12 | Current-sense disable. **Sheet note: pulled high disables CS; drive pin low to enable it.** |
| `VBAT` | VM1→(Q1)→VBAT | Motor power supply (12 V rail). |
| `VCC` | +3V_1 | Logic supply (3.3 V). |
| `CP` | — | Charge-pump pin (internal high-side gate drive). |
| `OUTA` / `OUTB` | to XT30 | Bridge outputs — the coil connects differentially across these. |
| `GNDA` / `GNDB` | PGND1 | Power-ground returns. |

### Control / bias resistor network (per channel)

The control lines use a mix of **series current-limit** resistors (≈1 kΩ) and
**pull-up/bias** resistors (≈4.7 kΩ) between the header, the IC, and `+3V_1`/`GND1`:

| Channel 1 | Channel 2 | Value | Typical role |
|---|---|---|---|
| R6 | R8 | 1 kΩ | Series resistor on an EN/DIAG / control line |
| R5 | R1 | 1 kΩ | Series resistor on an EN/DIAG / control line |
| R7 | R9 | 1 kΩ | Series resistor on CS / control line |
| R3 | R10 | 4.7 kΩ | Pull-up / bias on the PWM (carrier) input |
| R2 | R11 | 4.7 kΩ | Pull-up / bias (EN/DIAG) |
| R4 | R12 | 4.7 kΩ | Pull-up / bias (CS_DIS) |

> The exact net-by-net assignment of each resistor is best confirmed against the KiCad
> netlist; the values and functional grouping above are read directly from the sheet. The
> 4.7 kΩ resistors provide the required pull-ups for the VNH5019 open-drain `EN/DIAG` pins and
> default-bias the `PWM`/`CS_DIS` lines into a safe state; the 1 kΩ resistors limit current and
> protect the logic pins.

---

## 5. Locked-antiphase PWM scheme (key design concept)

On-sheet note (verbatim):

> *"To centre the PWM output around 0 (and between +/-VM), the PWM pin is kept constant, while
> the direction is flipped using a differential pair. I think it's called 'Locked-Antiphase'."*

### How it is wired

| Ref (Ch1 / Ch2) | Part | Function |
|---|---|---|
| **U4 / U3** | NC7SVU04P5X | Single-gate CMOS **inverter** (TinyLogic, SOT-553). Powered from `+3V_1`/`GND1`. |

- The header provides **INB** (the phase/direction signal from the ESP32).
- The inverter generates **INA = NOT INB**, so `INA` and `INB` always form a complementary
  (antiphase) pair → this sets bridge **direction**.
- The VNH5019 **PWM** pin receives the ESP32 **carrier**, which gates the whole bridge on/off
  → this sets **amplitude / current magnitude**.

With INA/INB locked in antiphase and the carrier modulating amplitude, the effective coil
voltage is centred around 0 V and swings between +VM and −VM as direction flips — i.e. a
sign-controlled (bipolar) drive with magnitude set by carrier duty.

### Mapping to the firmware

This matches the firmware convention (`src/main_tilt.cpp` and the `PhaseController` library):

| Firmware concept | Board net | Effect |
|---|---|---|
| `*_PWM_PIN` (phase square wave) | header → `INB` → inverter → `INA` | Direction / phase (the inverter is on this line) |
| `*_CARRIER_PIN` (carrier) | header → VNH5019 `PWM` pin (direct) | Amplitude / current (no inverter — carrier goes straight to the bridge) |

This is the hardware basis for the firmware note *"the carrier frequency is directly into the
H-bridge; the inverter is on the PWM pin."*

---

## 6. Connectors and interfaces

| Ref | Part | Channel | Purpose |
|---|---|---|---|
| **J49** | XT60PW-M | shared | 12 V main power input (`VM1` / `PGND1`). |
| **J1** | Conn_01x06 (6-pin header) | Ch1 | Logic/signal interface from ESP32: carries `INB1` (phase), control lines (`EN/DIAG`, `CS`, `CS_DIS`), `CS` sense return, and `+3V_1`/`GND1`. |
| **J2** | Conn_01x06 (6-pin header) | Ch2 | Same as J1 for Channel 2 (`INB2` …). |
| **J10** | XT30UPB-M | Ch1 | Coil output (bridge `OUTA`/`OUTB`). High-current XT30. |
| **J5** | XT30UPB-M | Ch2 | Coil output for Channel 2. |
| **J4** | Conn_01x02 | Ch1 | "Capacitor array mounting" — 2-pin tap for an external capacitor bank. |
| **J3** | Conn_01x02 | Ch2 | "Capacitor array mounting" for Channel 2. |

### Mechanical

| Ref | Item |
|---|---|
| **H1, H2, H3, H8** | Mounting holes (four-corner board mounting). |

---

## 7. Full bill of materials (per sheet)

### Shared / power

| Ref | Value / Part | Notes |
|---|---|---|
| J49 | XT60PW-M | 12 V input |
| D1 | SMCJ18A | TVS, 18 V standoff |
| H1,H2,H3,H8 | MountingHole | mechanical |

### Per channel (Channel 1 ref / Channel 2 ref)

| Ch1 | Ch2 | Value / Part | Function |
|---|---|---|---|
| U2 | U1 | VNH5019A-E | Full-bridge motor driver |
| U4 | U3 | NC7SVU04P5X | Logic inverter (antiphase) |
| Q1 | Q2 | FDB8447L | N-ch power MOSFET (rail protection/switch) |
| C5 | C3 | 100 µF | Bulk cap (VBAT) |
| C7 | C8 | 0.1 µF | HF bypass (VBAT) |
| C4 | C6 | 1 µF | MOSFET/supply decoupling |
| C1 | C2 | 0.1 µF | +3V_1 decoupling |
| R3 | R10 | 4.7 kΩ | PWM line bias |
| R2 | R11 | 4.7 kΩ | EN/DIAG pull-up |
| R4 | R12 | 4.7 kΩ | CS_DIS pull-up |
| R6 | R8 | 1 kΩ | Series limit (control) |
| R5 | R1 | 1 kΩ | Series limit (control) |
| R7 | R9 | 1 kΩ | Series limit (CS) |
| J1 | J2 | Conn_01x06 | Signal header |
| J10 | J5 | XT30UPB-M | Coil output |
| J4 | J3 | Conn_01x02 | Cap-array mount |

---

## 8. Operational notes & cautions

- **CS_DIS default:** per the sheet, `CS_DIS` is pulled **high to disable** current sense;
  drive the corresponding header pin **low to enable** the analog `CS` reading.
- **EN/DIAG are open-drain fault flags:** if the VNH5019 detects over-temperature, short, or
  under-voltage it pulls `EN/DIAG` low. The firmware can read these to detect a faulted channel.
- **Inductive load:** coils are inductive; the TVS (D1) and bulk caps protect the supply, while
  the VNH5019's integrated body diodes handle the freewheel current. Keep the bulk cap close to
  the IC `VBAT` pins.
- **Two boards for a full cluster:** each board = 2 coils (2 A each). The 8 A aggregate /
  4-coil spec requires two of these boards.
- **Carrier polarity:** the carrier goes **directly** into the VNH5019 `PWM` pin (no inverter),
  so the firmware's carrier-duty convention must match the IC's active-high `PWM` enable. (This
  is the polarity relationship being tracked in the firmware-side debugging.)
- **Power/signal ground split:** route `PGND1` (high current) and `GND1` (logic) separately and
  star-join near the input.

---

## 9. Open questions / verify against netlist

1. **Exact resistor net assignment** — values are certain; per-pin mapping of the 1 kΩ vs
   4.7 kΩ resistors to specific `EN/DIAG`/`CS`/`PWM` nets should be confirmed in KiCad.
2. **Q1/Q2 (FDB8447L) topology** — confirm whether the MOSFET is reverse-polarity protection,
   a load switch, or soft-start, and how its gate is biased.
3. **Signal header (J1/J2) pinout** — confirm the exact pin order (which of the 6 pins is
   `INB`, `EN/DIAGA`, `EN/DIAGB`, `CS`, `CS_DIS`, `3V3/GND`) against the ESP32 channel-map in
   `README.md`.
4. **`CP` charge-pump cap** — verify presence/value of the charge-pump capacitor on the
   VNH5019 `CP` pin (required for correct high-side drive).
```
