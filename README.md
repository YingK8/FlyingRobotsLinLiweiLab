# FlyingRobotsLinLiweiLab

A magnetically actuated micro flying robot. A four-coil phased array drives a
rotating field that spins and tilts a passive magnetic rotor; a stereo camera
rig closes the loop on its pose.

| Where | What |
|---|---|
| [`ESP32_PMW/`](ESP32_PMW/README.md) | ESP32 firmware: phased PWM coil drive, current sensing, balance loop |
| [`ESP32_PMW/controller/`](ESP32_PMW/controller/README.md) | the vision → control pipeline (camera → calib → pose → control) |
| `ESP32_PMW/ai/` | sweeps, system ID, validation, instrumentation, plotting |
| `PCB/`, `ESP32_PMW/docs/` | the H-bridge power stage: KiCad sources and the board writeup |
| `3DModels/` | printed frame, coil plate, takeoff rig |
| `calculations/` | coil resonance and capacitor sizing |
| `pico/` | earlier RP2040 current-characterization work (see `pico/PROGRESS.md`) |

Start with [`ESP32_PMW/README.md`](ESP32_PMW/README.md) for the firmware, or
[`ESP32_PMW/controller/README.md`](ESP32_PMW/controller/README.md) for the
control pipeline. Each controller stage carries its own `theory.md` chapter
deriving what it implements.
