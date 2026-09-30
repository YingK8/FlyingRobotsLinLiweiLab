#pragma once

#include "driver/gpio.h"
#include <Arduino.h>

/**
 * Synchronous lock-in on the VNH5019 CS pins: measures the phase of the coil
 * CURRENT against the phase the firmware commanded, one burst at a fixed
 * frequency. This is the measurement `controller/control/coil_phase.py` consumes
 * and `controller/control/theory.md` 22 derives.
 *
 * WHY THIS IS NOT THE CURRENT-SENSE PATH IN current_sense.h
 * --------------------------------------------------------
 * That path is built to report amplitude and deliberately destroys phase three
 * separate ways: the ADC is paced at ~1 kHz, a 50 ms EMA smooths it, and
 * driveTelemetry() prints at 2 Hz. This one bypasses all three -- raw reads, no
 * filter, accumulated in quadrature and reported once at the end.
 *
 * WHY 2f AND NOT f
 * ----------------
 * The VNH5019 CS pin is UNSIGNED: it mirrors whichever high-side FET is sourcing,
 * so a sinusoidal coil current arrives full-wave rectified. |sin(x)| has no
 * component at its own rate; its first harmonic is at 2x. So the lock-in runs at
 * 2f and halves the angle, which recovers the current phase modulo 180 deg --
 * harmless when the spread being measured is a few degrees.
 *
 * WHY THIS DOES NOT REOPEN FIELD-ORIENTED CONTROL (theory.md 17.2)
 * ---------------------------------------------------------------
 * 17.2 rules FOC out because commutation needs SIGNED per-phase current sampled
 * synchronously EVERY CYCLE in real time. This is a calibration: one frequency,
 * held, averaged over hundreds of cycles, off the flight path. It answers a
 * question 17.2 was not asking and does not give the loop an angle it can fly on.
 *
 * MUX SKEW
 * --------
 * The ESP32 ADC needs a throwaway read after switching pins (see current_sense.cpp),
 * so the four channels are sampled sequentially, not simultaneously. At 200 Hz each
 * 50 us of skew is 3.6 deg -- the whole effect being measured. Rather than correct a
 * nominal offset, every sample carries its OWN timestamp taken next to the
 * conversion, so the skew never enters the arithmetic.
 */

struct ProbeResult {
  float amp[4];        ///< lock-in magnitude at 2f, ADC counts (gain is arbitrary).
  float phaseDeg[4];   ///< current phase re. commanded, deg, wrapped to (-90, +90].
  float coherence;     ///< worst channel: fraction of AC power sitting at 2f. 0..1.
  uint32_t n;          ///< samples per channel.
};

/**
 * @brief Run one lock-in burst. Blocking for `ms`. The caller must already have the
 *        coils driving at `fHz` and settled.
 * @param adcPins            CS ADC pins (constants.h::ADC_PINS).
 * @param nCh                channels to sample (<= 4).
 * @param fHz                DRIVE frequency; the lock-in runs at twice this.
 * @param ms                 burst length. Longer is quieter, and hotter.
 * @param commandedPhaseDeg  what each channel was told to do (e.g. PHASES_CW).
 * @param out                result; untouched on failure.
 * @return false if the arguments are unusable or no samples were taken.
 */
bool coilProbe(const gpio_num_t *adcPins, int nCh, float fHz, uint32_t ms,
               const float *commandedPhaseDeg, ProbeResult &out);
