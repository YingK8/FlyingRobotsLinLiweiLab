// Duty-cycle ramp experiment — AUTOMATED one-shot sweep (no serial needed).
//
// Two VNH5019 boards: board 1 = A,C  /  board 2 = B,D (channel index 0=A,1=B,2=C,3=D).
// Each board drives one opposed coaxial coil pair (axis 1 = A&C, axis 2 = B&D).
//
// ONE BOARD AT A TIME — no parallel runs. The inactive board is held completely
// at 0 (both its channels' carrier duty = 0) for the entire time the other board
// is active. This isolates the within-board opposed-pair mutual coupling and that
// board's own supply sag, with zero cross-board interference. Per board we ramp
// each coil solo, then both coils together:
//
//   board 1:  gap->ramp A   gap->ramp C   gap->ramp {A,C}
//   board 2:  gap->ramp B   gap->ramp D   gap->ramp {B,D}
//   -> graceful shutdown (ramp drive to 0, halt)
//
// Comparing a coil's peak solo vs with its same-board partner shows the mutual
// coupling / per-board supply sag within an axis.
//
// Common-ground + no USB isolator means we can't talk to the ESP32 while powered,
// so the sequence runs ONCE on boot, then de-energizes the coils. Start the
// PicoScope recorder and reset the ESP32 together; capture >= ~62 s.
// Constant 190 Hz rotating-field drive; only the carrier duty is ramped.

#include <Arduino.h>
#include "PhaseController.h"

const int NUM_CHANNELS = 4;

const gpio_num_t A_PWM_PIN = GPIO_NUM_32;
const gpio_num_t B_PWM_PIN = GPIO_NUM_23;
const gpio_num_t C_PWM_PIN = GPIO_NUM_27;
const gpio_num_t D_PWM_PIN = GPIO_NUM_25;

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_33;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_13;
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_14;
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_26;

const gpio_num_t PWM_PINS[NUM_CHANNELS] =     {A_PWM_PIN,     B_PWM_PIN,     C_PWM_PIN,     D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN, C_CARRIER_PIN, D_CARRIER_PIN};

// rotation is counter-clockwise: A -> C -> B -> D
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 180.0, 90.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};

const int PWM_FREQ = 15000;       // carrier frequency (Hz)
const float DRIVE_FREQ = 190.0f;  // coil drive (commutation) frequency (Hz)
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {0, 0, 0, 0};

// Sweep timing.
const unsigned long RAMP_MS = 8000;       // 0 -> 100% carrier ramp per segment
const unsigned long GAP_MS  = 2000;       // off-gap before each segment
const unsigned long SHUTDOWN_MS = 2000;   // graceful ramp-down at the end

// Active set per segment, as a channel bitmask (bit i = channel i; 0=A,1=B,2=C,3=D).
// One board at a time; every channel not in the mask is held at 0 (incl. the
// entire inactive board).
const uint8_t SEQ[] = {0b0001,   // A      board 1, solo
                       0b0100,   // C      board 1, solo
                       0b0101,   // A,C    board 1, both (opposed pair)
                       0b0010,   // B      board 2, solo
                       0b1000,   // D      board 2, solo
                       0b1010};  // B,D    board 2, both (opposed pair)
const char *const SEG_LABEL[] = {"A", "C", "A+C", "B", "D", "B+D"};
const int N_SEQ = sizeof(SEQ) / sizeof(SEQ[0]);

const int LED_PIN = 2;

PhaseController *controller;
unsigned long start_ms = 0;
int last_idx = -1;
bool done = false;

void setup() {
  Serial.begin(115200);
  delay(1000);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES,
                                   NUM_CHANNELS);
  controller->begin();
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);
  controller->setGlobalFrequency(DRIVE_FREQ);

  Serial.printf("duty-ramp sweep: %d segments, %lums ramp / %lums gap, total %lums\n",
                N_SEQ, RAMP_MS, GAP_MS, N_SEQ * (GAP_MS + RAMP_MS));
  start_ms = millis();
}

void loop() {
  if (done) {
    delay(100);
    return;
  }

  controller->run();

  const unsigned long slot = GAP_MS + RAMP_MS;
  unsigned long elapsed = millis() - start_ms;

  if (elapsed < (unsigned long)N_SEQ * slot) {
    int idx = elapsed / slot;
    unsigned long within = elapsed % slot;
    float duty = (within < GAP_MS) ? 0.0f
                                   : 100.0f * (within - GAP_MS) / RAMP_MS;
    // Every channel not in the mask (including the whole inactive board) gets 0.
    for (int i = 0; i < NUM_CHANNELS; i++)
      controller->setCarrierDutyCycle(i, ((SEQ[idx] >> i) & 1) ? duty : 0.0f);
    digitalWrite(LED_PIN, duty > 0.0f ? HIGH : LOW);

    if (idx != last_idx) {
      last_idx = idx;
      Serial.printf("t=%lu  segment %d  [%s]  mask=0x%X\n",
                    millis(), idx, SEG_LABEL[idx], SEQ[idx]);
    }
  } else {
    Serial.printf("t=%lu  sweep complete -> graceful shutdown\n", millis());
    controller->shutdown(SHUTDOWN_MS);   // ramp all coils to 0 and halt
    digitalWrite(LED_PIN, LOW);
    done = true;
  }
}
