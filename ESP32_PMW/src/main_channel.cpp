#include <Arduino.h>
#include "PhaseController.h"
#include "PhaseSequencer.h"
#include <FS.h>
#include <SPIFFS.h>

// CONFIGURATION
const int NUM_CHANNELS = 4;


const gpio_num_t A_PWM_PIN = GPIO_NUM_32;
const gpio_num_t B_PWM_PIN = GPIO_NUM_23;
const gpio_num_t C_PWM_PIN = GPIO_NUM_27;
// FIXED: Changed from 35 (Input Only) to 23 (Safe Output)
const gpio_num_t D_PWM_PIN = GPIO_NUM_25; 

const gpio_num_t A_CARRIER_PIN = GPIO_NUM_33;
const gpio_num_t B_CARRIER_PIN = GPIO_NUM_13;
const gpio_num_t C_CARRIER_PIN = GPIO_NUM_14;
const gpio_num_t D_CARRIER_PIN = GPIO_NUM_26;

const gpio_num_t PWM_PINS[NUM_CHANNELS] =     {A_PWM_PIN,     B_PWM_PIN,      C_PWM_PIN,      D_PWM_PIN};
const gpio_num_t CARRIER_PINS[NUM_CHANNELS] = {A_CARRIER_PIN, B_CARRIER_PIN,  C_CARRIER_PIN,  D_CARRIER_PIN};

// rotation is counter-clockwise: A -> C -> B -> D
const float INITIAL_PHASES[NUM_CHANNELS] = {0.0, 180.0, 90.0, 270.0};
const float INITIAL_DUTY_CYCLES[NUM_CHANNELS] = {50.0, 50.0, 50.0, 50.0};
const gpio_num_t SYNC_PIN = GPIO_NUM_8;

const int PWM_FREQ = 20000; 
const float carrier_duty = 100.0;
const float INITIAL_CARRIER_DUTY_CYCLES[NUM_CHANNELS] = {carrier_duty, carrier_duty, carrier_duty, carrier_duty};

// const float start_freq = 1.0f;
// const float end_freq = 190.0f;
// const unsigned long ramp_duration_ms = 40000;

const float start_freq = 1.0f;
const float end_freq = 500.0f;
const unsigned long ramp_duration_ms = 8000;
// const unsigned long second_ramp_duration_ms = 5000;

// --- INDICATOR LED CONFIGURATION ---
const int LED_PIN = 2; 
bool led_state = false; 
unsigned long lastBlinkTime = 0;       // <-- added for blink timing

// declare as global POINTERS (do not initialize yet)
PhaseController* controller;
PhaseSequencer* seq;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);
  delay(1000);
  // SPIFFS.begin(true); 
  
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW); 

  // pinMode(GPIO_NUM_32, INPUT);

  // instantiate the objects dynamically NOW that the OS has booted
  controller = new PhaseController(PWM_PINS, INITIAL_PHASES, INITIAL_DUTY_CYCLES, NUM_CHANNELS);
  controller->begin();

  // create sequencer with valid controller pointer
  seq = new PhaseSequencer(controller);
  controller->initCarrierPWM(CARRIER_PINS, PWM_FREQ, INITIAL_CARRIER_DUTY_CYCLES);

  seq->add
  // seq->addLinearRampTask(start_freq, end_freq, ramp_duration_ms); 
  seq->compile(25, 0.0f, INITIAL_DUTY_CYCLES, INITIAL_PHASES);

  seq->start();
}

void loop() {
  controller->run(); 
  seq->run();  

  // Serial.print(digitalRead(GPIO_NUM_32));

  // --- Blink LED and print frequency every 500 ms ---
  unsigned long now = millis();
  if (now - lastBlinkTime >= 500) {
    lastBlinkTime = now;
    led_state = !led_state;
    digitalWrite(LED_PIN, led_state);

    // Retrieve current ramp frequency (adjust method name if needed)
    float freq = controller->getFrequency();  
    Serial.print("Ramp Frequency: ");
    Serial.print(freq);
    Serial.println(" Hz");
  }
}