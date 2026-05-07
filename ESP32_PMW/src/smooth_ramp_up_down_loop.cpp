// void loop() {
//   controller.run(); // hardware timer drift compensation
//   seq.run();        // state machine queue

//   // Note: Changed from 200.0f to 200.0f so it triggers when your ramp finishes
//   if (controller.getFrequency() >= 200.0f) { 
//     unsigned long current_millis = millis();

//     // 1. Process State Machine
//     switch (sweep_state) {
//       case SWEEPING_DOWN:
//         carrier_duty_sweep -= d_duty_step;
//         if (carrier_duty_sweep <= lower_d) {
//           carrier_duty_sweep = lower_d;
//           sweep_state = WAITING_BOTTOM;       // Move to pause state
//           wait_start_time = current_millis;   // Record when pause started
//         }
//         break;

//       case WAITING_BOTTOM:
//         // Pause between ramp down and ramp up
//         if (current_millis - wait_start_time >= wait_time_ms) {
//           sweep_state = SWEEPING_UP;
//         }
//         break;

//       case SWEEPING_UP:
//         carrier_duty_sweep += d_duty_step;
//         if (carrier_duty_sweep >= 100.0f) {
//           carrier_duty_sweep = 100.0f;
//           sweep_state = WAITING_TOP;          // Move to pause state
//           wait_start_time = current_millis;   // Record when pause started
//         }
//         break;

//       case WAITING_TOP:
//         // Pause between ramp up and switching pairs
//         if (current_millis - wait_start_time >= wait_time_ms) {
//           active_pair = (active_pair == 0) ? 1 : 0; // Switch active pairs
//           sweep_state = SWEEPING_DOWN;              // Start over
//         }
//         break;
//     }

//     // 2. Apply Output States
//     if (active_pair == 0) {
//       // Pair 0 & 3 is active, Pair 1 & 2 rests at 100%
//       controller.setCarrierDutyCycle(0, carrier_duty_sweep);
//       controller.setCarrierDutyCycle(3, carrier_duty_sweep);
//       controller.setCarrierDutyCycle(1, 100.0f);
//       controller.setCarrierDutyCycle(2, 100.0f);
//     } else {
//       // Pair 1 & 2 is active, Pair 0 & 3 rests at 100%
//       controller.setCarrierDutyCycle(0, 100.0f);
//       controller.setCarrierDutyCycle(3, 100.0f);
//       controller.setCarrierDutyCycle(1, carrier_duty_sweep);
//       controller.setCarrierDutyCycle(2, carrier_duty_sweep);
//     }
//   }
// }