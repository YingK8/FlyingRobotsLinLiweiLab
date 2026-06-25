CsvPhaseSequencer: manually sequence trajectory waypoints (100%->99%->98%...)
JsonPhaseSequencer: automatically sequence tasks (ramp up vs down)
PhaseSequencer: you can also directly call task sequencing inside main.cpp
PhaseController: low-level coil controller that generates software-timed pwm
CoilBalancer: low-level current-balance feedback loop; reads per-coil CS ADC and trims carrier duty so opposing pairs (A=B, C=D) carry equal current at the highest common level