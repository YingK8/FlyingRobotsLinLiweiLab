#include "MegaPhasePwm.h"

// --- Class Method Definitions ---

MegaPhasePwm::MegaPhasePwm() {
    m_icr = 0;
    m_prescalerBits = 0;
    for (int i = 0; i < 6; i++) m_duty[i] = 0.5;
    for (int i = 0; i < 3; i++) m_phase[i] = 0.0;
}

void MegaPhasePwm::begin(float initialFrequency) {
    m_phase[0] = 0.0;
    m_phase[1] = 60.0;  
    m_phase[2] = 120.0;
    setFrequency(initialFrequency);
}

void MegaPhasePwm::stopTimers() {
    GTCCR = 0x81;
    TCCR1B = 0x18;
    TCCR3B = 0x18;
    TCCR4B = 0x18;
    GTCCR = 0x0;

    pinMode(11, INPUT);
    pinMode(12, INPUT);
    pinMode(5, INPUT);
    pinMode(2, INPUT);
    pinMode(6, INPUT);
    pinMode(7, INPUT);
}

void MegaPhasePwm::startTimersSynchronously() {
    pinMode(14, OUTPUT);
    digitalWrite(14, HIGH);
    digitalWrite(14, LOW);
    
    GTCCR = 0x81;
    setPhase(0, m_phase[0]);
    setPhase(1, m_phase[1]);
    setPhase(2, m_phase[2]);

    TCCR1B |= m_prescalerBits;
    TCCR3B |= m_prescalerBits;
    TCCR4B |= m_prescalerBits;
    GTCCR = 0x0;
}

uint8_t MegaPhasePwm::calculatePrescaler(float freq, uint32_t& icrVal) {
    const long F_CPU_LONG = F_CPU;
    uint32_t icr_calc;
    uint8_t prescaler_bits;

    // Try prescalers in order
    icr_calc = (uint32_t)((F_CPU_LONG / (1.0 * freq)) - 1.0);
    if (icr_calc <= 65535) {
        prescaler_bits = 0x01;
    } else {
        icr_calc = (uint32_t)((F_CPU_LONG / (8.0 * freq)) - 1.0);
        if (icr_calc <= 65535) {
            prescaler_bits = 0x02;
        } else {
            icr_calc = (uint32_t)((F_CPU_LONG / (64.0 * freq)) - 1.0);
            if (icr_calc <= 65535) {
                prescaler_bits = 0x03;
            } else {
                icr_calc = (uint32_t)((F_CPU_LONG / (256.0 * freq)) - 1.0);
                if (icr_calc <= 65535) {
                    prescaler_bits = 0x04;
                } else {
                    icr_calc = (uint32_t)((F_CPU_LONG / (1024.0 * freq)) - 1.0);
                    if (icr_calc > 65535) icr_calc = 65535;
                    prescaler_bits = 0x05;
                }
            }
        }
    }
    
    icrVal = icr_calc;
    return prescaler_bits;
}

float MegaPhasePwm::setFrequency(float freq) {
    uint32_t new_icr;
    uint8_t new_prescalerBits = calculatePrescaler(freq, new_icr);

    if (new_prescalerBits != m_prescalerBits) {
        // Hard restart needed for prescaler change
        stopTimers();
        m_prescalerBits = new_prescalerBits;
        m_icr = (uint16_t)new_icr;

        setupPins();

        // Configure all timers for Fast PWM Mode 14
        TCCR1B = 0x18; TCCR1A = 0xA2; ICR1 = m_icr;
        TCCR3B = 0x18; TCCR3A = 0xA2; ICR3 = m_icr;
        TCCR4B = 0x18; TCCR4A = 0xA2; ICR4 = m_icr;

        // Restore duty cycles
        for(int i=0; i<6; i++) setDutyCycle(i, m_duty[i]);

        startTimersSynchronously();
    } else {
        // Smooth frequency change - update ICR and phases
        m_icr = (uint16_t)new_icr;
        
        // Update ICR values (buffered, takes effect at next cycle)
        ICR1 = m_icr;
        ICR3 = m_icr;
        ICR4 = m_icr;
        
        // Small delay to let timers process the ICR change
        // delayMicroseconds(10);
        
        // Re-apply duty cycles and phases
        for(int i=0; i<6; i++) setDutyCycle(i, m_duty[i]);
        for(int i=0; i<3; i++) setPhase(i, m_phase[i]);
    }

    // Calculate actual frequency
    float prescalerVal = 1.0;
    if (m_prescalerBits == 0x02) prescalerVal = 8.0;
    else if (m_prescalerBits == 0x03) prescalerVal = 64.0;
    else if (m_prescalerBits == 0x04) prescalerVal = 256.0;
    else if (m_prescalerBits == 0x05) prescalerVal = 1024.0;
    
    return (F_CPU / (prescalerVal * ((float)m_icr + 1.0)));
}

void MegaPhasePwm::setDutyCycle(uint8_t channel, float duty) {
    if (duty < 0.0) duty = 0.0;
    if (duty > 1.0) duty = 1.0;

    m_duty[channel] = duty;
    
    if (m_icr == 0) return;
    
    uint16_t ocrVal = (uint16_t)((float)m_icr * duty);
    
    switch (channel) {
        case 0: OCR1A = ocrVal; break;
        case 1: OCR1B = ocrVal; break;
        case 2: OCR3A = ocrVal; break;
        case 3: OCR3B = ocrVal; break;
        case 4: OCR4A = ocrVal; break;
        case 5: OCR4B = ocrVal; break;
    }
}

void MegaPhasePwm::setPhase(uint8_t timerBank, float degrees) {
    degrees = fmodf(degrees, 360.0f);
    if (degrees < 0.0) degrees += 360.0;
    
    m_phase[timerBank] = degrees;
    
    if (m_icr == 0) return;

    uint32_t period_ticks = (uint32_t)m_icr + 1L;
    uint16_t phase_ticks = (uint16_t)((degrees / 360.0) * (float)period_ticks);

    // Add small delay for smoother transitions
    // delayMicroseconds(1);
    
    noInterrupts();
    switch (timerBank) {
        case 0: TCNT1 = phase_ticks; break;
        case 1: TCNT3 = phase_ticks; break;
        case 2: TCNT4 = phase_ticks; break;
    }
    interrupts();
}

void MegaPhasePwm::setupPins() {
    pinMode(11, OUTPUT);
    pinMode(12, OUTPUT);
    pinMode(5, OUTPUT);
    pinMode(2, OUTPUT);
    pinMode(6, OUTPUT);
    pinMode(7, OUTPUT);
}