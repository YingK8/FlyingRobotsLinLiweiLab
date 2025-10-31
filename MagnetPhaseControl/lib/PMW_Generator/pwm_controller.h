#ifndef PMW_CONTROLLER_H
#define PMW_CONTROLLER_H
#include <Arduino.h>
#include "pmw_config.h"

enum class OperationState { ACTIVE, STOPPED };
enum class OperationDirection { SWT_FORWARD_DIR, SWT_REVERSE_DIR, FIXED_FORWARD_DIR, FIXED_REVERSE_DIR };
enum class RequestState { REQ_START_FWD, REQ_START_REV, REQ_STOP, NO_REQUEST_PENDING };
enum class DutyStartType { DUTY_25_PERCENT, DUTY_10_PERCENT, DUTY_02_PERCENT };
enum class DutyRangeType { NO_LIMIT, MAX_25, MAX_25_MINUS_DEADTIME };
enum class DutyRampType { RAMP_ENABLED, RAMP_DISABLED };

class PWMController {
public:
    void initialize();
    void handleRequests();
    void handleDutyAdjustment();
    
    // Public for ISR access
    OperationState opState = OperationState::STOPPED;
    OperationDirection opDirection = OperationDirection::SWT_FORWARD_DIR;
    RequestState currentRequest = RequestState::NO_REQUEST_PENDING;
    
    // Configuration (set in initialize())
    DutyRangeType dutyCycleRange = DutyRangeType::NO_LIMIT;
    DutyStartType startUpDuty = DutyStartType::DUTY_25_PERCENT;
    DutyRampType dutyDownRamp = DutyRampType::RAMP_DISABLED;

private:
    void startPWMForward();
    void startPWMReverse();
    void stopPWM();
    void dutyAdjust(int up_down);
    void dutyRampDown();
    float getDutyValue();
    
    int startUpDutyCycleValue = 0;
    int ocrResult = 0;
};

extern PWMController pwmController;

#endif // PMW_CONTROLLER_H