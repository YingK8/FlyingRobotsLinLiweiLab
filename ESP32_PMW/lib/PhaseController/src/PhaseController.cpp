#include "PhaseController.h"

PhaseController::PhaseController(const int* pins, const float* phaseOffsetsDegrees, int numChannels) {
    _numChannels = numChannels;

    // 动态分配内存
    _pins = new int[_numChannels];
    _phaseOffsetsPct = new float[_numChannels];
    _dutyCycles = new float[_numChannels];
    _params = new PhaseParams[_numChannels];

    // 初始化参数
    for (int i = 0; i < _numChannels; i++) {
        _pins[i] = pins[i];
        _phaseOffsetsPct[i] = constrain(phaseOffsetsDegrees[i], 0.0, 360.0) / 360.0;
        _dutyCycles[i] = 50.0;
    }

    _pwmFreqHz = 10.0;
    _currentCyclePos = 0;
    _lastLoopMicros = 0;
}

PhaseController::~PhaseController() {
    delete[] _pins;
    delete[] _phaseOffsetsPct;
    delete[] _dutyCycles;
    delete[] _params;
}

void PhaseController::begin(float initialFreqHz) {
    for(int i = 0; i < _numChannels; i++) {
        pinMode(_pins[i], OUTPUT);
        digitalWrite(_pins[i], LOW);
    }
    setFrequency(initialFreqHz);
    _lastLoopMicros = micros();
}

void PhaseController::setFrequency(float newHz) {
    if (newHz < 0.1 || newHz > 2000.0) return; 

    float oldPeriod = _periodUs;
    float newPeriod = 1000000.0 / newHz;

    // 保持相位连续性：按比例缩放当前周期位置
    if (oldPeriod > 0) {
        _currentCyclePos = (_currentCyclePos / oldPeriod) * newPeriod;
    } else {
        _currentCyclePos = 0;
    }

    _pwmFreqHz = newHz;
    _periodUs = newPeriod;
    updatePhaseParams();
}

void PhaseController::setDutyCycle(int channel, float dutyPercent) {
    if(channel < 0 || channel >= _numChannels) return;
    _dutyCycles[channel] = constrain(dutyPercent, 0.0, 100.0);
    updatePhaseParams();
}

void PhaseController::setGlobalDutyCycle(float dutyPercent) {
    for(int i = 0; i < _numChannels; i++) {
        _dutyCycles[i] = constrain(dutyPercent, 0.0, 100.0);
    }
    updatePhaseParams();
}

float PhaseController::getFrequency() const {
    return _pwmFreqHz;
}

// 预计算所有通道的时间参数，减少 run() 中的运算量
void PhaseController::updatePhaseParams() {
    for(int i = 0; i < _numChannels; i++) {
        float width = _periodUs * _dutyCycles[i] / 100.0;
        
        _params[i].start = _periodUs * _phaseOffsetsPct[i];
        _params[i].end = _params[i].start + width;

        // 处理跨越周期边界的情况 (Wraparound)
        if (_params[i].end > _periodUs) {
            _params[i].end -= _periodUs;
            _params[i].wraps = true;
        } else {
            _params[i].wraps = false;
        }
    }
}

void PhaseController::run() {
    // 1. 时间步进
    unsigned long now = micros();
    unsigned long dt = now - _lastLoopMicros;
    _lastLoopMicros = now;

    if (dt > _periodUs * 2) dt = 0; // 防止溢出错误

    // 2. 更新相位
    _currentCyclePos += dt;
    if (_currentCyclePos >= _periodUs) {
        _currentCyclePos -= _periodUs;
    }

    // 3. 更新所有通道输出
    for(int i = 0; i < _numChannels; i++) {
        bool active;
        if (_params[i].wraps) {
            // 跨周期：当前位置在 [Start, End] 范围之外时为低，反之为高
            active = (_currentCyclePos >= _params[i].start || _currentCyclePos < _params[i].end);
        } else {
            // 正常：当前位置在 [Start, End] 范围内为高
            active = (_currentCyclePos >= _params[i].start && _currentCyclePos < _params[i].end);
        }
        digitalWrite(_pins[i], active);
    }
}