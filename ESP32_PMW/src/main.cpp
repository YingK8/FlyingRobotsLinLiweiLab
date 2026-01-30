/**
 * ESP32 4-Channel Smooth Phase Controller (Jitter-Free)
 * * 修复: 解决了改变频率时的波形抖动/断裂问题
 * * 原理: 使用"相位累加器" (Phase Accumulator) 代替绝对时间取模
 * * 功能: 保持波形连续性 (Phase Continuity)
 */

#include "Arduino.h"

// --- 引脚定义 ---
const int PIN_PHASE_0   = 15; // A组
const int PIN_PHASE_90  = 33; // C组
const int PIN_PHASE_180 = 12; // B组
const int PIN_PHASE_270 = 27; // D组

// --- 动态运行变量 ---
int pwmFreqHz = 50;
float periodUs = 1000000.0 / pwmFreqHz; // 使用 float 提高精度

// --- 相位累加器变量 ---
// 这些变量用于追踪我们在波形中的当前位置，而不是依赖系统绝对时间
float currentCyclePos = 0;       // 当前周期内的位置 (0 -> periodUs)
unsigned long lastLoopMicros = 0; // 上次循环的时间

// --- 频率控制变量 ---
unsigned long lastFreqUpdateTime = 0;
int targetFreqHz = 350;
float FreqStep = 10.0;

// --- 独立 Duty Cycle 变量 (0.0 - 100.0) ---
float dutyCycle0   = 50.0; 
float dutyCycle90  = 50.0; 
float dutyCycle180 = 50.0; 
float dutyCycle270 = 50.0; 

// --- 内部计算变量 ---
unsigned long width0, width90, width180, width270;
unsigned long offset0, offset90, offset180, offset270;

/**
 * 设置频率并保持相位连续性
 * 关键算法: 频率改变时，按照比例缩放当前的位置
 */
void setFrequency(int newHz) {
  if (newHz < 1 || newHz > 1000) return;

  float oldPeriod = periodUs;
  float newPeriod = 1000000.0 / newHz;

  // 1. 计算我们当前在旧周期中的百分比位置 (Ratio)
  float ratio = currentCyclePos / oldPeriod;

  // 2. 更新频率和周期
  pwmFreqHz = newHz;
  periodUs = newPeriod;

  // 3. 将当前位置映射到新周期 (保持百分比不变)
  // 这就像拉伸橡皮筋一样，波形不会断裂
  currentCyclePos = ratio * newPeriod;

  // 4. 重新计算相位偏移点
  offset0   = 0;
  offset90  = (unsigned long)(periodUs * 0.25);
  offset180 = (unsigned long)(periodUs * 0.50);
  offset270 = (unsigned long)(periodUs * 0.75);
}

void setup() {
  Serial.begin(115200);
  Serial.println("--- ESP32 无抖动磁场控制器 ---");
  
  pinMode(PIN_PHASE_0, OUTPUT);
  pinMode(PIN_PHASE_90, OUTPUT);
  pinMode(PIN_PHASE_180, OUTPUT);
  pinMode(PIN_PHASE_270, OUTPUT);

  setFrequency(50); 
  lastLoopMicros = micros();
}

// 辅助函数: 检查相位窗口
bool checkPhase(float currentPos, unsigned long shift, unsigned long width) {
  if (width == 0) return false;
  
  // 浮点数比较，加上一点容差
  float endPos = shift + width;
  
  // 处理跨越周期末尾的情况 (Wrap around)
  // 如果 结束点 > 周期，说明窗口跨越了 0点
  if (endPos > periodUs) {
    float wrappedEnd = endPos - periodUs;
    return (currentPos >= shift || currentPos < wrappedEnd);
  } else {
    // 正常情况
    return (currentPos >= shift && currentPos < endPos);
  }
}

void loop() {
  // 1. 计算时间增量 (Delta Time)
  unsigned long now = micros();
  unsigned long dt = now - lastLoopMicros;
  lastLoopMicros = now;

  // 2. 频率平滑扫描控制
  // 每 0.1秒 增加 10Hz
  if (now - lastFreqUpdateTime >= 100000) { 
    lastFreqUpdateTime = now;
    if (pwmFreqHz < targetFreqHz) {
      setFrequency(pwmFreqHz + FreqStep);
    } else if (pwmFreqHz > targetFreqHz) {
      setFrequency(pwmFreqHz - FreqStep);
    }
  }

  // 3. 推进相位累加器
  // 我们手动增加位置，而不是询问系统时间
  currentCyclePos += dt;

  // 4. 处理周期循环
  while (currentCyclePos >= periodUs) {
    currentCyclePos -= periodUs;
  }

  // 5. 计算脉宽
  width0   = periodUs * dutyCycle0 / 100.0;
  width90  = periodUs * dutyCycle90 / 100.0;
  width180 = periodUs * dutyCycle180 / 100.0;
  width270 = (unsigned long)(periodUs * dutyCycle270 / 100.0);

  // 6. 更新引脚
  // 使用当前累加的位置进行判断
  digitalWrite(PIN_PHASE_0,   checkPhase(currentCyclePos, offset0,   width0));
  digitalWrite(PIN_PHASE_90,  checkPhase(currentCyclePos, offset90,  width90));
  digitalWrite(PIN_PHASE_180, checkPhase(currentCyclePos, offset180, width180));
  digitalWrite(PIN_PHASE_270, checkPhase(currentCyclePos, offset270, width270));
}