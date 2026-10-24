# Flight START 对时：第一版

更新：2026-10-23。支持 `flight` 固件 + Python `hover_controller_runner`。代码已接入；尚未烧录、实拍或验证光学/电流延迟。`tilt` 和独立 `pmw_fly` 尚未接入这套采集流程。

## 工作方式

1. Python runner 打开唯一串口，保存原始接收日志，按既有流程 reset 到 IDLE。
2. 新 flight 固件初始化后 LED 保持灭，输出 `SYNC_READY v=1`。
3. 同一个 runner 打开并录制双相机。对时模式要求至少 0.25 秒的录制源时间跨度后才允许 START；这是软件前置检查，不证明帧已经成功编码或 LED 确实可见。
4. 使用原有 `seq=clear/ramp/go` 启动路径。在接受 `seq=go` 后，固件依次切换 LED、应用 sequencer 初始状态及四路 carrier，并保存调用前后的设备时间；随后才打印事件。
5. 离线 `sync-start` 找到两机各一个 LED 上升沿，并与唯一的设备事件关联，建立相机 A、相机 B、ESP 到实验时间的映射。

一次 run 只允许一次 START。重拍、重启、停止后再起飞都新建 run。对时模式的面板 takeoff 不再额外 reset；构造 runner 时已经 reset。其他模式保持原有 re-arm 行为。

## 事件与时间含义

固件输出格式：

```text
SYNC_READY v=1
SYNC_START v=1 led0=10000000 led1=10000010 drive0=10000012 drive1=10000100
```

示例数字不是测量结果。四个时间都是 `esp_timer_get_time()` 的微秒值：

- `led0/led1` 包围 LED GPIO 写操作。
- `drive0/drive1` 包围 `seq.start()` 和首次 `applyMixer()`；各通道写入并非同时发生。
- 没有在这两个操作之间打印串口，以免串口阻塞人为拉开间隔。
- 正常 stop/landing 完成时 LED 灭；GPIO14 的阻断行为保留。LED 灭不是精确电流归零的测量。

这些是**软件调用时间区间**，不是示波器测得的 PWM 边沿、LED 发光时刻或电流建立时刻。离线暂取 LED 调用区间中点为 ESP 的 START 参考，光学边沿有帧间区间与未知曝光/传输延迟；`hardware_sync_validated=false`。实际电流延迟不消除。

当前 Arduino ESP32 的 `millis()` 和上述微秒时钟同源，原有电流日志 `t=` 仍按毫秒解析。第一版拒绝跨设备 reset、多 START 和 millis 回绕。单次事件只估 offset，不声称验证了漂移；结束第二脉冲尚未实现。

## 上机前准备

- 使用匹配实际机位、相机顺序、分辨率和旋转的标定；世界坐标仍按静态参考单独建立。
- 确认两台相机都能清晰看到同一个 START LED，位置与运动目标不重叠。
- 需要先由实验者按既有流程刷入更新后的 **flight** 固件。这份文档与本轮验证没有执行 upload，也没有改变 GPIO map。
- 保留原有供电、停止操作和 thermal 流程。这里没有新增串口驱动入口，不运行另一个相机 recorder 或串口 reader。

## 采集配置

先用 CLI 新建 run（以下命令在 `ESP32_PMW/` 执行）：

```bash
.venv/bin/python -m LLM init LLM/runs/run_001
```

在你现有、已经确认参数的 Python flight 配置 `existing_cfg` 上添加以下选项。不要从示例猜测端口、相机索引、标定或驱动参数：

```python
from dataclasses import replace
from pathlib import Path
from controller.control.hover_controller_runner import fly

run = Path("LLM/runs/run_001")
cfg = replace(
    existing_cfg,
    source="camera",
    record=str(run / "raw/captures"),
    csv_dir=str(run / "raw/controller"),
    log=None,
    telemetry_path=str(run / "raw/telemetry.jsonl"),
    sync_start=True,
    takeoff=False,  # 先看画面；准备好后用既有面板 takeoff
)
# 这是实际硬件操作：仅在你准备好进行该次实验时运行。
fly(cfg)
```

`existing_cfg` 需要保持你确认过的 `port/camera/rig/width/height/fps/segments` 等字段；`dry_run` 必须为 False。运行时确认 LED 初始为暗、双视频正在记录，再按面板 takeoff。过早按键会拒绝启动并退出；重新准备新 run。

原始串口 JSONL 以独占模式创建，不覆盖旧实验。记录点在现有 `SerialComm` 唯一 reader 内，包括发命令时碰巧收到的行。`host_receive_s` 是 host 处理收到行的时刻，不是曝光或设备应用时间。原 runner 的 `.log/.csv` 仍保存 host 命令与 ramp 配置。

录制数据位于 `raw/captures/<程序打印的目录>/`，保留 `frames.csv/meta.json`。结束时仍通过原有停止流程关闭线圈。

## 离线导入与对齐

完成录制后：

```bash
.venv/bin/python -m LLM import-flight LLM/runs/run_001 \
  LLM/runs/run_001/raw/captures/实际录像目录

.venv/bin/python -m LLM sync-start LLM/runs/run_001 \
  --roi-a 实际x 实际y 实际宽 实际高 \
  --roi-b 实际x 实际y 实际宽 实际高
```

ROI 是各相机保存图像中的 LED 像素区域，中文占位符必须替换。成功后 `processed/sync.json` 包含：

- `clocks.camera_A/camera_B/esp`：三个时钟映射；不需要再手动执行 `clock`。
- `led.A/B`：光学边沿区间和确认帧；对应亮度轨迹另存 CSV。
- `device_start`：原始四个时间戳、日志行号、首次驱动应用相对 LED 的调用区间。
- `input_sha256`：包含原始串口日志哈希，日志修改后必须重新对齐。

只有一个明确事件时才自动关联；没有 READY/START、重复事件、时钟重置、日志损坏、两机不是各一个边沿时都会拒绝。发生失败不会沿用旧同步结果。需要检查实际 ROI 和原视频，不把其他光源变化当成 START。

随后按 README 建立 world、运行 pose 和 process。`pattern.json` 仍需保存实际选择的计划；本次只新增 START 的设备时间证据，**不代表已完成整个 ramp 或任意四路 schedule 的逐次 device-applied 记录**。

## 首次台架验收

- [ ] 新 flight 固件输出 READY，LED 在 START 前为暗。
- [ ] 原始双视频有足够暗帧，START 后两机均有清晰亮帧。
- [ ] 原始日志仅一次 START，四个时间戳顺序合理。
- [ ] `sync-start` 成功，人工对照视频/LED 亮度轨迹和设备事件。
- [ ] 用实际测量核对 LED、驱动边沿与电流的延迟/抖动；仅软件编译和合成测试不能勾选这一项。

电流仍是约 2 Hz 的滤波 CS 估计；这次改动没有增加电流采样带宽，也没有提供真实 current phase。
