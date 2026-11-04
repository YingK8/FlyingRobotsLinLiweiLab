# LLM：双目视频 + 线圈电流 + LED 对时的飞行记录

运行 `record_experiment.ipynb`：导入、设置各一格，之后每一步一格（可单独重跑）。

1. **准备**（不驱动线圈）：`Bench(...)` → `preview_cameras`（对准、对焦、补光）→ 标定（`collect_charuco_frames` → `calibrate_intrinsics` → `calibrate_extrinsics`，或 `use_calibration(path)`）→ `verify_calibration` → `photograph_reference` →（给了 `PLUMB` 时）`find_holder_marks` + `locate_fixture` → `record`（试录）。
2. **实验**（会驱动线圈）：`show_ramp(RAMP)` → `run(RAMP, note=...)` → `analyse()`。

每一步都是 `Bench`（`bench.py`）的一个方法，可单独重跑。另有 `check_serial()`（只握手）和 `drive_without_cameras(RAMP)`（无录像驱动）。

只有标定已验证 **且** 装置已定位时，`run()` 才做 **正式实验**；否则做 **sync check**（视频 + 电流 + LED，无几何）。
自检：`.venv/bin/python -m LLM.tests.run`（离线，不碰硬件）。

## 目录

| 位置 | 内容 |
|---|---|
| `record_experiment.ipynb` | 主 notebook：准备、实验两格 |
| `bench.py` | `Bench`：把下面各模块按实验顺序串起来 |
| `record/` | 实验前与实验中：`experiment`（session/run 准备、导入、后处理）、`panel`（浏览器录制面板）、`takes`（录像校验/导入）、`ramps`（读取/校验 ramp 文件）、`telemetry`（串口日志解析）、`drive_only`、`serial_check`、`io`（run 目录/JSON/哈希） |
| `analyse/` | 实验后：`led`（LED 边沿与标记解码）、`start`（START/STOP 关联、`find_led`）、`rig`（标定验证、装置定位）、`pose`（双目三角化）、`world`（重力坐标）、`pipeline`（导出）、`takeoff`（起飞事件）、`spinup`（转子是否转起来）、`holder*`（holder 四象限标记检测与定位，见 `HOLDER_MARKERS.md`）、`holder_layout.json`（v3 CAD 20 点）、`coil_base.step`（底座 CAD，与标记同坐标系待核对）、`fixture_layout.json`（旧纸质 AprilTag） |
| `ramps/` | 每个 run 驱动的线圈频率曲线（见下） |
| `tests/` | 离线自检 |
| `thermal/` | `coil_thermal.py`：线圈温度模型（无传感器）。flight 路径缺它就拒绝驱动；`.last_energised` 是本机状态，不入库 |
| `design/` | 离线线圈阵列 / 三维 plant / MPC 模型，`matlab/` 为原始仿真（原 `ai/design`，见 `controller/control/theory.md` §12–15） |

## 什么时候重标
- **完整标定**（`CALIBRATION = None`）：首次使用、重新对焦、单台相机移动、支架变形、换采集模式。
- **只做 `verify_calibration()`**：两台相机随刚性支架整体移动。PASS 即可沿用旧 rig，然后重拍参考照片。

标定板：`bench.BOARD`，9x6 letter，卡尺量得方格 16.52 mm。采集会自动拍静止姿态，按 q 结束。FAIL 时看 `diagnostics/`，不放宽门限。

## 对时原理
- **START 前**：reset 后 flight 输出 `SYNC_READY v=1`，LED（D2/GPIO2）保持灭。
- **START**：`seq=go` 启动驱动并发出 LED START 标记：亮 30 ms、灭 30 ms、再亮。
  固件输出 `SYNC_START v=2 led0 led1 drive0 drive1 on_us gap_us`（ESP 微秒时间）。
- **STOP**：停止时发出镜像标记（灭-亮-灭）和 `SYNC_STOP v=2`。
- **分析**：`analyse()` 在 START 附近找两机中的 LED（`start.find_led`），解出两种标记，
  把相机 A、相机 B 和 ESP 映射到同一实验时钟，t=0 在 LED。有 STOP 标记时还能测出时钟漂移。
- **局限**：视频时间是主机读取完成时间，不是曝光时刻；LED 到电流的延迟未测。
  CS 约 2 Hz，记录在 `raw/telemetry.jsonl`。计划、主机发送、设备执行、实测四者分开保存。
- **每个 run 只有一次 START**：重拍就是新的 `run()`。录像中不要按 ESP reset。

## 安全
- 没有任何东西会自动停线圈。停机手段：面板 Stop、Interrupt、`safe_off.py`、GPIO14、关电源。
- `run()` 和 `drive_without_cameras()` 不按温度模型等待冷却（操作者决定，2026-10-28），驱动时长仍记账。
- ESP 每次 reset（烧录、打开串口）时保持线圈电源关闭。

## Ramp（`ramps/*.json`）
一个 ramp 就是一次 run 的线圈频率曲线，按段顺序执行，段与段必须首尾相接：

```json
{"note": "为什么有这个 ramp",
 "segments": [{"from_hz": 2, "to_hz": 50, "seconds": 5, "shape": "ease", "k": 2},
              {"from_hz": 50, "to_hz": 50, "seconds": 10, "shape": "linear"}]}
```
- `shape`：`linear`、`ease`（k≥1）、`poly`（f ∝ tᵏ）、`exp`；`k` 缺省 1。秒数精确到毫秒，总长 ≤55 s（`controller/control/ramp.py` 校验，拒绝而不截断）。
- 只控制频率，各通道 duty/phase 用固件默认值。到时主机发 stop，再录 2 秒。
- 现有：`alignment_50Hz`（复现 2026-09-14 的 50 Hz 对照）、`linear_to_190Hz_hold`（40 s）、`two_step_linear_to_190Hz_hold`（35 s）、`linear_to_250Hz`（30 s）。都未经台架验证。
- 注意：转子起转需要约 3 s 停留在 5 Hz 以下；三个 linear ramp 都只有不到 1 s。

## run 目录
```
ramp.json manifest.json config.json   （2026-11-02 前的 run 是 pattern.json，仍可分析）
calibration/  stereo_rig.json, world_frame.json, verification.json
raw/          camera/{A/A.mp4, B/B.mp4, frames.csv, meta.json}, telemetry.jsonl, host_commands.log
processed/    sync.json, led_A.csv, led_B.csv, current.csv, command_planned.csv, pose*.csv, events.json
export/       input_curves.png, overview.png, pose.csv, current.csv, camera_A/B.mp4 + frame maps
```
姿态标注（`analyse(annotations=csv)`）的列：
`frame_index,ax_px,ay_px,bx_px,by_px[,a_tip_x_px,a_tip_y_px,b_tip_x_px,b_tip_y_px]`。
像素坐标对应保存的视频，每组 A/B 必须是同一个物理点；tip 标出 robot 的正轴方向。
NaN 表示未观测，不是零。

## 待办
- `holder_layout.json` 的 bearing 实测位置未填（`survey_confirmed=false`）；holder 检测只在 CAD 渲染图上验证过，实拍识别率与定位精度待测。
- `fixture_layout.json`（旧 AprilTag）仍缺 CAD tag 角点和 bearing。
- 还没有双脉冲固件的实录：`find_led`、光学延迟、LED 可见性都未经台架验证。
