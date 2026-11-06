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
| `viser_app.ipynb` / `app.py` | 同一流程的 viser 界面：两格启动，每步一个按钮（未满足前置时变灰）；标定用留出帧自检，平台由 10 s 静止画面拟合、绳为 z 轴，运行中不更新；含 LED 测试、录像实验、丢弃/继续、回放 |
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
- 录像运行到 ramp 时长会由主机发送 stop；Stop 或 Interrupt 也会请求停止并等待 OFF 回应。USB 断开、进程强杀或主机失效时不能保证停机，独立停机手段仍包括 GPIO14 和关电源。
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
- 光学对时与几何有效性按每次 run 的处理报告判断；完成录像不代表完成同步验证。

## 本机 notebook：1280×800 @ 120 fps 与 axis_pose

2026-11-04 合入 `fc87a76` 后，本机 notebook 恢复 A/B=(1,0)、1280×800、120 fps、两机默认不旋转（可在预览格修改）。本机默认自动曝光（EXPOSURE_MS=None），每个相机入口显式切回自动；填写毫秒数才启用新版手动曝光/增益，实际帧率以试录源时间为准。`Bench` 库默认值未改，notebook 显式传入本机设置。

发布的 notebook 默认 `SESSION_ID=None`、`CALIBRATION=None`，不依赖本机实验目录；需要恢复时显式填写自己的 session 和标定路径。先遮挡镜头核对 A/B，保持镜头/对焦、成像模式和双目相对几何一致；标定复用以误差证据和适用范围判断。整体移动刚性双目装置时，相对于实验台/重力的坐标关系仍需更新。`PLUMB=None` 时可先录 sync check 并尝试 axis_pose。

顺序：导入/设置 → 0 建立 bench → 1 预览 → 2 加载或重标 → 3 验证 → 4 参考照片 → 6 试录 → 可选 6A 空背景 → 7 ramp → 8 录像实验 → 9 LED/电流分析 → 可选 9A axis_pose。暂不做 world 时跳过 5。填写本次 NOTE；试录有 warnings 时检查并填写本次 WARNING_REVIEW。逐格运行，不要 Run All。

axis_pose 适配层保留在 `LLM/axis_pose.py`，导入已改为 `record/`、`analyse/`；核心 `controller.pose.axis_pose` 不变。6A 只打开相机；9A 只处理已保存录像，默认均关闭。9A 填两机 robot ROI、空背景及确认项后设 RUN_AXIS_POSE=True，先检查 500 帧，再将 AXIS_MAX_FRAMES=None 处理全片。新建 bench 清空背景选择。

## 实验性 axis_pose diagnostics（2026-11-02）

`processed/axis_pose_trials/<timestamp_uuid>/` 每次运行创建新目录；不替换 `pose_camera.csv`、`pose.csv` 或正式 report。

- `axis_pose.csv`：一行对应一对实际编码帧，保留 `frame_index`、`encoded_frame_index`、`t_source_a_s` / `t_source_b_s`；`elapsed_capture_s` 从第一对保存帧的平均 host 时间开始，不是 LED START 时间。
- `x_est_mm,y_est_mm,z_est_mm` 是 camera A 坐标下的区域质心位置代理；`nx_est,ny_est,nz_est` 为同一坐标下的候选单位轴。`angle_to_reference_deg` 相对显式给定的 `reference_axis_camera`，不能叫重力倾角。正负号／分支来自参考和连续性，`axis_sign_observed=false`。
- `axis_estimate_valid` / `position_estimate_valid` 只表示通过本次算法门限；`position_valid=false`、`orientation_valid=false`、`physical_accuracy_validated=false`。不伪造轴端点，不交给 takeoff 或正式 world 转换。缺失／拒绝帧的估计值为 NaN。
- `reason`、每机 `a_reason/b_reason`、`a_area_px/b_area_px`、`ray_gap_mm`、`stereo_skew_s`、`reference_reset` 记录诊断；失败或时间缺口后重置选支参考，绝不把旧轴写成当前观测。
- `metadata.json` 保存算法／输入哈希、ROI、背景模式、时间／角度门限、显式参考、处理帧数与接受数；`inputs/` 保存本次 rig 和空背景快照。门限沿用核心或工程起点，不是实物精度。
- `curves.png` 和 `overlays.jpg` 供人工检查；叠加图保留源帧编号和源时间，绿色为 mask、黄色为区域短主轴。最多均匀抽取 6 对图像，不能替代全视频检查。
- notebook Block 6A 只在显式开关下打开相机保存背景；9A 只读完成的录像，兼容 `sync_check`。两格均默认关闭，不驱动硬件。无背景时需显式 `background_mode='threshold'`，适用性需按实拍确认。

## 单次无录像试转：2→210 Hz

`ease_to_210Hz` 是明确的 2→210 Hz、30 s EASE（k=2），没有末端保持；主机到时发送 stop。运行 `bench.drive_without_cameras('ease_to_210Hz', confirm_start=True)`，不需要相机、标定、ROI 或试录。关闭其他串口使用者；先关线圈电源运行该格，收到 READY 后按提示开电源、输入 START，才会发送 ramp。其他输入取消；Interrupt 尝试 stop。此等待是操作者的启动选择，不是温度模型冷却 gate，驱动时间仍记账。USB 断开或进程强杀仍不能保证停止。

旧 `rise_2to210Hz_30s_ease_k2.json` 在合并前提交 ed7fcb2 中的内容其实为 2→250 Hz / 30 s / linear，因此本次没有直接复用该历史文件。新文件已校验实际编码为 `seq=ramp:2:210:30000:1:2`；未上机验证。

## Block 8 自动录像与频率输出

运行 Block 8 会加载当前 panel 代码，等待 READY 后开始录像；录像约 5 秒后发送 LED 标记和 ramp，每次运行生成新的 run。Interrupt 请求 stop、等待 OFF 并保存已录视频，重复运行无需重建 bench。设置格修改 RAMP 或 NOTE 后需重新执行该格。

每秒输出一次 ESP 回报频率；超过 2 秒无新频率时标为陈旧回报。RAMP/HOLD 阶段由主机时间和计划估计，不能当成固件实际执行确认。频率回报不是 robot 转速。

LED 识别前导码目前会阻塞主循环约 0.8 秒；可能在保存帧时间戳中产生间隔，必须以 `frames.csv` 检查，不将固定 120 FPS 播放速度当作实际采样时钟。

离线检查：`.venv/bin/python -m LLM.tests.run`。录像、标定原图、上传打包与本机备份保留在忽略目录中；提交 notebook 时移除输出和个人 session 绑定。
