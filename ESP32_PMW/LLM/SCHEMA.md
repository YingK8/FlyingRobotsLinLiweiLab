# LLM v1 数据契约

## 文件布局

```text
run_001/
  manifest.json                  # source、版本、git、实验对象与初始条件
  config.json                    # LED / skew / takeoff / export 门限
  pattern.json                   # 计划输入；与实际执行独立
  calibration/
    stereo_rig.json              # 每次实验保存标定快照
    world_frame.json             # 重力/原点/正交基/测量来源
  raw/
    camera/A/A.mp4
    camera/B/B.mp4
    camera/frames.csv            # index,t_capture,skew_s,t_a,t_b,written
    camera/meta.json
    camera/import.json          # 原目录与导入哈希（导入已有 take 时）
    world_reference.json        # 可选：bearing、垂线对应像素
    annotations.csv             # 可选：物理点与轴端点
    telemetry.jsonl             # 可选：原始串口日志
    applied_commands.csv        # 可选：设备实际应用快照
  processed/
    sync.json                   # 时钟映射 + LED 区间 + 输入哈希
    led_A.csv / led_B.csv        # 每帧 ROI 平均亮度，保留源时间
    pose_camera.csv             # camera A 三维坐标与原始时间
    pose_camera.meta.json       # 标定/标注/视频哈希与门限
    pose.csv                    # gravity world + 实验时间
    command_planned.csv         # frequency,duty,phase 理想计划采样
    command_applied.csv         # 有实际执行数据时才生成
    current.csv / current.meta.json
    events.json                 # 起飞事件、门限、确认与裁剪区间
  export/
    report.md / overview.png
    pose.csv / command_planned.csv / current.csv
    command_applied.csv          # 可选
    camera_A.mp4 / camera_B.mp4  # 相同实验时间网格的观看预览
    camera_A_frame_map.csv / camera_B_frame_map.csv
    quality.json / package.json
```

JSON schema_version=1。源文件默认不覆盖，处理产物可重建。CSV NaN 表示未观测/无效，不能解释成零。

## 输入 pattern

`frequency_segments` 每段为 `[from_hz,to_hz,seconds,mode,k]`，复用 `controller.control.ramp`。mode=0 polynomial、1 ease、2 exponential，频率连续，总时长服从已有 ramp.check。只支持按时间变化，`trigger="time"`；takeoff 触发的在线分支明确拒绝。

`channel_updates` 从 t_s=0 开始，递增的完整四路快照，按零阶保持：

```json
{"t_s": 0, "duty_pct": [20,20,20,20], "phase_deg": [0,90,180,270]}
```

四路次序始终 A/B/C/D。duty 是 carrier duty 的计划值，单位 %。phase 是设定的 commutation phase，单位 deg；它不是实测电流相位。没有电流闭环标定之前，不提供一个虚假的 `target_current_A` 接口。

计划只定义到最后一段结束，**不会暗示该时刻线圈已经关闭**。当前 CLI 不发送这些输入，也不执行 channel_updates。

## 物理点像素标注

```csv
frame_index,ax_px,ay_px,bx_px,by_px,a_tip_x_px,a_tip_y_px,b_tip_x_px,b_tip_y_px
0,320,200,260,200,320,180,260,180
```

坐标单位 pixel，针对保存的视频图像。frame_index 是原 `frames.csv` 中的 index，掉帧后编号可以不连续。中心与 tip 必须分别在两相机中指向同一个物理点；tip 定义 robot 正轴方向。盘子的两个 ellipse center 未必对应同一物理点。

可省略整组四个 tip 列；缺失中心或 tip 用空白。程序输出所有已编码双目帧，即使没有对应标注，也会留下该帧及无效原因。

## 姿态与坐标

- `pose_camera.csv`：x_mm,y_mm,z_mm,nx,ny,nz 位于 calibrated camera A。
- `pose.csv`：同名数值位于 `coordinate_frame=experiment_world`；保留 source_coordinate_frame。
- `position_valid`、`orientation_valid` 分别控制三维位置与单位轴向量有效性。
- `alpha_x_deg=atan2(nx,nz)`、`alpha_y_deg=atan2(ny,nz)`；不观测绕自身轴的 spin/roll。
- 保留原 `t_source_a_s`、`t_source_b_s`，并添加 `t_s`、原始和对齐后的 stereo skew。
- `fit_rms_px`、ray_angle_deg、position_reason、orientation_reason、pose_method 记录质量与来源。

垂线给出的 z 必须是向上（bottom→top）。`R_camera_from_world` 的列为实验 x/y/z 在 camera A 的表示：

```text
p_world = R_camera_from_world.T @ (p_camera - origin_camera_mm)
n_world = R_camera_from_world.T @ n_camera
```

world frame 需要在相机、bearing、参考标记保持固定的实验间复用；它不会每次从 robot 初始姿态重新估计。

## 时间与 LED

`t_experiment = scale * t_source + offset_s`。相机 A/B 分别映射；单锚 scale=1 且 drift_measured=false。两锚以上拟合 affine；两点拟合零残差不证明同步精度。

LED edges 保存最后一帧 dark 到首帧 high 的 source-time 区间，以中点作对齐估计，同时另存连续帧确认的时间。这个区间不含未知曝光/USB 延迟，不能报告成严格的光学误差上界。

已有 recorder 记录 host read completion，不是传感器曝光时刻。两相机经过 LED 对时之后仍不是同时曝光；几何阶段和处理阶段都会拒绝过大的时间差，但真正的精度须根据运动速度实测。

2026-10-27 recorder 元数据增量（兼容旧文件）：`requested` 保存请求尺寸、FPS 与传输 FOURCC；`cameras` 保存驱动报告的尺寸、FPS、backend、FOURCC；这些均不等于实测帧率。`fps` 为 MP4 固定播放 FPS，`fps_measured` 为写入器收到的配对速率（含 queue drops）；实际保存帧率从 written=1 的时间戳计算。`codec` 为两视频使用的编码器，`capture_counters` 区分 grabber 已取得帧与被覆盖的未消费帧，不能当作传感器掉帧计数。

`encoder_finalized=false` 表示尚未完成/超时，导入拒绝；编码线程完成后保存最终 `encoder_errors`。`capture_error` 非空时拒绝导入，保留部分录像供检查。`capture_audit.json` 是可选的本地验收报告：`integrity=PASS` 只证明两路解码帧数和 written 行数一致，采样速率/帧间隔/host skew 另以数值和 warnings 报告，不证明曝光同步。

### Flight START/STOP v2

原始 `telemetry.jsonl` 可以包含：

```text
SYNC_READY v=1
SYNC_START v=2 led0=<us> led1=<us> drive0=<us> drive1=<us> on_us=<us> gap_us=<us>
SYNC_STOP v=2 t0=<us> t1=<us> on_us=<us> gap_us=<us>
```

整数都是设备单调微秒。`led0/led1` 包围 START 标记第一个边沿的 GPIO 写操作，`drive0/drive1` 包围首次 sequencer + carrier 应用，`t0/t1` 包围 STOP 标记第一个边沿；都不代表实测光学或电流起始。`on_us/gap_us` 是固件实际使用的脉冲与间隔宽度，离线容差由它推导。

`sync-start` 要求每份日志一个 READY、一个 START、至多一个 STOP，且两台视频各解出同一对脉冲边沿；日志有 STOP 而视频没有（或反之）视为录像被截断并拒绝。`sync.json.markers` 保存每台相机的解出结果（边沿时刻、实测 on/gap、容差），`cross_camera` 保存两台第一边沿之差与其上限，`device_start`/`device_stop` 保存原始时间、日志行号和 LED 中点参考。有 STOP 时相机时钟有两个锚点，`fit_clock` 估计比例，`drift_measured=true`；只有一个锚点时仍为 false。原始日志哈希纳入 `input_sha256`，修改后必须重跑。`v=1`（单个上升沿）的日志被拒绝，不降级为单锚点。仍为 `hardware_sync_validated=false`。具体采集步骤见 [START_SYNC.md](START_SYNC.md)。

## 电流原始日志

每行一个 JSON：

```json
{"host_receive_s":1234.56,"line":"t=12500 freq=180.0 | I[A]: A=0.32 B=0.30 C=0.31 D=0.33 | duty[%]: A=20 B=20 C=20 D=20 | trip=0"}
```

`t=` 为设备毫秒，解析成 t_device_s。host_receive_s 只记录何时收到数据，不作为设备应用时刻。坏行的行号/原因保存在 current.meta.json；不把上一条电流复制到下一帧。设备时间重置/重复需要切分实验，不能跨 reset 套一个映射。

当前字段 `current_A_A` 到 `current_D_A` 的末尾 A 是 Ampere，中间字母是通道；measurement_kind=`filtered_current_sense_estimate`。原固件约 2 Hz 输出，无法恢复驱动周期内的 current phase。未来有足够带宽的有符号采样时，另加 `i_raw(t)` 及幅值/相位提取链路，不能复用此字段声称已经测到。

## 实际执行快照（可选）

`raw/applied_commands.csv` 需要完整列：t_device_s、frequency_hz、duty_A_pct…duty_D_pct、phase_A_deg…phase_D_deg、provenance=`device_applied`。只放已经实际应用并有设备时间的快照；firmware 当前尚未产生这个新日志。

原始 host-send 日志可以额外保存在 raw 中，但不能复制到这个文件。没有 ESP→实验时间锚时不产生 t_s。

裁剪时会额外保留窗口前最后一条已观测快照，标记 context_before_window=true，并保留原始应用时刻。它提供历史上下文，不代表系统已证明两条快照之间没有其他更新。

## 起飞与裁剪

baseline 的 z 中位数与 1.4826×MAD 估计噪声，阈值取 max(min_height_mm, noise_sigma_factor×sigma)。连续有效帧保持超过阈值，且上升速度/趋势满足门限后确认。

`onset_s` 是首个过阈值观测，`onset_interval_s` 保留与之前一帧的区间；`confirmed_s` 是满足持续时间后的确认。它们不是精确的物理脱离时刻。

状态为 detected、no_takeoff 或 uncertain。无效观测/长缺口阻断候选。只有 detected 才按起飞裁剪，否则保留共同可用区间。

两个预览视频都使用相同 60 Hz 实验时间网格，选最近源帧，超过最大偏差时显示黑帧/MISSING。frame_map 保留重复帧、源索引、原实验时间、相差秒数；预览播放的第 0 秒是裁剪窗口开始，不是实验零点。

`package.json` 保存输入、处理产物、代码的 SHA-256。给下游模型时同时提供 source、quality 和事件说明，不只给曲线图片。

## 统一工作流增量（2026-10-27）

- session.json 固定 session_id 与 camera config。录像 meta 保存同一快照；trial_audit / capture_audit 的 input_sha256 绑定实际文件。正式导入不接受传入的旧报告，重新解码审核当前 take。
- calibration/verification.json 是固定 K/dist/R/t 的新图检验。calibration/evidence/ 保存验证原图、装置参考图、布局副本；对应 *_evidence.json 映射原路径与包内路径。
- fixture_layout.json 中四角是 decoded canonical TL/TR/BR/BL，含黑边；不能按图像左上排序。bearing_center_fixture_mm 与所有 tag 都属于同一个 fixture frame。world evidence 保存 T_camera_A_from_fixture、双目垂线像素与输入哈希。
- frequency-only pattern 使用 channel_settings_status=firmware_defaults_unrecorded 和 channel_updates=[]；sample_pattern 输出未知 duty/phase 为 NaN。此状态不表示零 duty 或零 phase。只有这一模式可由新实验面板执行；旧四路计划仍为离线输入。
- raw/host_commands.log 记录 host-send；raw/telemetry.jsonl 为唯一 reader tee；不能改名为 applied_commands。host_start_send_complete_s / host_scheduled_stop_send_complete_s 只是 host 发送完成时刻。
- export/input_curves.png 与 input_curves.meta.json 可在 pose 前生成，status=inputs_only_not_full_pipeline；不是完整轨迹/起飞报告。
