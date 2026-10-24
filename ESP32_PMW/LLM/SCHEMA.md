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

### Flight START v1

原始 `telemetry.jsonl` 可以包含 `SYNC_READY v=1` 与 `SYNC_START v=1 led0=<us> led1=<us> drive0=<us> drive1=<us>`。四个整数是设备单调微秒，顺序为 LED GPIO 写入前/后、首次 sequencer + carrier 应用前/后；不代表实测光学或电流起始。

`sync-start` 仅接受每份日志一个 READY 和一个 START、每路视频一个光学上升沿。`sync.json.device_start` 保存原始时间、日志行号、LED 中点参考和 `drive_apply_interval_t_s`；`clocks.esp` 从该设备参考映射到实验零点。原始日志哈希纳入 `input_sha256`，修改后必须重跑。仍为 `hardware_sync_validated=false`，单事件 `drift_measured=false`。具体采集步骤见 [START_SYNC.md](START_SYNC.md)。

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
