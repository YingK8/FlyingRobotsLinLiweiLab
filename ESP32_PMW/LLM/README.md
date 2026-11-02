# LLM：双目视频 + 线圈电流 + LED 对时的飞行记录

运行 `record_experiment.ipynb`，一格一步。每一步都是 `Bench`（`bench.py`）的一个方法：

```python
bench = Bench(cameras=(1, 0), serial_port="/dev/cu.usbserial-0001")
bench.preview_cameras()        # 遮挡一个镜头确认 A/B；对焦；之后不再动镜头
bench.calibrate()              # collect_charuco_frames -> calibrate_intrinsics -> calibrate_extrinsics
bench.use_calibration(path)    #   ……或复用已有 stereo_rig.json
bench.verify_calibration()     # 最终机位拍 >=5 个新的静止板子姿态；须 PASS
bench.photograph_reference()   # 移走板子；LED 与垂线两机都要看得到
bench.find_holder_marks()      # 可选：holder 四象限交点；填 holder_regions.json 后再跑一次命名
bench.locate_fixture(plumb)    # 可选：已命名交点 + bearing + 垂线 -> 世界原点与重力
bench.record()                 # 10 秒纯相机试录，自动审核
bench.run(pattern, note="...") # Record -> START -> 自动 stop。会驱动线圈
bench.analyse()                # LED 对时 -> export/input_curves.png；给 annotations= 时再出 XYZ/起飞
```

只有标定已验证 **且** 装置已定位时，`run()` 才做 **正式实验**；否则做 **sync check**（视频 + 电流 + LED，无几何）。
自检：`.venv/bin/python -m LLM self-check`（离线，不碰硬件）。

## 目录

| 位置 | 内容 |
|---|---|
| `record_experiment.ipynb` | 主 notebook，一格一个 `bench` 步骤 |
| `bench.py` | `Bench`：把下面各模块按实验顺序串起来 |
| `record/` | 实验前与实验中：`experiment`（session/run 准备、导入、后处理）、`panel`（浏览器录制面板）、`takes`（录像校验/导入）、`pattern`、`telemetry`（串口日志解析）、`drive_only`、`serial_check`、`io`（run 目录/JSON/哈希） |
| `analyse/` | 实验后：`led`（LED 边沿与标记解码）、`start`（START/STOP 关联、`find_led`）、`rig`（标定验证、装置定位）、`pose`（双目三角化）、`world`（重力坐标）、`pipeline`（导出）、`takeoff`（起飞事件）、`holder*`（holder 四象限标记检测与定位，见 `HOLDER_MARKERS.md`）、`holder_layout.json`（v3 CAD 20 点）、`coil_base.step`（底座 CAD，与标记同坐标系待核对）、`fixture_layout.json`（旧纸质 AprilTag） |
| `patterns/` `tests/` | 输入 pattern、离线自检 |

## 什么时候重标
- **完整 `calibrate()`**：首次使用、重新对焦、单台相机移动、支架变形、换采集模式。
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

## Pattern（`patterns/*.json`）
- `frequency_segments`：每行 `[from_hz, to_hz, seconds, mode, k]`。mode 0 = poly、1 = ease、2 = exp，由 `controller/control/ramp.py` 校验。
- 只执行 frequency pattern（`channel_updates: []`）。到时主机发 stop，再录 2 秒。
- `rise_2to210Hz_30s_ease_k2` 是旧的 `ramp.DEFAULT`。
- `alignment_2to5_5to50Hz_hold10s_19p5s` 复现 2026-09-14 的 50 Hz 电流对照。
- `rise_2to190Hz_30s_linear_hold10s`（40 s）、`rise_2to20Hz_5s_20to190Hz_20s_linear_hold10s`（35 s）：2026-10-28 新增，未经台架验证。

## run 目录
```
pattern.json manifest.json config.json
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
