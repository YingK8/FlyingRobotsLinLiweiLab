# LLM 工作状态

## 2026-10-28：GitHub 同步前验证

- 本轮重新运行完整 `LLM self-check`、`controller.camera.test_record` 和 `controller.calib.test_calibrate`，全部通过；flight / tilt 固件编译通过，未执行烧录或硬件操作。
- 发布范围包含 LLM notebook、实验面板、同步联调、无录像驱动、pattern 库，以及配套相机/标定/串口代码和当前线圈相位表。发布的 notebook 清除历史输出，`RUN_DRIVE_ONLY=False`；本地实验 notebook 与原始 runs 数据保留。
- 原始实验数据、图片及电流分析输出仍位于被忽略的 `LLM/runs/`，本次代码同步不包含这些数据；离线检查和固件编译不代表物理精度或电流 phase 已验证。

## 2026-10-28：取消 notebook 模型冷却等待

- 用户明确要求由本人判断线圈是否需要冷却。Block 7B/8 移除 `_thermal_gate`，不再依据估算温度延迟启动；保留 pattern 校验、READY、到时/异常/Interrupt 停止和共享 SerialComm 驱动时长记账。以下历史条目中的 thermal gate 描述不再适用于这两个入口，其他 controller 工作流不变。
- 7B 读取下拉框当前选项；7B/8 重新加载各自入口，避免 notebook 缓存旧等待逻辑。若旧 cell 正在等待，须先 Interrupt；已有实验输出与原始日志保留。
- 完整 `.venv/bin/python -m LLM self-check` 通过，新增断言防止两个入口重新调用模型等待；14 个 notebook 代码格编译通过。验证仅用离线数据与 mock，未启动实际相机、串口或线圈。

## 2026-10-28：与 alignment-rate 驱动对照

- 无录像 `drive_only/20261027_235833_a2a77e` 有 START、30 s ramp 和 OFF 回报；61 条 CS 中 A/C/D 全零、B 最大 0.03 A。用户报告两路电源各 10 V/10 A 限流，显示电流约 0.1 A 后下降；电源读数不等于 CS 或线圈 RMS。
- 对照 9 月 14 日归档 commit `863a6c00`，现用频率/commutation/carrier/序列路径没有修改；PwmController 新增的批量 setPhases 不用于本次 frequency-only 路径。flight 默认 CW、50% commutation、100% carrier、无 balance 与归档相同。未找到逐渐降低 carrier 的代码路径；不等于已验证实际引脚波形/驱动板。
- 历史 `results/tilt_sweep/20260914_112949_droneC/sweep.log` 在 50 Hz、四路 100% 时有 A=0.98/B=2.28/C=1.19/D=1.88 A 的 CS 记录；旧 A/C 零点已知有偏置，不能据此规定当前电源电流或保证实物条件一致。
- 新增只复现旧 50 Hz 实验 ramp 与切断前 hold 的可选 pattern（19.5 s），没有改变当前选项、固件、引脚或触发硬件。需用同一台架实测，根因仍未确定。

## 2026-10-27：无录像 frequency 测试

- 新增 notebook 可选 Block 7B 与 `drive_only.run_frequency_pattern`：复用 pattern、ramp、CommandLink 和 thermal，仅串口驱动，不需要相机/ROI。自动 READY→START→按时 STOP，显示并保存设备回报。仍点亮固件 LED，但不声称光学同步或实际电流正确。
- CommandLink 增加独立 `require_ready`，无录像检查不伪造 pre-roll、不放宽原 `sync_start` 门限。七项离线测试覆盖真实串口类的 mock READY、正常完成、无 READY、缺 START、固件拒绝、中断和 STOP 失败关闭。助手未运行实物驱动。
- 完整 `.venv/bin/python -m LLM self-check` 通过；notebook 全部代码格编译通过。实机输出与电流仍待用户台架测试。

## 2026-10-27：重复 READY 失败诊断

- 实际连续失败日志为空；USB 枚举可见不代表 ESP 已启动。根因尚未实测确认。
- sync_start 从固定等 1.5 秒改为 5 秒持续收取 READY；失败区分零字节与收到数据但无完整 READY，仍拒绝发送 ramp。新增独立 `LLM.serial_check`，仅 reset/接收并留日志。
- 完整 LLM self-check 通过，覆盖 1 秒/3 秒 READY 与 5 秒无数据拒绝；独立诊断入口用 mock 验证成功关闭和失败返回。未由助手打开实际串口或启动线圈。

## 2026-10-27：同步联调入口

- notebook 增加 MODE="sync_check"，运行 0→1→4→6→7→8→9；正式 measured 模式保留几何门限。联调模式绑定本 session 的试录与静态参考，保存白线照片但不声称已构建 z/原点。
- 输入曲线增加两机 LED 亮度，与 frequency、四路低频 CS 共三个面板。联调后处理明确不进入 pose/takeoff。仍沿用原 CommandLink、thermal 和停止路径；没有修改固件、GPIO 或驱动参数。
- 完整 LLM self-check 通过：新增联调准备/旧 session 拒绝、真实编码合成视频的 START 对时和三面板 PNG 解码验证；notebook 代码格编译通过。尚未连接相机/串口或运行实物 pattern。

## 2026-10-27：统一 notebook 工作流

- 已重排 record_experiment：ChArUco 完整标定 / 固定 rig 新图验证 / AprilTag+垂线定位 / 试录 / pattern 选择 / 正式实验 / 后处理。详见 WORKFLOW.md。
- 用户确认已刷 flight。新增 localhost 浏览器录制面板；开环公共 frequency 使用现有 CommandLink、ramp、SerialComm 和 thermal；主线程发送，按钮只设置事件。一次 run 一次 START，结束先 stop 再收尾视频。尚未实机执行。
- 配置、试录、rig、参考图与 session/hash 绑定；导入重新审核当前 take，修复旧 notebook 的 TESTED_CAMERA/REPORT 可串用问题。
- patterns/ 增加历史 DEFAULT 的命名副本，正式运行冻结计划；未知四路 duty/phase 写 NaN，不冒充执行/实测。
- AprilTag 自动识别/已知布局双目拟合已实现；CAD 有四角 block，但每个 ID 的具体面/朝向、tag 角点与 bearing CAD 坐标尚缺，模板保持 survey_confirmed=false。不会猜坐标。
- 后处理可先生成已对时 frequency/CS 输入曲线；缺物理点标注时明确停在 needs_pose_annotations。真实幅值/phase、自动轴姿态仍未完成。
- 离线验证：7 项新回归覆盖合成 ChArUco 验证、改变双目位姿拒绝、真实 tag 检测器识别合成 AprilTag 图、世界坐标、旧状态拒绝、未知通道和 mock START/停止/异常收尾；完整 LLM self-check 已通过；额外用合成 CS/START 时间映射生成输入曲线并实际解码 PNG，notebook 所有代码格编译通过。
- 图形界面真实浏览器验收未完成：本工具环境拒绝 localhost socket bind，已停止合成 UI 测试进程。按钮流程已用 mock 检查；相机、串口、固件及 Windows UI 均未实际运行。


## 2026-10-27：跨平台录像与逐格 notebook

- 新增 `record_experiment.ipynb`：环境/双视频编解码检查、逐台探测、双机 A/B 预览、10 秒试录验收、限时正式录像、报告和 LLM 导入；可选复用原标定入口。只用相机，无串口或线圈操作。
- 修正 Windows 默认后端为 DirectShow，允许显式 MSMF；Windows 探测不调用 system_profiler。编号需按实际画面确认，未实现 Windows USB 身份绑定。
- 相机请求 FPS 与视频播放 FPS 分别记录，`--fps` 已传入相机；原始时间戳不重造。LLM 录像不再默认通过重读丢弃时间差较大的配对，保留源帧供离线质量判断。
- 双机部分打开失败自动释放已打开设备；编码器支持 avc1/mp4v 回退；编码完成后才标记 finalized，最终编码错误写入元数据。Interrupt 正常收尾；采集错误、未完成编码会被 LLM 导入拒绝。
- `audit_flight` 完整解码核对两视频与 written 行数，分别报告实际保存帧率、帧间隔、host 时间差与播放器 FPS。PASS 仅代表文件/时间映射一致，警告及硬件同步边界保留。
- 已在本机 macOS 运行：LLM 全套 self-check（含 8 项跨平台离线回归）、原录像 self-check/test_record、标定 test_calibrate；全部通过。avc1/mp4v 均已合成双视频编码并实际解码。Windows 分支和设备失败场景使用 mock，未访问相机或串口。
- 待实机：Windows 同时两机的后端/模式/USB 带宽、编码吞吐、长录制与重复打开；Mac 当前双机回归；notebook 在目标 Jupyter 前端的 inline 预览、Interrupt 和标定操作验收。原生 C++ AVFoundation 路径未移植，旧 alignment rate 专用驱动脚本不属于此次 Windows 录像支持范围。

更新：2026-10-26。新增 LED 实际接线记录；flight START + Python runner 日志 tee + 离线 sync-start 已实现，硬件验证结果仍待记录。使用方法见 [START_SYNC.md](START_SYNC.md)。

操作者入口：[EXPERIMENT_CHECKLIST.md](EXPERIMENT_CHECKLIST.md)。新增逐步操作与现场进度表；目前硬件事项仍为待实测，清单创建不代表实验已完成。

2026-10-26 最新相机审核：用户确认两轮之间整体移动过 holder。setup_001 混合 31 pairs 后 FAIL（RMS 11.3381 px）；按真实批次隔离后，新一轮全部图片独立求内外参 PASS（9/13 pairs，RMS 0.4041 px）。已保存为 setup_002/calibration/stereo_rig.json，原图完整保留。后续使用 setup_002 的当前机位结果；B 姿态覆盖有警告，实物尺度、最终机位核验及世界坐标尚未完成。详见 [补拍审核](CALIBRATION_APPEND_REVIEW_2026-10-26.md)；[首轮审核](CALIBRATION_REVIEW_2026-10-26.md) 保留为历史。

## 已落地

- [x] 2026-10-26 当前实物 ChArUco 板采用 square=16.52 mm、marker=12.39 mm（用户最新选择，替代 12.22 mm；marker 按原设计比例推算，未独立复测）；标定 CLI 支持 `--marker-mm` 独立覆盖，第二批独立标定软件验收通过，实物尺度与世界坐标仍待核验。

- [x] 2026-10-26 用户确认 START LED 已接 D2 和 GND；详见 [当前 LED 接线](START_SYNC.md#当前-led-接线)。此项仅记录接线，亮灯与同步验证仍待完成。

- [x] 独立 LLM Python package、CLI、实验目录和元数据约定。
- [x] 复用双相机录制；严格导入实际编码帧与原始时间/掉帧映射。
- [x] 标定快照、camera A→重力坐标变换；从 bearing/垂线像素直接构造坐标。
- [x] LED ROI 检测、事件区间、单事件/多事件时钟映射。
- [x] 物理点与轴端点的双目三角化，位置/姿态独立质量门限。
- [x] 复用现有自动 disc 分割，标记为实验性中心代理，不伪造轴角。
- [x] common frequency + 四路 duty/phase 的 pattern 保存与采样。
- [x] 原始 serial 日志被动接收和 CS telemetry 解析；保留坏行诊断。
- [x] flight 的 seq=go 内生成 LED/首次驱动调用时间区间；IDLE LED 暗，GPIO map 不变。
- [x] Python flight runner 在现有 reader 内保存 JSONL；发送命令时收到的行也保留。
- [x] 对时模式要求固件 READY 和至少 0.25 s 已提交录制的源时间跨度，每 run 一次 START。
- [x] sync-start 自动关联两路单次光学边沿和设备事件；拒绝缺失/重复事件、重启和损坏日志。
- [x] takeoff 检测、共同裁剪窗口、两路视频预览和帧映射。
- [x] 曲线、CSV、quality、输入/代码哈希、报告导出。
- [x] 原始证据不覆盖，处理失败标记不完整，避免旧电流/旧预览混入新报告。

## 当前验证范围

2026-10-26：独立板尺寸参数验证通过（16.52/12.22 mm 正确传入求解入口；非法尺寸拒绝；未指定 marker 时保留比例缩放），LLM 全套离线 self-check 通过。未打开相机或进行实物标定。

2026-10-23：flight 与 tilt 固件编译通过（未 upload）；LLM 全套离线自检、SerialComm 自检和既有 test_panel 控制面板回归通过。新增测试覆盖 raw RX tee、发送时接收、预录制/单 START 限制、坏事件/重启拒绝、合成双视频边沿与 ESP 事件关联。没有实测延迟或拍摄硬件。

`python -m LLM self-check` 覆盖 LED 暗亮迟滞、时钟拟合、视频/时间戳对应、掉帧映射、合成三角化、错误几何拒绝、独立姿态有效性、垂线坐标、起飞短尖峰/缺口、日志损坏、ramp 曲线、完整导出与重跑。

合成 demo 有 240 个已知投影双目帧，位置和轴重建均可通过门限，检测到预期高度过阈值事件。它验证代码连接与已知几何，不能当作实拍轨迹/姿态精度，也没有验证自动 CV 在真实 robot 上的识别率。

已运行完整 self-check、既有 ramp 自检和 Python 编译检查；已查看 `runs/demo_v1/export/overview.png` 与预览首帧。两路预览各解码 133 帧，与各自 frame_map 一致；package 输入/处理文件哈希均匹配。未进行实拍验证。

## 下一次上机应完成的顺序

1. **静止测试**：固定两相机，完成匹配分辨率/旋转的标定，实测标定板尺寸；拍 bearing、垂线和 robot。验证三维原点与 z 方向、像素重投影，检查水平视角下轴端点是否可见。
2. **先验证时间基准**：更新后的 flight 在 IDLE LED 灭，在 seq=go 内记录 START。Python runner 对时模式已接入，需刷入新 flight 后台架确认。tilt 仍是旧步骤闪烁，尚未适配。
3. **测启动偏差**：用同一次记录观察 LED 光强（必要时光电传感器）、驱动边沿和 CS/实际电流，报告固定延迟、抖动和测量带宽。电流建立晚于 START 属于物理响应，应保留。用第二脉冲检查长视频漂移；单脉冲不能验证。
4. **验证真实日志**：Python runner 的唯一 reader tee 已实现，需实测完整保存。START 首次应用时间区间已经记录；整个 ramp 和后续四路 frequency/duty/phase 更新的完整 device-applied 快照仍未实现。
5. **真实 CV 验证**：对若干静止/旋转/起飞帧人工标注物理点，比较自动中心代理；优先解决轴方向可观测性，未通过前姿态角仍标无效。
6. **takeoff 验证**：以可见的 bearing 脱离为人工参考，选旋转中但尚未离开的 baseline；估计噪声、误检/漏检、onset 偏差，再确定门限。
7. **完整一次实验包**：实际 pattern、视频、电流、START 和 takeoff 都能互相追溯后交给师兄。

## 尚未实现的能力

- tilt / 独立 native runner 的 START 采集适配；固件结束第二光学脉冲与漂移验证。
- 根据在线检测到的 takeoff 立即改变输入的执行器；当前 only timed pattern，离线检测不等于在线控制。
- 任意四路 duty/phase schedule 的自动执行、设备 ACK 和实际应用记录；当前是保存/解析接口。
- 自动 shaft/axis 识别与稳定的两个姿态角；当前需要可见同一物理端点的标注/检测输入。
- 有符号高带宽电流采集与真实 current magnitude/phase 估计。
- 交互标注界面、自动 ROI 选择、LLM 调用、磁场模型拟合与 trajectory planning 优化。

## 本轮对现有项目的改动

- 新增 LLM/，实验输出忽略在 LLM/runs/。
- pyproject.toml 将 LLM 加入现有打包列表，没有新增第三方依赖。
- controller/control/ramp.py 新增离线 frequency_at，复用既有 ramp 语义，不更改当前驱动入口/默认参数。
- 保留已有 MATLAB、notebook、SPIFFS 和 results 的工作区改动。
- 2026-10-23：修改 flight START、共享 driveBoot 可选 LED 初态、Python runner/SerialComm；新增 LLM/start.py、sync-start CLI、回归测试和 START_SYNC.md。其余固件的默认 LED 行为保持原样。
