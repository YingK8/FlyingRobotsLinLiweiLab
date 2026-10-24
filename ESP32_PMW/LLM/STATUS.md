# LLM 工作状态

更新：2026-10-23。新增 flight START + Python runner 日志 tee + 离线 sync-start；未使用相机、串口或线圈，未烧录固件。使用方法见 [START_SYNC.md](START_SYNC.md)。

操作者入口：[EXPERIMENT_CHECKLIST.md](EXPERIMENT_CHECKLIST.md)。新增逐步操作与现场进度表；目前硬件事项仍为待实测，清单创建不代表实验已完成。

## 已落地

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
