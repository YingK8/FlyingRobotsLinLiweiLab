# LLM 实验操作清单

2026-10-27 新增 Mac / Windows 共用的[逐格录像 notebook](record_experiment.ipynb)：先做环境检查、双机预览和 10 秒录像验收，再正式采集。它复用下面的录制/标定接口，不替代物理参考、START 或轨迹验收。下文保留原 Mac 操作记录；Windows 不照抄旧索引或 `.venv/bin/python` 路径。

更新：2026-10-26。面向第一次上机采集；方框表示需要实际完成的事项，不能把已有代码或合成 demo 当作实测完成。Flight START 的代码与用法见 [START_SYNC.md](START_SYNC.md)，尚未进行硬件验收。

当前操作者已确认双相机预览可用，实物标签 A 对应当前索引 1，B 对应当前索引 0，因此本次使用 `--indices 1 0`。拔插后重新核对实物标签与索引。LED 接线已记录在 START_SYNC.md；相机标定阶段不需要启动线圈、LED 或使用垂线。

**你现在先做第 1–4 步：看清双相机画面 → 固定机位和标定 → 拍静态重力参考 → 建立实验坐标。** 第一阶段的成果是一组可追溯的静态参考数据，不需要启动线圈，也不需要等电流 phase 采集做好。

已有 [README.md](README.md) 说明完整接口，[STATUS.md](STATUS.md) 记录软件进度，[SCHEMA.md](SCHEMA.md) 定义数据格式。本文件负责说明操作者先做什么、每一步完成后留下什么。

## 0. 准备物品和记录位置

- [ ] 两台 ELP 相机、稳定的支架和 USB 连接。
- [ ] 足够的照明，背景与 robot、绳子有明显的明暗反差；相机是黑白的。
- [ ] 平整的 ChArUco 标定板，以及用于测实际格距的尺或卡尺。
- [ ] 铅垂锤、悬挂支架、干净背景。当前讨论的中心出线、尖端明确的单只垂锤可以先试；是否已购买/到货：________。
- [ ] bearing 上一个两台相机都能定位、以后不会改变的实体参考点。
- [ ] 后续使用的 START LED，以及 PCB/current sensing 的接线资料；它们不阻塞第 1–4 步。

先进入项目目录。下文命令均在此目录运行：

```bash
cd "/Users/yyc/Documents/LAB/Lin_Lab/Micro Drone/FlyingRobotsLinLiweiLab/ESP32_PMW"
```

本次清单统一使用 **1280×800**，与现有标定采集入口一致。现有旧 rig 不能因为文件存在就当作当前机位的标定；LLM 也不会自动把不同分辨率的标定转换过来。

含 `实际A索引`、`实测单格毫米` 等中文占位符的命令，要填入真实值后执行。首次用 `setup_001`；已有数据时用 `setup_002` 等新名称，保留旧数据。

## 1. 只打开两个相机，确认画面

```bash
.venv/bin/python -m controller.camera.record --indices 1 0 --mode 1280x800
```

按当前 A/B 标签顺序并排预览。点击预览窗口后，空格开始/停止录像，`q` 退出。现在可以只看画面。

- [ ] 两台相机都能看到 bearing、robot 及预期飞行范围。
- [ ] 确认焦点和照明：robot 的可追踪位置、轴端点是否看得清，画面是否过暗、过曝或模糊。
- [ ] 分别遮挡一台镜头，确认画面 A/B 对应的物理相机，给支架和线缆贴 A/B 标签。
- [ ] 调好后固定相机、镜头焦点和 bearing，并拍一张布置记录。

默认画面旋转 180°，适配倒装。若要取消，使用 `--no-flip`；之后的标定和录像也必须全部使用同一选择。

退出预览后，可以查询本次连接的索引：

```bash
.venv/bin/python -m controller.camera.identify
```

记录：A 索引 ______；B 索引 ______；分辨率 1280×800；rotate180 为 ______。索引是当前连接的结果，拔插后重新确认，不能长期假定一定是 0、1。

**完成标准：两路画面稳定，A/B 身份明确，机位与焦点固定。** 同时只开一个相机程序；预览退出后再做标定。完整的 `controller/run.ipynb` 含驱动步骤，纯预览时不要 Run All。

## 2. 为当前机位做相机标定

两台相机已刚性固定在同一个 holder 时，可以先在便于摆板的位置标定，再整体搬到实验位置。先按预计工作距离调好 W/T 变焦、N/∞ 对焦和光圈，锁定相机相对位姿。整体搬动后核对标定；在最终位置才建立 bearing/重力坐标。单独转动相机、改变间距、变焦或对焦后需重新评估并通常重做标定。

### 2.1 建立静态参考 run，准备标定板

```bash
.venv/bin/python -m LLM init LLM/runs/setup_001
```

填写该目录 `manifest.json` 的 `initial_conditions`：robot、coil setup、bearing 的标识，以及机位、A/B、分辨率和旋转选择。尚未知的项目保留未知，不填写假值。

- [ ] 选择与实物一致的板型。现有 PDF 可直接打印，不需要另行下载随机板型：

  | PDF | 图案设计尺寸（不含纸张留白） | `--board` |
  |---|---|---|
  | [较大 9×6 板](../controller/calib/assets/charuco_9x6_letter.pdf) | 约 150×100 mm，单格约 16.67 mm | `9x6_letter` |
  | [小格 9×6 板](../controller/calib/assets/charuco_9x6_6mm.pdf) | 约 54×36 mm，单格约 6 mm | `9x6_6mm` |
  | [窄长 4×12 板](../controller/calib/assets/charuco_4x12_6mm.pdf) | 约 24×72 mm，单格约 6 mm | `4x12_6mm` |

  可先用较大 9×6 板，前提是在预计工作距离能放入两台相机的共同视野，并能改变倾角。视野较窄时使用小格版；不要为了把板塞进画面，在已经选好实验焦距后随意改变 W/T。这里是 ChArUco 棋盘，不能用装置上的零散 AprilTag 直接替代。
- [ ] 新打印时选 Letter 纸、实际尺寸/100%，关闭 Fit to page/缩放适应；打印 PDF 原件，不打印预览截图。保留图案周围白边，平整贴在硬板上，避免褶皱、翘曲和覆盖图案的反光胶带。
- [ ] 测实际单格边长：9×6 板可量棋盘图案外边界宽 W、高 H（不含白边），分别计算 W/9、H/6。两者应在测量误差内相符；明显不符时检查测量、打印拉伸和贴板变形，不靠取平均掩盖。单格是棋盘格，不是格内 marker。把板型、原始测量和换算值写入 manifest 的 notes。

2026-10-26：已渲染检查较大和小格 9×6 PDF，并用现有板型检测器在渲染图上各检出 40/40 ChArUco 角点、27 个 marker。这只验证文件与程序匹配，不代表实物打印尺寸或相机标定已通过。

**不要照抄历史的 5.95 mm，也不要只相信文件名中的 6 mm。** 当前 JSON 的 `square_mm_measured` 标志不能代替你对这张实物板的测量记录。

### 2.2 采集并求解

当前进度（2026-10-26）：下面 setup_001 示例保留为首次采集流程；两轮搬动前后图片已分离，当前通过的软件标定在 setup_002。继续当前实验时使用 setup_002，详见 [第二轮审核](CALIBRATION_APPEND_REVIEW_2026-10-26.md)，不要向混合批 setup_001 继续追加。

2026-10-26 当前实物板：9×6 ChArUco、DICT_4X4_100；square 使用背面标注的 16.52 mm，marker 按用户最新选择使用 12.39 mm。这是按原设计比例计算的 12.5 × 16.52 / 16.667 ≈ 12.39 mm，假设实物采用同一设计且等比例缩放，未独立复测；替代此前选择的 12.22 mm。下面命令明确指定两项尺寸；换板时应重新填写，不能沿用这组数值。激活 `.venv` 后可把 `.venv/bin/python` 写成 `python`。

```bash
.venv/bin/python -m controller.calib.calibrate \
  --bag LLM/runs/setup_001/raw/calibration_images \
  --rig LLM/runs/setup_001/calibration/stereo_rig.json \
  --out LLM/runs/setup_001/calibration/diagnostics \
  --indices 1 0 \
  --board 9x6_letter \
  --square-mm 16.52 \
  --marker-mm 12.39
```

若第 1 步选了不旋转，这里加 `--no-flip`。本入口默认采集 1280×800，不需要添加 `--mode`。

1. 将标定板放在支撑物上，停稳后让程序自动保存合格照片。
2. 改变板的位置、深度和倾角，覆盖 robot 将活动的区域；保证有足够多两台相机同时看见的姿态。
3. 看界面提示补缺少的角度和区域。不要拿着板一直晃动，也不要只在一个平面内平移。
4. 空格是暂停/继续自动拍照，`q` 结束采集并开始求解。这里空格的作用与录像不同。

首次可采集约 20–30 个不同且合格的双目姿态作为操作目标，不是程序保证通过的数量。覆盖中央、左右、上下，加入近/中/远距离和左右、上下倾斜；每次摆放后等 saved pair 数增加再换姿态。一直不增加时查看 waiting on 的提示，不要只在同一姿态等待。最终以验收结果和误差/覆盖检查为准。

- [ ] 求解显示 `PASS overall`，写出本次 `stereo_rig.json`。
- [ ] 查看 diagnostics 中的覆盖范围、重投影和去畸变图；检查是否存在明显异常。
- [ ] 后续用一个实测距离做三维尺度核对；低像素残差本身不能证明毫米尺度正确。

已有 bag 默认会复用旧照片。需要重拍时使用新 setup 名称，不使用 `--override` 覆盖证据；同一布置补照片才考虑 `--append`。失败时按提示改善采集，不靠放宽门限通过。

**留下：标定照片、测量记录、diagnostics、stereo_rig.json。** 此后移动相机或调整镜头焦点，都要重新评估标定；机位变化需要重建重力参考。

## 3. 拍一组静态重力参考

### 3.1 挂好垂锤

- [ ] 在线圈关闭时操作，并让钢制垂锤远离 robot 的永磁体，避免横向磁力使其偏斜。
- [ ] 固定悬挂端，垂锤完全悬空，不碰桌面、线圈或支架；等摆动停止，避开气流。
- [ ] 两台相机同时拍到一段拉直的绳子、垂锤本体和 bearing 参考点。
- [ ] 初次采用本体两点：顶部中心出线点为 `plumb_top`，底部尖端为 `plumb_bottom`。检查本体轴线与直绳方向是否一致；偏心或尖端不正时不要用本体轴定义重力。

本体两点通常比细线容易手动定位。若上端看不清，可用清晰的上方悬挂点与下方尖端。不要把任意轮廓角点、反光亮点或两幅图中随便选的线段端点当作同一个物理点。

### 3.2 录 5–10 秒并导入

```bash
.venv/bin/python -m controller.camera.record \
  --out LLM/runs/setup_001/raw/captures \
  --indices 实际A索引 实际B索引 \
  --mode 1280x800 \
  --note "static world reference; coils off; stationary plumb bob and bearing"
```

旋转选择保持一致。空格开始，静止录 5–10 秒，再按空格停止，最后 `q` 退出。记下程序打印的实际录像目录。

```bash
.venv/bin/python -m LLM import-flight LLM/runs/setup_001 \
  "LLM/runs/setup_001/raw/captures/实际录像目录名"
```

- [ ] 导入成功，两个视频能完整解码。
- [ ] 保留 `frames.csv` 和 `meta.json`；不要只保留视频或截图。
- [ ] 在原始帧上看清 bearing、上点、下点；两个相机的点必须指向同一个实物位置。

**留下：该 run 的 raw/camera 中的双视频、帧时间和采集信息。** 这一步还不需要 LED START；没有 START 的静态录像不能伪装成已经与电流对齐的实验。

## 4. 从静态录像建立实验坐标

当前已有三点三角化功能，**尚无交互式点选界面，也没有自动识别垂线/垂锤**。第一次可把第 3 步的录像和三点定义交给我协助提帧、制作标注；使用其他点选工具时必须记录原始像素坐标，而不是缩放截图上的坐标。

- [ ] 从已导入且垂锤静止的一对帧中选取三个点：bearing、plumb_top、plumb_bottom。
- [ ] 按 [README 的建立重力坐标部分](README.md#3-建立重力坐标) 填写 `LLM/runs/setup_001/raw/world_reference.json`。
- [ ] `frame_index` 使用本 run 的原始帧编号；有掉帧时不把视频第几帧直接当原编号。

```bash
.venv/bin/python -m LLM world-from-points LLM/runs/setup_001 \
  --reference LLM/runs/setup_001/raw/world_reference.json \
  --calibration LLM/runs/setup_001/calibration/stereo_rig.json
```

新标定已保存旋转信息且与录像一致时，不需要 `--confirm-preprocessing`；不能用这个选项强行接受不匹配的数据。

坐标约定：原点 = bearing 参考点；+z = 下点指向上点；+x = 相机 A 的右方向在垂直 z 的平面内的投影；+y = z × x。桌子不必水平，robot 初始朝向不用于定义重力。

- [ ] 生成 `calibration/world_frame.json`，检查三点重投影及上下方向。
- [ ] 由我协助选几个其他静止帧，单独离线比较方向并保存核对结果，保留原参考和 world_frame；不把一次成功输出当作精度验证。不要修改已保存的参考后重跑同一命令，已有 world_frame 会拒绝覆盖。
- [ ] 保存完整 setup run 和现场布置记录。此后可以移走垂锤，相机和 bearing 保持固定。

**到这里，你已经完成第一阶段。** 静态 run 不必运行 `LLM process`；它还缺少后续实验的同步和轨迹输入。

### 后续实验如何复用这份坐标

只在 A/B 身份、机位、镜头、分辨率、图像旋转和 bearing 参考点均保持一致时，复用同一份 `stereo_rig.json` 与已经建立的 `world_frame.json`，并在新 run 的 manifest 记录 setup run 来源、保留原参考录像。当前没有专门的 `reuse-world` 命令，首次复用由我协助检查和保存快照。

**不要把 setup 的 world_reference.json 像素标注直接交给另一个 run 的 world-from-points。** 那个命令会去读另一个 run 的视频，帧编号相同不等于画面相同。

另一种方式是在同一条实验录像开头先拍静态参考，再移走垂锤，继续录制并随后执行 START。更新后的 Python flight runner 可按 START_SYNC.md 设置 record、telemetry_path、sync_start=True、takeoff=False，准备好后从既有面板启动；不能同时运行第二个相机程序。

## 5. 并行补齐同步与电流记录，再进入驱动实验

下面是正式同步采集前的待办。Flight START 与 Python runner 日志代码已接入，仍需确认实际固件、接线和台架表现；完整执行快照仍待实现。

| 待办 | 你需要提供/操作 | 我需要补齐或核对 | 完成证据 |
|---|---|---|---|
| LED START | 确认当前使用 flight 还是 tilt 固件、LED 接线及两相机可见区域 | 同一设备事件触发 LED 与输入开始，记录设备时刻，保持 GPIO map | 同次录像、START 日志与驱动事件可对应 |
| 原始日志 | 确认实际驱动入口、串口与 PCB 版本 | 在现有唯一串口 reader 处保存原始行和 host 接收时间；保存真实应用的频率/duty/phase | telemetry.jsonl、执行记录及设备时间 |
| 电流读数 | 核对既有标定的仪器、测量位置、量程与工况；提供当前板卡资料 | 沿用已有 SENS，核对读数定义和采样链路 | 清楚说明这是何种电流、校准来源和适用范围 |
| 时间误差 | 在台架上对照 LED、驱动及 CS/电流变化 | 检查固定延迟、抖动及漂移，保存不确定性 | 可检查的时间对照，不声称完全同时 |
| robot 观测点 | 确认两机都能看见的定位点和轴正方向端点 | 人工标注基准，验证自动位置与轴识别 | 真实帧上的位置/姿态有效性 |

电流已经有历史标定：`src/drive_common.h` 使用 A/B/C/D = 15.26/15.28/15.57/15.34 A/V，`src/Tue30JunKCalibration.csv` 保留 I/V 数据。尚需核对它们的完整来源和当前工况，不能说从未标定。普通日志经过约 50 ms 滤波、约 2 Hz 输出，只能按已有 CS 电流估计保存，不能直接称为实际交流基波峰值、RMS 或 phase。

第一版可以先交付带明确定义的 CS 曲线，继续推进视频和 pattern；四路真实 current magnitude/phase 仍列为后续测量任务，不能用 commanded phase 替代 measured phase。

- [ ] 明确 START 事件与设备时间，并验证两相机拍得到暗→亮边沿。
- [ ] 保存日志的入口已经接入实际运行路径；单个串口只有一个 reader，单套相机只有一个采集者。
- [ ] 已检查实际频率应用记录与计划 pattern 的区别。
- [ ] 用已确认的台架条件验证同步误差；实际电流晚于 START 的响应延迟保留，不人为拉齐。

旧固件 LED 开机亮灯/tilt 步骤闪灯不能直接当 START；更新后的 flight 才有本文的 START 事件。`LLM record-telemetry` 不能与正在占用串口的 controller 并行抢读；新路径用原 reader 的 tee。此清单前四步不要求连接 ESP；`probe`、固件烧录/复位及驱动 notebook 都不属于相机测试步骤。

## 6. 采第一组完整实验，先固定一种 pattern

前一步同步与日志联调通过后再做。使用已有、确认过的驱动流程与 thermal 记账，不另外拼接串口启动命令；`LLM` 当前只记录和处理数据，不执行任意 pattern。

- [ ] 新建独立 run，填写 robot、bearing、coil setup、固件版本、初始位置、旋转方向等信息。
- [ ] 选择实验者已确认的短 pattern；保存公共 frequency 随时间变化，以及四路 duty 与 commanded phase。示例 pattern 不是可直接上机的配方。
- [ ] 同一采集者先开始双视频和日志，保留 LED 亮起前的画面与 bearing 上的初始状态。
- [ ] 执行已确认的 START 和输入，记录起飞前、离开 bearing、起飞后一段时间。
- [ ] 按现有驱动流程停止输出，再正常结束采集；确认双视频、帧时间、原始日志完整保存。
- [ ] 在相同条件下重复同一种 pattern，先检查重复性，再改变频率曲线或四路参数。

当前没有“检测到 takeoff 后立即切换 pattern”的在线能力。第一轮只用预先按时间定义、且现有驱动入口确实支持的输入。

## 7. 离线对齐、检查 takeoff、导出

由我协助完成首个真实 run，具体命令和字段按 [README](README.md) 与 [SCHEMA](SCHEMA.md) 执行：

1. 导入已经结束的双视频，保存对应标定和重力坐标快照。
2. 标出两相机 LED 区域；更新后的 flight + JSONL 日志用 `sync-start` 同时关联两相机与 ESP，不再单独执行 clock。
3. 其他具有人工验证时间锚的数据仍可使用 sync + clock；重新运行 sync 会清除旧 ESP 映射。第一版 flight 只发一次 START，不能验证漂移，不能用 host-send 时刻替代设备时间。
4. 生成 robot 同名点标注并运行 `pose`；轴端点不可见时保留位置，姿态标为无效。`auto-pose` 当前仅是实验性中心代理。
5. 人工查看 robot 离开 bearing 的时刻，以真实“旋转但尚未离开”的片段选 baseline，核对高度、上升趋势和持续时间门限。
6. 运行 `process`，检查检测 onset 与确认时间，以及原视频中的实际离开时刻是否一致；不一致时记录原因，不只挑成功片段。

- [ ] 两路视频、frequency、四路 CS 曲线共用实验时间轴。
- [ ] commanded duty/phase 与 measured 电流分开命名；未测量的幅值/phase 不补假值。
- [ ] 从 takeoff 前一小段开始，用相同时间窗口裁剪两路视频和所有曲线；保留完整原始记录。
- [ ] 报告的未同步、无效姿态、掉帧和失败原因均已检查；没有用合成数据冒充实测。

## 8. 交给师兄的数据包

交付整个 run 文件夹，而不仅是曲线截图。至少能找到：

- 两路原视频、逐帧时间和采集元数据。
- 相机标定、实验坐标、静态参考的来源记录及原始参考数据。
- 输入 pattern；区分计划、host 发送、设备实际应用。
- 四路原始电流日志、换算/标定说明；真实 magnitude/phase 若尚未实现则明确注明。
- LED 对时结果、设备时钟映射、轨迹与有效性、takeoff 结果。
- `export/report.md`、`overview.png`、共同时间窗口的视频及 CSV。

## 本次进度记录

| 项目 | 状态/结果 | 数据路径或备注 |
|---|---|---|
| 双相机预览 | 用户已提供双路预览画面 | 2026-10-26；尚不代表标定通过 |
| 物理 A/B、模式与旋转确认 | A=当前索引 1、B=当前索引 0 已由用户指定；旋转仍需保持一致 | 拔插后重新核对；标定采集 1280×800 |
| 标定板实物尺寸与标定通过 | 第二轮独立计算 9/13 pairs，RMS 0.4041 px，软件验收通过 | 当前使用 setup_002；16.52 / 12.39 mm，独立量尺验证待做；[第二轮审核](CALIBRATION_APPEND_REVIEW_2026-10-26.md) |
| 静态参考录像 | 待实测 | |
| world_frame 与尺度/方向核对 | 待实测 | |
| flight START 与 Python 串口日志接入 | 已实现，待实测 | START_SYNC.md；完整 device-applied 快照仍待实现 |
| current sensing 历史标定工况核对 | 待确认 | 已有系数与 I/V 数据 |
| 首个真实同步实验包 | 待前置完成 | |
| 真实电流 magnitude/phase | 待测量链路验证 | 不以设定 phase 替代 |

现场遇到问题时，保存报错、实际命令和原始数据，再在此表记录原因；不要覆盖旧 run 来掩盖失败。
