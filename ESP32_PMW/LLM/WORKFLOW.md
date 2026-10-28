# LLM 实验工作流

入口是 [record_experiment.ipynb](record_experiment.ipynb)。2026-10-27：用户确认已刷 flight；D2/GND LED 接线沿用。代码与离线验证不代替实机联调。

## 先做同步联调（暂缺 CAD/bearing）

重新打开更新后的 notebook 并 Restart Kernel（不要 Run All）。Block 0 默认 `MODE="sync_check"`。
依次运行 **0 → 1 → 4 → 6 → 7 → 8 → 9**；跳过 2、3、5。

- Block 1：核对 A/B、LED 和静止垂线在两机可见，然后 Close preview。
- Block 4：Save static reference pair，再 Close preview；联调模式显示带原始像素坐标的两张图，不要求 tag 检测。白线只是已保存的参考，尚未重建 z。
- Block 6：Record，10 秒后自动收尾；审核报告，完整性应 PASS，有 warnings 时如实核查后填说明。
- Block 7：选择并预览 pattern，填实际 SERIAL_PORT、ROI_A/ROI_B、NOTE；LED ROI 紧贴发光区域，使用保存图像的原始像素坐标。列表只枚举端口，不连接硬件。
- Block 8：这一格才连接/reset flight。核对后 Record，约 1 秒后 START，结束自动 stop + 2 秒后录；不要在录像中按 ESP reset。保留唯一串口、预录制、暗 ROI、一次 START 与停止逻辑；不按估算温度等待冷却。
- Block 9：保留 ANNOTATIONS=None、EXPERIMENTAL_AUTO=False。生成 `export/input_curves.png`，含 frequency、四路 CS、两机 LED 亮度；原始照片随 run 保存到 `raw/static_reference/`。

`sync_check_inputs_only` 表示只完成输入诊断，不是完整三维数据包。`INCOMPLETE.txt` 保留完整几何 pipeline 尚未完成的状态。按各自 LED 边沿对齐后的重合是处理结果，不证明同时曝光或实际电流起始；CS 仍约 2 Hz。此模式不使用旧 world、不输出 XYZ/轴角/takeoff，也不会假定已测得三维重力。

恢复完整三维实验用 MODE="measured"，执行原有完整流程；不能把联调输出直接当作已经通过几何验证的 run。

## 每次要不要重标相机

不必因为刚性 holder 整体移动就自动重做完整标定。内参描述镜头，双目外参描述两相机之间的关系；理想刚体整体移动不会改变它们。改变的是相机相对 bearing/coil 的位置。

昨天两批数据不相容：混合 RMS 11.3381 px，第二批独立求解 0.4041 px。用户报告整体搬过 holder；这些结果不能单独证明搬动必然改变双目外参，也不能排除微小形变、镜头变化或标定偏差。

| 情况 | 操作 |
|---|---|
| 首次使用、调焦/变焦、单台移动、支架变形、裁剪/模式改变 | 完整 ChArUco 标定，再用新图片验证 |
| 两相机刚性整体搬动 | 最终机位新拍 ChArUco 验证旧参数；PASS 可复用；重新定位装置坐标 |
| 同一机位连续拍多次，镜头/装置都未动 | 当天 session 的验证、world 可复用；每次新建 run、重新记录 START |
| 改了相机配置或重新开 kernel/session | 清空旧 gate，重新核对 A/B、验证和试录 |

验证固定 K/dist/双目 R,t，只优化板子 pose；需要至少 5 个新静止 pairs，包含不同位置和倾角。不剔除失败样本来制造 PASS。工程门限并不证明实物毫米精度；标定板尺寸仍需独立核验。完整标定默认输出 `square_mm_measured=false`，传入一个不同数字不再自动冒充量测证据。

## Notebook 顺序

0. 创建本次 session；冻结相机顺序、尺寸、旋转、FPS、backend、codec。更改设置后重新运行此 block。
1. 开 preview，按物理贴纸核对 A/B；如有需要，调焦后再进入标定。
2. 首次或验证失败：完整 ChArUco 采集/求解。每次独立目录，不跨搬动 append。
3. 最终位置拍新板子，验证 rig；FAIL 不能进入已标定的正式实验。
4. 移走板，拍静止装置参考；自动识别 tag36h11，保存有 ID 的 overlay。
5. CAD tag corners + bearing + 双目垂线求世界坐标，同时验证重投影。
6. 点 Record 试录 10 s；自动逐帧审核，确认帧率/host skew 等警告。
7. 选择 pattern，预览曲线，填本机 ESP 端口、两机 LED ROI、实验条件。
8. 打开正式面板 → 核对 A/B 与场景未动 → Record → START pattern + LED → 到时停止/手动 Stop → 自动验收并导入。
9. 后处理先对齐 LED/ESP 并画 frequency + CS；物理点标注到位后再做 XYZ、takeoff、共同裁剪与完整导出。

浏览器服务仅绑定 `127.0.0.1`。相机操作和串口控制都在主线程；按钮回调仅提交事件。当前正式实验为开环 pattern 记录，没有在线轨迹控制。点击 Stop、录制限时结束、Interrupt 或捕获异常时，尝试先发送 stop，再完成视频。不能将其当作断线/强杀进程也有效的硬件保护；沿用 GPIO14 和供电停止操作。

## Flight 与 reset

### 可选 Block 7B：无录像驱动排查

只运行 Block 0、Block 7 的 pattern 选择/参数，再单独运行 7B（`RUN_DRIVE_ONLY=True`）；先结束其他串口程序。它读取下拉框当前选中的 pattern，不打开相机，不要求 ROI/试录/标定，不修改既有视频流程的门限。收到 READY 后立即执行所选频率序列，打印设备频率和 CS，按 pattern 时长停止；2 秒无 START 回报、固件拒绝/按钮停止、Interrupt 都尝试发送 stop 并关闭串口记账。测试结束将代码中的开关改回 False，不要 Run All。

`runs/drive_only/<timestamp>/` 保存 pattern、原始 RX、发送记录和 result.json；没有视频/曝光对时。STOP 发出与固件 OFF 回报分别记录，都不能代替电流实测。USB 断开/进程强杀仍不能保证停机。

flight reset 后进入 IDLE，不自动执行 ramp。面板通过已有 CommandLink/SerialComm 自动 reset，检查 `SYNC_READY v=1`，独占记录原始 RX 日志。Record 之后至少有 0.25 s 源时间跨度，且两个 LED ROI 当前为暗，START 才可用；操作者仍需确认 ROI 实际覆盖 LED。启动由现有 `ramp.seq_lines` 生成 `seq=clear/ramp/go`，ESP 在 go 内触发 LED 和驱动并输出 START 时间。

不要在这次录像内再次按硬件 reset。每个 run 只允许一次 START。启动后按计划时长停止，再保留 2 s；这仍是软件调度，不是精确电流建立/结束测量。视频保存的是 host read-completion 时间，不是同步曝光。

按用户 2026-10-28 的要求，Block 7B/8 不再按估算温度等待冷却，由操作者判断是否继续。实际驱动时间仍由共享 SerialComm 记账；参数检查、READY、到时停止及 Interrupt 停止路径保留。此 notebook 不烧录固件、不改 GPIO。其他 controller 工作流的 thermal gate 保持原样。更新时若旧 cell 正在等待，先 Interrupt；7B/8 会重新加载各自入口，无需重置 session。

## AprilTag 布局尚缺的实物信息

已知：四角 block 的三个面各一张，共 12 张；family tag36h11、ID 1–12；用户转述尺寸 25 mm。已找到 `CAD/coil_array_mount/coil_array_mount_body1.step` 和 `.iges`，STEP 只有一个命名实体，没有 tag ID/旋转和 bearing 基准标注。

需填 [fixture_layout.json](config/fixture_layout.json)：

- 每个 ID 的四个黑色外边角点在同一个 CAD frame 中的位置（mm）。按该编码的标准朝向 TL/TR/BR/BL 排列；标签旋转会改变角点对应。
- bearing 物理中心在 CAD frame 中的位置，不能默认等于底板中心或高度为零。
- 一个固定的装置 x 方向提示。
- 确认 25 mm 是编码区域黑边的外边长，不含周围白纸。

程序先做两台图中的 ID 标注，便于补这张对应表；不会从编号顺序推测位置。当前定位需要至少两个已知 tag 同时被两机看见。重力来自静止垂线两个对应点的双目三角化；一台相机看见一条线还不够。

z=垂线向上，origin=CAD bearing 经过装置位姿转换的位置；x=装置固定方向投影到垂直 z 的平面，y=z×x。这样桌面不水平也保持正交，camera holder 搬动后不会让世界 XY 随相机重新转动。

## 已自动化与仍须人工确认

自动：新 session/run 命名、防旧报告混用、配置/rig/参考文件哈希绑定、相机文件解码验收、tag ID 识别及已知布局定位、pattern 快照与同源执行、START 日志、LED 对时、输入曲线、已有 pose/takeoff/export 调用。

人工：机械刚性与焦距状态、物理 A/B、布局测量/ID 对应、垂线像素、LED ROI 和实际可见性、实验对象与 pattern 选择、CV 物理点/轴的真实性。自动 ellipse center 仍是未验证的代理量；不会输出有效轴角。原始 CS 约 2 Hz，不能得到真实高频幅值/phase。

验证 gate 绑定 session/config，不会自动感知看不见的 holder 松动；移动后须重开 session/重新验证与定位。原始证据和快照保留，运行结果不能直接标为硬件验证成功。

## 保留采集图片后重新验证

如果采集时 kernel 已加载旧版模块，报告保存曾报 NumPy bool 的 JSON 序列化错误，不用重拍。采集结束后在 notebook 新格运行以下代码（`bag`、`RIG`、`SESSION` 沿用刚才那次采集）：

```python
import importlib
import LLM.setup_geometry as geometry
importlib.reload(geometry)
VERIFICATION = geometry.verify_bag(
    bag, RIG, SESSION['camera'], session_id=SESSION['session_id'])
display(VERIFICATION)
```

这是同一批原图重算，不增加或覆盖原图。如果改了镜头/相对位姿则不能靠重算代替重新采集。新开 kernel 会自然加载修正版。

## SYNC_READY 启动失败：单独诊断

先关闭线圈外部供电，保留 ESP 的 USB；确认 Block 8 已停止，关闭其他串口监视器。
不用重新标定或重启 notebook。在新代码格执行以下命令，它用新的 Python 进程加载最新代码，
只 reset / 等待 READY，不打开相机，不发送 pattern：

```python
import subprocess, sys
result = subprocess.run(
    [sys.executable, '-m', 'LLM.serial_check', '--port', SERIAL_PORT],
    cwd=PROJECT,
)
```

等待窗口最多 5 秒。`no serial bytes received` 表示该窗口内完全没接收到字节；
收到字节但没有完整 READY 则检查保存的原始日志。PASS 只确认启动握手，不验证电流测量。
输出位于 `LLM/runs/serial_checks/`。诊断失败不要进入 START；应继续检查供电、USB/UART、EN/BOOT 和固件启动。
若诊断 PASS，恢复正常供电前确认实验状态，然后在原 kernel 重载驱动模块，使 Block 8 使用新版等待逻辑：

```python
import importlib
import controller.control.link as serial_link
import controller.control.hover_controller_runner as flight_runner
importlib.reload(serial_link)
importlib.reload(flight_runner)
```

保持现有 SESSION / ROI / 试录报告，重新运行 Block 8 创建新 run。
