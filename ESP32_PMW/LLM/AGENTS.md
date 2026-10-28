# LLM 实验 pipeline 的工作约定

先读本目录 README.md、STATUS.md、SCHEMA.md，以及仓库 CLAUDE.md。

- 这一部分统一称为 LLM。当前交付是给下游 LLM 使用的实验数据包；LLM 调用、磁场优化和自动执行另行接入。
- 原始视频、源时间戳、标定、pattern 和 serial 日志是证据。保留原件；处理产物可以重建。不得从视频 FPS 伪造采集时刻。
- 始终分开 planned、host_sent、device_applied 和 measured。duty 不是电流幅值，设定 phase 不是实测电流 phase。
- LED START 定义实验零点；真实电流起始允许有延迟。单个 LED 事件不能验证漂移，host read-completion 不是曝光时间。
- 实验坐标 z 来自静止垂线，origin 来自固定 bearing；x 是方向提示在垂直 z 平面上的投影。不得拿 robot 初始姿态当重力或给每次视频重新去均值。
- `position_valid` 与 `orientation_valid` 独立。自动 ellipse center 是尚未验证的代理量，不能写成真实物理点定位精度。未观察到轴端点时不要输出“有效”的两个姿态角。
- takeoff 是带持续确认的高度阈值事件，暂为 bearing 脱离的代理量。保留 onset、confirmation、区间和失败原因；无效帧或长缺口不能用上一帧填充。
- 导出两视频使用同一个实验时间窗口；预览重采样必须保留 source-frame mapping，原始视频供分析。
- 复用 controller 的录制/标定/分割/串口接口；ramp 定义和采样只放 controller/control/ramp.py。不绕过 SerialComm，不改 GPIO map。2026-10-28 用户明确取消 LLM Block 7B/8 的模型冷却等待，由操作者判断是否继续；不要自动重新加入 _thermal_gate。串口驱动时长记账与停止逻辑保留，其他 controller 工作流不受此更改影响。
- `LLM` CLI 当前没有驱动/烧录操作；notebook 的显式驱动沿用唯一 ramp 入口与串口驱动记账。不能把此模块的离线验证记成上机验证。
- 非平凡改动运行 `.venv/bin/python -m LLM self-check`，不引入 pytest。只报告实际运行过的验证，并同步 STATUS.md 的完成项和待验证项。
- 用户已有的 MATLAB、notebook、SPIFFS 和 results 改动不属于本模块，保留原状。
