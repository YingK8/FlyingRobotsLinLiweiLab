# Pattern 库

在 `record_experiment.ipynb` 的 Block 7 下拉选择。复制一个 JSON 修改后重新运行选择器即可。

命名建议：`rise_2to210Hz_30s_ease_k2.json`；多段用 `rise_then_fall_2to210to170Hz_30s_5s.json`。名字只是说明，程序以 JSON 内容为准，预览图会显示实际形状。

每段 `[起始Hz, 结束Hz, 秒, mode, k]`，mode 0=polynomial、1=ease、2=exponential。相邻频率必须连续，时间为整毫秒；总长度受现有 `controller.control.ramp.check` 限制。这里存配置，不新增 ramp 算法。

当前文件是 2026-10-27 从 `ramp.DEFAULT` 复制的历史 preset，不是当前装置的已验证飞行配方。先查看曲线再选择。录制开始不会发送 ramp，必须另点 START。

`alignment_2to5_5to50Hz_hold10s_19p5s.json` 是电流故障对照：来自归档的
`results/alignment_rate/20260914_droneC_10hzs/050hz/tilt.json`，先用 5 s 从 2 到 5 Hz，
再用 4.5 s 到 50 Hz，两段 EASE k=2；随后 50 Hz 保持 10 s（旧 3 s settle + 7 s hold）。
它只复现切断 A/C 之前的频率曲线，19.5 s 后由 host stop，不执行旧实验后半段的
通道切断与降频。运行仍用当前 flight，未验证当前台架；并非完整旧固件复现。
可在 Block 7 重跑选择器选取，然后单独运行 Block 7B，不需烧录。

当前浏览器实验入口只执行公共 frequency：

- `trigger: "time"`。
- `channel_updates: []`，`channel_settings_status: "firmware_defaults_unrecorded"`。
- 四路 duty/phase 保留固件默认；当前未完整记录其设定值，不输出假数字。实际 CS 估计来自 serial。
- 带 `channel_updates` 的旧离线示例仍可画图，但此执行入口拒绝执行，避免记录了未执行的四路 schedule。
- 正式开始前冻结为 run 下的 `pattern.json`，由同一份数据生成 `ramp.seq_lines` 并保存 host command log。原始计划、host 发送、设备 START 和实测 CS 分开保存。
- 当前入口在整段 ramp 时长到达后发 stop，再录 2 s。停止时间由 host 调度，有串口/采集延迟；没有把它标成精确 device-applied 时间。
- takeoff 触发分支、每路独立 frequency、实测 current phase 仍未实现。
