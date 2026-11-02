# Coil array holder 标记检测

最新 `holder_v3_boss_D73p8_H5` 使用黑白四象限标记，不是 AprilTag。每个图案为 16×16 mm，四个象限各 8×8 mm，定位点是材料边界的中心交点。顶部 4 个、侧面 16 个，总共 20 个。现有 `inspect_tags` / `localize_fixture` 继续用于旧纸质 tag36h11。

## 使用入口

在 `record_experiment.ipynb` 第 5 步（`Bench` 方法）：

1. `bench.photograph_reference()` 保存静态双目参考图。
2. `bench.find_holder_marks()` 第一次运行生成参考目录的 `holder_regions.json`，并显示候选交点（`c0`、`c1` 只是当前图候选序号）。按原始像素填写两台相机可见的 CAD 点与 `[x,y,width,height]`，再运行一次自动细化并检查叠加图。每个框只包含一个目标交点，四周留至少 13 px；没有观测到的点不要填写。
3. `bench.locate_fixture(plumb)` 将名义布局复制为 session 内 `holder_layout.json`。确认打印件与 CAD 对应，实测并填写 `bearing_center_fixture_mm`，再设置 `survey_confirmed=true`。`plumb` 填本次静止垂线的双目 top/bottom 像素。

仅检测不需要标定、bearing、串口或相机连接；sync check 可跳过 2、3。输出 `A_holder_landmarks.png`、`B_holder_landmarks.png` 和 `holder_detections.json` 可重建；原图不改写。模板不会覆盖已有文件。

## CAD 命名与身份限制

| 名称 | 物理点 |
|---|---|
| P1、P2、P3、P4 | 四根柱子的顶面交点，CAD Z=40 mm |
| Pn_X_L、Pn_X_H | 固定 CAD X 的朝外侧面，CAD Z=12 / 30 mm |
| Pn_Y_L、Pn_Y_H | 固定 CAD Y 的朝外侧面，CAD Z=12 / 30 mm |

P1=(1.5,−37.5,40)，P2=(97.5,−37.5,40)，P3=(97.5,58.5,40)，P4=(1.5,58.5,40) mm。X 侧面位于 −10.5 或 109.5 mm，Y 侧面位于 −49.5 或 70.5 mm；完整值见 [holder_layout.json](holder_layout.json)。这些是原始 CAD 坐标，不能当作相机或重力坐标。

## CAD 实体：`coil_base.step`

`coil_base.step`（2026-11-02 由 Autodesk 导出，单位 mm，单个实体 Body1）是装置底座的 CAD，与 `holder_layout.json` 放在一起，以后用同一个 CAD→camera A 变换（`localize_holder` 的 `T_camera_A_from_fixture`）把它叠到标记位置上。

- 包围盒 X −28…159.5、Y −109.5…71.25、Z −5…40 mm。顶面 Z=40 与 P1–P4 一致，Y 上限 71.25 与 Y 侧面 70.5 mm 只差 0.75 mm，**看起来**与 landmark 同一 CAD 坐标系，但未核对：landmark 来源 `CAD/coil_array_mount/landmarks_v3_boss/landmarks.json` 不在本仓库，STEP 也比标记范围大（X 到 159.5、Y 到 −109.5）。叠加前先在 CAD 里确认原点一致。
- 尚无代码读取它；它不参与定位，也不改变 `holder_layout.json` 的哈希。

柱子旁的 1–4 个小白点表示 P1–P4，每点直径 2.4 mm。小点被遮挡会造成数错，且数量不区分 X/Y 面；本实现没有自动圆点解码，也不会从图像左右位置猜 CAD 身份。操作者一次指定当前静态图中的命名框，算法自动定位框内交点。换图后文件哈希不匹配，必须重新建立对应。

ROI 的 JSON 结构如下；`image_sha256` 由模板自动生成，不能用示例替代。空字典表示该相机尚未指定点：

```json
{
  "schema_version": 1,
  "image_sha256": {"A": "由模板生成", "B": "由模板生成"},
  "regions": {"A": {}, "B": {}}
}
```

在 `regions.A` 或 `regions.B` 中添加 `"P1_Y_H": [x,y,width,height]` 等条目。没有检测、框中有多个候选、多个名称指向同一交点时都不输出该名称的有效 `center_px`。周围其他棋盘格也可能成为候选，必须检查命名区域。

## 检测与定位

`holder_detection.detect_checker_centers` 用角点候选、亚像素细化，以及多个同心圆上的明暗四扇区验证交点；外框角点、普通边缘和圆点不符合此局部结构。默认采样半径为 3/5/8/12 px，至少两个尺度通过；最低灰度对比为 35/255。这些是离线工程默认值，尚未按实物标定。`quality` 是诊断分数，不是概率或毫米精度。

`holder_geometry.localize_holder` 每机要求至少 4 个非共线已命名点，可使用不同侧面。保持相机内参和双目外参固定，联合拟合一个 CAD→camera A 刚体变换。它检查正深度、面朝向、几何退化、重投影 RMS<1 px / 单点<3 px 和候选多解；不通过时拒绝输出世界坐标。对称地错误命名所有点仍可能通过几何检查，因此明确的物理身份核对不可省略。

只有相同完整名称才代表同一物理点。holder 的不同侧面点不会相互三角化。世界原点由实测 bearing 的 CAD 坐标转换得到，z 始终由静止垂线 bottom→top 决定，x 是 CAD 固定方向在水平面上的投影。

## 验证范围

离线测试覆盖旋转/透视/模糊/噪声、外框/边缘/圆点负例、缺失/歧义/重复 ROI、输入哈希、独立双目对应与几何拒绝。CAD 渲染图检查检出侧视 4/4、斜视 12/12 个可见交点。实拍识别率、遮挡/反光、打印尺寸误差和物理定位精度均待验证。保存实际参考对，检查命名叠加图和重投影残差后才能评估实物效果。

源布局来自 `CAD/coil_array_mount/landmarks_v3_boss/landmarks.json`，并保存其 SHA-256。该历史 JSON 只记录侧面编号点；本配置依据 `landmarks_v1/BuildLandmarks.py` 补记仍存在于 v3 模型的顶部编号点，未修改历史 CAD 文件。
