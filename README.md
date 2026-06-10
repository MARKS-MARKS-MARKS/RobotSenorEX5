# 实验五：多层柜子中的物品类别识别

本项目实现《机器人感知与人机交互》实验五的完整流程：RealSense D435 RGB-D 采集、Grounding DINO 开放词汇物品识别、柜层/空位检测、3D 坐标输出、推荐放置策略和可视化结果生成。

## 快速开始

```bash
bash scripts/setup_env.sh
bash scripts/check_env.sh
bash scripts/run_live.sh
```

如果现场暂时无法连接 RealSense，可以先生成并回放一个演示 session：

```bash
conda activate rs_exp5
python -m exp5 --mode demo --session data/sessions/demo
bash scripts/run_replay.sh data/sessions/demo --detector replay
```

## 常用命令

```bash
# 实时识别。默认每 5 帧更新一次 Grounding DINO 类别，其余帧复用缓存以提高稳定性
python -m exp5 --mode live --new-item Tableware --temporal-frames 3 --detect-every 5

# 默认每次 live/capture/replay 启动前都会自动标定柜体 ROI；
# 如果要使用配置文件或命令行里的固定 ROI，可以加 --no-auto-roi
python -m exp5 --mode live --no-auto-roi

# 如果背景被误判为空位，手动指定柜体 ROI：x1,y1,x2,y2
python -m exp5 --mode live --roi 80,40,560,450 --temporal-frames 3

# 如果柜子在俯视/斜视下不是标准矩形，可以指定多边形 ROI
python -m exp5 --mode live --roi-poly-norm "0.12,0.08;0.91,0.05;0.95,0.96;0.10,0.98"

# 自动从当前画面标定柜体 ROI，生成一个可复用的配置片段
bash scripts/run_calibrate_roi.sh --output outputs/roi_calibration --calibration-output configs/calibrated_roi.yaml

# 使用标定后的 ROI 配置运行
python -m exp5 --mode live --config configs/calibrated_roi.yaml --temporal-frames 3 --detect-every 5

# 采集一帧 RGB-D 数据到 session
python -m exp5 --mode capture --session data/sessions/cabinet_01 --temporal-frames 5

# 采集后立即处理，并指定待放置物品实际尺寸：宽,高,深，单位 m
python -m exp5 --mode capture --session data/sessions/cabinet_01 --process-after-capture --item-size 0.20,0.12,0.12

# 离线回放已保存 session
python -m exp5 --mode replay --session data/sessions/cabinet_01

# 检查环境、CUDA、RealSense 和依赖
bash scripts/check_env.sh
```

输出默认保存到 `outputs/latest/`：

- `annotated.png`：完整场景可视化图
- `top_view.png`：按层板展开的俯视占用/空位图
- `map_3d.png`：由当前 RGB-D 帧生成的三维点云建图渲染
- `scene_map.ply`：可用 Open3D、CloudCompare 等工具打开的彩色点云
- `result.json`：物品、层板、空位和推荐放置坐标
- `session/`：capture 模式保存的 RGB-D 数据

## 已加入的拓展

- 泛化性：支持 ROI 自动标定、手动像素 ROI、归一化 ROI、多边形 ROI、多帧 RGB-D 中值融合、类别词表扩展、类别独立阈值、按待放置物体尺寸过滤空位。
- 检测鲁棒性：Grounding DINO 输出不再把多个候选合成大框；只在中心点很近时视为同一物体并保留更紧的原始框。未归入四类但被 `unknown_prompts` 命中的语义检测会作为未知障碍物参与空位判定；已知类别候选低于该类阈值但高于 `unknown_threshold` 时，也会降级成未知障碍。不会再用“过小/过扁框”规则过滤盘子。
- 可视化：实时图上显示稳定 ID、类别、空位尺寸和置信度；右侧面板输出类别统计、3D 坐标和推荐位置；实时同步显示独立 `Top View` 窗口、`3D Map` 窗口和参数控制窗口。
- 增加功能：`--new-item` 可以输入类别或具体物品词，例如 `cup` 会自动归到 `Tableware`；`--item-size` 会改变空位判定；实时参数窗口也可以直接调整新增物体类别和尺寸。

## 说明

- 物品检测主方法为 Grounding DINO，符合开放词汇/零样本要求。
- 未归入四类的物体优先由语义检测生成 `semantic_unknown:*` 障碍；语义漏掉时，再用深度/颜色前景聚类作为兜底 `Obstacle`，不会强行赋予四类类别。
- `configs/default.yaml` 里的 `detector.unknown_prompts` 和 `unknown_threshold` 控制未知障碍语义检测；实时参数窗口的 `Unknown th %` 可现场调节阈值。
- 三维建图由当前对齐 RGB-D 帧在柜体 ROI 内生成，实时用于 `3D Map` 显示，结束时保存 `map_3d.png` 和 `scene_map.ply`。
- 实时参数控制窗口支持调整新增物体类别/尺寸、全局检测阈值、四类类别阈值、检测间隔、多帧融合、ROI、空位阈值和稳定参数。按 `r` 可重新标定 ROI，按 `s` 可保存当前快照。
- 程序默认优先使用 GPU；如果 `torch.cuda.is_available()` 为 `False`，会明确提示并降级到 CPU。
- 空位只在柜体 ROI 内计算。默认会自动估计 ROI，也可以用 `--roi x1,y1,x2,y2` 或 `--roi-poly-norm "x1,y1;x2,y2;..."` 手动框住柜子，避免把柜子外背景当作空位。
- 当前默认层板提取为 `shelf_split_mode: point_cloud`：程序会在柜体 ROI 内把深度反投影成点云，先按 `voxel_size_m` 做体素降采样，再按相机竖直方向聚类宽支撑的水平平面带，最后结合 RGB-D 边缘补齐有效层板边界；如果点云候选不足，会退回稳定边缘兜底。柜层高度差异较大时，也可以填写 `shelf_boundaries_norm: [0.0, 0.31, 0.64, 1.0]` 手工覆盖。
- 物体框必须属于单一层：中心落在某层内，且上下边不能明显越过层边界；跨层大框会直接丢弃。
- 空位判定默认采用“按层投影占用”：某一层内检测到物体或未知障碍后，会把该物体框在这一层对应的整列矩形区域视为占用，剩余连续区域才会输出为 `Empty Space`。
- 实时稳定器补回短时漏检物体后，会再次按最终物体/障碍列表切分空位，避免蓝色空位框包住已经显示出来的物体。
- 实时模式默认开启静态场景稳定器：物体/空位需要连续确认后输出，短时漏检会保持上一状态，并平滑框坐标。调试时可加 `--no-stability` 关闭。
- 你的当前检查结果：`rs_exp5` 环境正常，PyTorch CUDA 可用，RealSense D435I 可枚举，USB 连接为 3.2。
