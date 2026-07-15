# rgbd2ply — Multi-Camera RGBD → Labelled Pointcloud Pipeline

> 多相机 RGBD → 带语义标签 3D 点云管线

A data pipeline that converts **multi-view RGBD recordings** into **semantically-labelled 3D point clouds** (`.ply`), using SAM3 open-vocabulary segmentation for automatic multi-object annotation, browser-based manual refinement, and dual-camera fusion.

将**多视角 RGBD 录像**转化为**带语义标签的 3D 点云**（`.ply`）。SAM3 自动标注 → 浏览器人工修正 → 双相机深度融合。

```
   RGBD Recording        SAM3 Auto-label        Manual Review         Labelled Pointcloud
   RGBD 录像              SAM3 自动标注          人工修正              带标签点云
  ┌──────────┐         ┌──────────────┐       ┌──────────┐         ┌─────────────┐
  │ cam1 rgb │────────▶│ hand         │──┐    │ box/click│────────▶│ x y z r g b │
  │ cam1 dep │        │ ice_box      │  │    │ seeds    │         │      label  │
  │ cam3 rgb │        │ metal_pot   │  │───▶│          │         │   (per-point)│
  │ cam3 dep │        │ ...          │  │    └──────────┘         └─────────────┘
  └──────────┘         └──────────────┘  │
                        cam{N}_labels_auto.npz  ──▶  cam{N}_labels_final.npz
```

---

## Project Structure · 项目结构

```
infra/
  pyproject.toml                  # Package config (pip install -e .)
  README.md                       # ← You are here
  setup.sh                        #   One-command environment setup · 一键配环境
  pyproject.toml                  #   Package config
  rgbd2ply/                       # Core pipeline · 核心管线
    config.yaml                   #   All paths & defaults — edit first · 先改这个
    cli.py                        #   Unified CLI (rgbd2ply run/batch/review/…)
    pipeline.py                   #   Step orchestrator · 步骤编排
    discovery.py                  #   Input auto-detection · 输入源自动识别
    prepare.py                    #   Frame extraction · 帧提取
    concepts.py                   #   Registry → SAM3 concepts · 标签→概念
    multi_concept_masks.py        #   SAM3 text→mask (called as subprocess)
    seed_masks.py                 #   SAM3 box/click→mask tracking
    fusion.py                     #   Depth back-project + align + filter → PLY
    review_ui.py                  #   Browser annotation UI · 浏览器标注界面
    sweep.py                      #   Prompt exploration · 提示词测试
    labelspec.py                  #   Label↔colour mapping · 标签颜色
    camera_utils/                 #   Camera utilities (vendored, self-contained)
    object_registry.json          #   Object label definitions · 物体标签定义
    auto_concepts.json            #   Generated SAM3 concept list
  deps/sam3/                      # SAM3 model + checkpoint (setup.sh installs) · 外部依赖
  config/                         # Calibration & reference · 标定与参考
    cam1_cam3_extrinsic.json      #   Cam1→Cam3 rigid transform
    env_exports/                  #   Conda environment YAMLs (for reference)
  venv/                           # Python virtual environment (setup.sh creates)
  data/                           # Place recordings here · 录像放这里
  runs/                           # Pipeline outputs · 管线输出
```

---

## Quick Start · 快速开始

```bash
# 0. First time only — one-command setup · 首次运行，一键配环境
bash setup.sh --sam3 /path/to/existing/sam3    # or: bash setup.sh (interactive)

# 1. Activate environment · 激活环境
source venv/bin/activate

# 2. Verify setup · 验证配置
rgbd2ply config

# 3. Discover available data · 查看可用数据
rgbd2ply discover "data/*"

# 4. Run full pipeline · 跑全流程
rgbd2ply run <recording_name>

# 5. Review & fix labels in browser · 浏览器修正标注
rgbd2ply review runs/<recording_name>
#    → Open http://0.0.0.0:8899/

# 6. Re-fuse after corrections · 修正后重新融合
rgbd2ply run <recording_name> --steps fuse --rerun
```

---

## Pipeline · 管线

| Step · 步骤 | CLI flag | What it does · 做什么 | Output · 输出 |
|---|---|---|---|
| **prepare** | `--steps prepare` | Extract RGB+depth frames from recordings or ROS bags · 从录像/ROS bag 提取帧 | `runs/<id>/cam{N}_frames/` |
| **concepts** | `--steps concepts` | Build SAM3 concept list from `object_registry.json` · 生成 SAM3 概念列表 | `auto_concepts.json` |
| **auto** | `--steps auto` | SAM3 text→mask on every frame, per concept · SAM3 逐帧逐物分割 | `cam{N}_labels_auto.npz` |
| **fuse** | `--steps fuse` | Back-project depth + align cameras + filter → `.ply` · 深度反投影+对齐+滤波 | `pointclouds/masked_rgb/*.ply` |

```
concepts ──┐
           ├──▶ auto ──▶ [review] ──▶ fuse
prepare ───┘
```

Steps are **idempotent** — completed steps are skipped unless `--rerun` is set.
每步是幂等的——已完成会自动跳过，加 `--rerun` 强制重跑。

---

## Data Format · 数据格式

Place recordings under `data/`. Each recording is a directory. Both image-mode and ROS-bag-mode are auto-detected per camera.

录像目录放 `data/` 下。图像模式和 ROS bag 模式自动检测，支持混合（如 cam1 用 bag、cam3 用图像）。

### Image mode · 图像模式

```
data/<recording_name>/
  camera_1_image/rgb/frame_times.csv + 000001.png …   # colour frames · 彩色帧
  camera_1_image/depth/frame_times.csv + 000001.png … # depth frames · 深度帧
  camera_1_intrinsics.json                             # cam1 intrinsics · 内参
  camera_3_image/rgb/frame_times.csv + 000001.png …
  camera_3_image/depth/frame_times.csv + 000001.png …
  camera_3_intrinsics.json                             # cam3 intrinsics · 内参
```

### Bag mode · ROS Bag 模式

```
data/<recording_name>/
  camera_1_rgb_depth.bag         # cam1 colour + depth bag
  camera_3_rgb_depth.bag         # cam3 colour + depth bag
  camera_1_intrinsics.json
  camera_3_intrinsics.json
```

---

## Configuration · 配置

All paths and defaults live in [`rgbd2ply/config.yaml`](rgbd2ply/config.yaml). Every value can be overridden via environment variable (prefix `RGBD2PLY_`).

所有路径和参数在 [`rgbd2ply/config.yaml`](rgbd2ply/config.yaml)。每个值都可以用环境变量覆盖（前缀 `RGBD2PLY_`）。

```bash
# Key config values · 关键配置
export RGBD2PLY_PATHS_DATA_ROOT=/path/to/recordings    # where data lives · 数据路径
export RGBD2PLY_PATHS_RUNS_ROOT=/path/to/output        # output directory · 输出路径
export RGBD2PLY_DEFAULTS_THR=0.5                        # SAM3 confidence · 置信度
export RGBD2PLY_DEFAULTS_STRIDE=30                      # frame subsampling · 帧采样间隔

# Verify · 验证
rgbd2ply config
```

---

## CLI Reference · 命令行

| Command | Purpose · 用途 |
|---|---|
| `rgbd2ply run <source>` | Full pipeline on one source · 单个数据源全流程 |
| `rgbd2ply batch "<glob>"` | Batch process multiple sources · 批量处理 |
| `rgbd2ply review <run_dir>` | Launch browser annotation UI · 打开浏览器标注 |
| `rgbd2ply sweep <frames_dir>` | Test prompts against a frames dir · 测试提示词 |
| `rgbd2ply discover <source>` | List available segments · 列出可用数据 |
| `rgbd2ply config` | Print current config · 打印当前配置 |

Common options for `run` / `batch`:

| Flag | Default | Description |
|---|---|---|
| `--steps` | `all` | Comma-separated: `prepare,concepts,auto,fuse` |
| `--rerun` | false | Force re-execute completed steps · 强制重跑 |
| `--thr` | 0.5 | SAM3 detection threshold · 检测阈值 |
| `--stride` | 30 | Frame subsampling (every Nth) · 帧采样步长 |
| `--only` | — | Substring filter on segment id · ID 过滤 |
| `--limit` | — | Max segments to process · 最大处理数 |

---

## Object Registry · 物体注册表

Labels are defined in [`rgbd2ply/object_registry.json`](rgbd2ply/object_registry.json). Each object has a **stable integer id**, a list of SAM3 text **prompts**, and a **max_instances** limit.

标签定义在 [`rgbd2ply/object_registry.json`](rgbd2ply/object_registry.json)。每个物体有固定的整数 id、一组 SAM3 文本提示词、和实例数上限。

```json
{
  "id": 2,
  "key": "ice_box",
  "name_en": "transparent plastic box",
  "prompts": ["transparent plastic box", "clear plastic container"],
  "kind": "object",
  "max_instances": 1,
  "auto": true
}
```

- `id` — **Never change** after masks/pointclouds exist · 已有数据后不要改
- `prompts[0]` — Used for auto-labelling; `sweep` can test alternatives · 自动标注用第一个
- `auto: false` — Skip in auto-label, seed-only via review UI · 仅人工标注

---

## Output Format · 输出格式

### Label masks (`cam{N}_labels_auto.npz` / `_final.npz`)

```python
data = np.load("cam1_labels_auto.npz")
data["labels"]         # int32 [T, H, W]  per-pixel label (0=bg, 1=hand, 2..=objects)
data["frame_indices"]  # int64 [T]        original frame numbers · 原始帧号
data["timestamps"]     # int64 [T]        timestamps in microseconds · 时间戳(微秒)
```

### Pointcloud (`frame_000123.ply`)

ASCII PLY with per-vertex fields: `x, y, z, red, green, blue, label`.

```python
import numpy as np
ply = np.loadtxt("frame_000123.ply", skiprows=9)
xyz = ply[:, :3]       # world-frame coordinates · 世界坐标
rgb = ply[:, 3:6]      # display colour · 颜色 (取决于 --color-mode)
label = ply[:, 6]      # integer label · 整数标签 (0=bg, 1=hand, 2..=objects)
```

---

## Environment · 环境

Run `bash setup.sh` for one-command setup. Or manually:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .                           # rgbd2ply CLI
pip install -e deps/sam3                   # SAM3 (obtain from github.com/facebookresearch/sam3)
pip install torch opencv-python-headless scipy numpy pyyaml rosbags einops pycocotools psutil
```

Reference conda environment YAMLs are in `config/env_exports/` (from the original machine; not needed for this venv).

`config/env_exports/` 里有原机器的 conda 环境导出文件（供参考，本 venv 不需要）。
