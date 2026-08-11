# 完整复现过程

> 机器：`july-R5300-G5`，1× NVIDIA H100 80GB PCIe  
> 时间：约 2026-07-31 → 2026-08-01

## 0. 目标

验证 FastWAM 官方 LIBERO checkpoint 能否在本机复现到论文量级成功率，并搞清楚为何在 LeRobot 文档路径下出现 **严格 0%**。

参考：

- 官方仓库：https://github.com/yuantianyuan01/FastWAM
- LeRobot 文档：https://huggingface.co/docs/lerobot/fastwam
- Hub 权重：`ZibinDong/fastwam_libero_uncond_2cam224`

---

## 1. 第一阶段：LeRobot 路径初评（失败）

### 1.1 环境与代码

| 项 | 值 |
|----|-----|
| 工作目录 | `/home/july/lerobot_alohamini` |
| conda | `lerobot_alohamini` |
| 策略代码 | `src/lerobot/policies/fastwam/` |
| 权重 | HF `ZibinDong/fastwam_libero_uncond_2cam224`（safetensors） |

### 1.2 实际跑的配置（偏离官方文档）

| 参数 | 文档复现 | 我们第一次 |
|------|----------|------------|
| `torch_dtype` | float32 | **bfloat16** |
| observation H/W | 256 | **224** |
| `n_action_steps` | 10 | 10 |
| `episode_length` | 300 | 300 |
| suite | libero_spatial | libero_spatial |
| episodes | 50/task | 5/task（后扩到 50） |

### 1.3 结果

- `libero_spatial`，50 episodes → **`pc_success = 0.0%`**
- 视频里机械臂**有运动**（会伸向目标区域），但抓取/放置从未完成
- 输出目录示例：`outputs/eval/2026-07-31/14-15-18_libero_fastwam/`

### 1.4 当时的误判与排除

1. **怀疑 bf16 / 224**：合理，但不该导致严格 0/50。  
2. **怀疑跑错仓库**（`lerobot_alohamini` vs `lerobot`）：对照后 FastWAM 核心一致，排除。  
3. **试 `bf16 + 256` smoke**：跑通，仍 **SR=0**。  
4. **试 `float32 + 256`**：模型加载约 57GB，旁边还有 pi05 训练占 ~22GB → **OOM**，未能完成对照。

到这一步只能确定：**不是权重完全坏了，而是评测链路存在系统性偏差**。

---

## 2. 第二阶段：按官方 GitHub 重建环境（成功）

用户要求：新建 conda，严格跟 [FastWAM](https://github.com/yuantianyuan01/FastWAM) 走；权重不一致就重下；LIBERO 能用就复用。

### 2.1 环境

```text
conda env: fastwam_official
Python:    3.10
Torch:     2.7.1+cu128
Repo:      /home/july/FastWAM
Install:   pip install -e .
MuJoCo:    3.3.2
LIBERO:    /home/july/LIBERO（复用；修了 PyTorch 2.7 的 torch.load weights_only）
```

### 2.2 权重

| 资产 | 处理 |
|------|------|
| 本地旧 LeRobot safetensors | 与官方 `.pt` **不一致**，不直接当官方 ckpt 用 |
| 官方 policy | 重新下载 `checkpoints/fastwam_release/libero_uncond_2cam224.pt` + `*_dataset_stats.json` |
| VAE / T5（DiffSynth converted） | hash 与官方一致 → 复用并 symlink 到 `checkpoints/DiffSynth-Studio/` |

### 2.3 Smoke

```bash
# 概念命令：官方 experiments/libero 单 task
# suite=libero_spatial, task_id=0, num_trials=1
```

结果：**1/1 success** → 官方权重本身是好的。

### 2.4 全量 eval（单卡顺序）

官方并行 manager 需要多卡 / tmux；本机只有 1×H100 且无 sudo 装 tmux，因此使用顺序脚本：

[`scripts/run_libero_eval_sequential.sh`](../scripts/run_libero_eval_sequential.sh)

要点：

- 顺序跑 `libero_10 → libero_goal → libero_spatial → libero_object`
- 每 task 50 trials，共 40 tasks = **2000 episodes**
- 每 task 会重新加载约 6B 模型（显存峰值约 47GB）
- 每个 episode 保存 mp4（文件名含 `success=True/False`）

输出目录：

```text
/home/july/FastWAM/evaluate_results/libero/libero_uncond_2cam224_1e-4_sequential_20260731_181135/
```

### 2.5 全量结果

| Suite | SR | 成功数 |
|-------|-----|--------|
| libero_spatial | 96.8% | 484/500 |
| libero_object | 99.4% | 497/500 |
| libero_goal | 96.4% | 482/500 |
| libero_10 | 95.8% | 479/500 |
| **Overall** | **97.1%** | **1942/2000** |

备注：脚本曾把部分 task 标成 `FAILED`（因为结果文件名是 `gpu0_task*_results.json`，脚本期望 `task*_results.json`），但进程 `rc=0` 且 50 episodes 齐全——**评测本身成功完成**。成功率以 JSON / 视频 `success=` 为准。

耗时量级：约十几个小时量级（单卡顺序 + 每 task 重新 load）。

---

## 3. 第三阶段：回到 LeRobot，定位 SR=0

官方链路已证明 checkpoint 可用 → LeRobot 的 0% 一定在 **集成 / preprocessor 契约**。

最终根因：**图像双重归一化**（详见 [`SR0_ANALYSIS.md`](SR0_ANALYSIS.md)）。

修复：加载 Hub checkpoint 后，强制把 preprocessor 的 `VISUAL` 从 `MEAN_STD` 改成 `IDENTITY`。

验证（与失败配置对齐的 smoke）：

| | 修复前 | 修复后 |
|--|--------|--------|
| 配置 | bf16 + 256 + task0 | 相同 |
| episodes | 1 或 50 | **5** |
| SR | **0%** | **100%（5/5）** |

日志关键字：

```text
FastWAM: overriding loaded preprocessor VISUAL normalization MEAN_STD → IDENTITY ...
```

---

## 4. 过程中踩过的坑（清单）

1. **LeRobot 初评 SR=0**：表象是「策略不会做任务」，根因是像素范围错误。  
2. **bf16 / 224 vs float32 / 256**：会造成掉分嫌疑，但不是 0% 的充分解释。  
3. **float32 OOM**：H100 80GB + 旁路训练占显存，无法完成官方文档级 float32 对照。  
4. **权重格式不一致**：LeRobot safetensors ≠ 官方 `.pt`；官方路径需重下。  
5. **单卡无法直接用官方多卡 manager**：改成 sequential 脚本。  
6. **status 文件误报 FAILED**：结果文件命名与脚本检查路径不一致。  
7. **PyTorch 2.7 + LIBERO**：`torch.load` 默认 `weights_only` 变更，需兼容修补。  
8. **MUJOCO_GL=egl**：无头服务器必须设 EGL，否则渲染失败。

---

## 5. 没做什么（边界）

- 没有从零训练 FastWAM。  
- 没有复现论文真机毛巾折叠（Galaxea R1 Lite 数据/代码未开源完整）。  
- 没有在本仓库上传多 GB 权重与数千条 mp4（只保留数字与过程）。
