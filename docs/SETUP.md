# 环境与命令备忘

> 以下为本次复现实际使用的路径与命令骨架，便于日后重跑。权重与数据集体积很大，不在本仓库内。

## 官方仓库路径（推荐用于对齐论文数字）

```bash
cd /home/july/FastWAM
conda activate fastwam_official

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export DIFFSYNTH_MODEL_BASE_PATH=/home/july/FastWAM/checkpoints
export TOKENIZERS_PARALLELISM=false
```

### 权重位置

```text
checkpoints/fastwam_release/libero_uncond_2cam224.pt
checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
checkpoints/DiffSynth-Studio/...   # VAE / T5 converted
```

### Smoke（单 task）

使用官方 `experiments/libero/eval_libero_single.py`，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=1 \
  gpu_id=0 \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json
```

### 全量（单卡顺序）

```bash
bash scripts/run_libero_eval_sequential.sh
# 或本仓库副本：
# bash /path/to/fastwam-libero-repro/scripts/run_libero_eval_sequential.sh
```

环境变量可覆盖：

| 变量 | 默认 | 含义 |
|------|------|------|
| `CONFIG` | `libero_uncond_2cam224_1e-4` | task config 名 |
| `CKPT` | `./checkpoints/fastwam_release/libero_uncond_2cam224.pt` | 权重 |
| `NUM_TRIALS` | `50` | 每 task episodes |
| `TASK_SUITES` | 四个 suite | 可缩到 `libero_spatial` 做快速重跑 |

---

## LeRobot 路径（曾 SR=0，需 VISUAL reconcile）

```bash
cd /home/july/lerobot_alohamini
conda activate lerobot_alohamini

# 文档风格（注意：未修双重归一化前会塌零）
lerobot-eval \
  --policy.path=ZibinDong/fastwam_libero_uncond_2cam224 \
  --policy.device=cuda \
  --policy.torch_dtype=bfloat16 \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --eval.batch_size=1 \
  --eval.n_episodes=5 \
  --seed=0 \
  --env.episode_length=300
```

修复后应在日志中看到：

```text
FastWAM: overriding loaded preprocessor VISUAL normalization MEAN_STD → IDENTITY
```

官方文档仍建议 `float32`；在 H100 80GB 上 float32 权重加载约 **57GB**，若旁边还有训练进程容易 OOM。

---

## LIBERO / 渲染

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
# LIBERO 源码：/home/july/LIBERO
```

PyTorch 2.7 下若 `torch.load` 因 `weights_only` 失败，需按环境做兼容（本机已对 LIBERO 做过修补）。
