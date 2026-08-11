# LeRobot 路径 SR=0 分析

## 结论

**不是权重坏了，也不是 LIBERO 环境完全装错。**  
在 LeRobot 评测链路里，图像被归一化了两次，Wan VAE 看到的像素分布系统性偏移，导致动作「有模有样」但任务成功率塌到约 **0%**。

同一套官方能力在官方仓库评测中可达到 **Overall 97.1%**；修好 preprocessor 后，LeRobot smoke 从 **0% → 100%（5/5）**。

---

## 现象

1. `lerobot-eval` + `ZibinDong/fastwam_libero_uncond_2cam224`
2. `libero_spatial`，50 episodes → `pc_success = 0.0%`
3. rollout 视频：机械臂有运动，轨迹因 task 而异 → **不是输出全零 / 权重没加载**
4. 改成 `bf16 + 256` 小 smoke，仍是 0
5. 官方仓库同能力权重 smoke / 全量 eval 正常 → **问题在 LeRobot 集成层**

---

## 根因：双重归一化（Double Normalization）

### 契约应该是什么

FastWAM（Wan VAE）期望：

```text
policy 输入图像 ∈ [0, 1]
VAE 边界再做一次：x * 2 - 1  →  [-1, 1]
```

代码侧约定（`configuration_fastwam.py`）：

```python
"VISUAL": NormalizationMode.IDENTITY
```

模型编码处（`wan/modular.py`）：

```python
# The Wan VAE expects pixels in [-1, 1]; model inputs arrive in [0, 1]
video_tensor = video_tensor * 2.0 - 1.0
```

### 实际发生了什么

Hub checkpoint 自带的 `policy_preprocessor`：

| 步骤 | 谁做的 | 变换 | 范围变化 |
|------|--------|------|----------|
| ① | preprocessor `MEAN_STD`，`mean=0.5, std=0.5` | `(x-0.5)/0.5` | **[0,1] → [-1,1]** |
| ② | 模型 VAE 前 | `x * 2 - 1` | **[-1,1] → [-3,1]** |

Wan VAE 训练分布是 **[-1, 1]**。输入被拉到约 **[-3, 1]** 后，latent 偏移，策略仍输出「看起来像动作」的序列，但闭环任务失败 → SR≈0。

### 用数值直觉理解 `x * 2 - 1`

| 原值 x（[0,1]） | `x*2-1` |
|----------------|---------|
| 0 | -1 |
| 0.5 | 0 |
| 1 | +1 |

注意：`(x-0.5)/0.5` **在数学上等价于** `x*2-1`。  
所以 preprocessor 已经做完「该做的那一次」，模型里再做一次就是重复。

---

## 为什么一开始容易误判

| 嫌疑点 | 为什么像真凶 | 为什么不是充分解释 |
|--------|--------------|--------------------|
| bf16 而非 float32 | 文档写 float32；VAE cast 警告 | 通常掉分，不该严格 0/50；bf16+256 仍 0 |
| 224 而非 256 | 文档 spatial 写 256 | 改成 256 后仍 0 |
| 跑在 `lerobot_alohamini` | 担心 fork 差异 | 与 `lerobot` FastWAM 核心一致 |
| 权重损坏 | SR=0 很吓人 | 官方仓库同能力权重 48/50、全量 97.1% |

关键对照实验：

1. **官方仓库**：高 SR → checkpoint 能力 OK  
2. **LeRobot 修 VISUAL→IDENTITY 后**：smoke 100% → 根因坐实为 preprocessor 契约

---

## 修复方式

加载 pretrained processors 后，调用 reconcile：

```python
def reconcile_fastwam_processors(config, preprocessor, postprocessor):
    """Force VISUAL to IDENTITY; leave STATE/ACTION norms untouched."""
    for step in preprocessor.steps:
        if not isinstance(step, NormalizerProcessorStep):
            continue
        visual_mode = step.norm_map.get(FeatureType.VISUAL)
        if visual_mode is None or visual_mode == NormalizationMode.IDENTITY:
            continue
        logging.warning(
            "FastWAM: overriding loaded preprocessor VISUAL normalization %s → IDENTITY ...",
            visual_mode,
        )
        step.norm_map[FeatureType.VISUAL] = NormalizationMode.IDENTITY
    return preprocessor, postprocessor
```

在 `factory.py` 加载 FastWAM processors 后调用。

本机对应文件：

- `/home/july/lerobot_alohamini/src/lerobot/policies/fastwam/processor_fastwam.py`
- `/home/july/lerobot_alohamini/src/lerobot/policies/factory.py`

### 验证结果

| | 昨天（未修） | 今天（修后） |
|--|-------------|--------------|
| 配置 | bf16 + 256 | 相同 |
| episodes | 失败 / 0% | **5/5 = 100%** |
| 日志 | 无 override | 出现 `MEAN_STD → IDENTITY` |

STATE / ACTION 仍使用 checkpoint 的归一化（未改）。

---

## 经验教训（可迁移）

1. **SR=0 且机器人仍在动** → 优先查观测/动作尺度契约，而不是先怀疑「模型没训好」。  
2. **Hub preprocessor 与模型代码可能不一致**；要以「模型入口期望的分布」为准做 reconcile。  
3. **官方仓库对照**是最快的「权重是否坏了」判决器。  
4. 同类问题也可能出现在其它把 VAE/扩散骨干接到 LeRobot processor 的策略上。
