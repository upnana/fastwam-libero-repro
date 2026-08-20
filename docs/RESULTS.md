# 结果汇总与写法注意

## 官方全量 LIBERO 评测

| 项 | 值 |
|----|-----|
| 权重 | `libero_uncond_2cam224.pt`（官方 release） |
| 机器 | 1× H100 80GB |
| 协议 | 4 suites × 10 tasks × 50 episodes |
| 总 episodes | **2000** |
| Overall | **97.1%（1942/2000）** |
| 论文参考 | ~97.6% |

### 分 suite

| Suite | SR | Successes |
|-------|-----|-----------|
| libero_spatial | 96.8% | 484/500 |
| libero_object | 99.4% | 497/500 |
| libero_goal | 96.4% | 482/500 |
| libero_10 | 95.8% | 479/500 |

逐 task JSON：[`../results/libero_uncond_2cam224_full_eval.json`](../results/libero_uncond_2cam224_full_eval.json)

### LeRobot 修复验证（smoke）

| 设置 | SR |
|------|-----|
| 修复前（双重归一化） | 0% |
| 修复后（VISUAL→IDENTITY） | **100%（5/5）** |

---

## 与论文数字对照

| | Spatial | Object | Goal | Long (libero_10) | Avg |
|--|---------|--------|------|------------------|-----|
| 论文量级（约） | ~97.6% 一带 | — | — | — | **~97.6%** |
| 我们复现 | 96.8 | 99.4 | 96.4 | 95.8 | **97.1** |

差异在正常波动 / 单卡顺序评测实现细节范围内，**足以宣称「对齐官方仿真复现」**。

---

## 动作蒸馏 MVP（次要结果）

| 项 | 值 |
|----|-----|
| Teacher | 官方 FastWAM `libero_uncond_2cam224`（冻结） |
| Dump | `libero_spatial` task 0–9，10 eps/task，**330** samples |
| Student | SmolVLA，5000 steps，仅 `a_teacher` |
| Eval | 对齐 dump task ids，100 eps |
| **Student SR** | **1.0%**（1/100，task 7） |

分析：[`DISTILL_RESULTS.md`](DISTILL_RESULTS.md) · JSON：[`../results/fastwam_distill_multitask_sr_summary.json`](../results/fastwam_distill_multitask_sr_summary.json)

---

## 简历 / 项目经历建议写法

**可以写：**

- 独立复现 FastWAM 官方 LIBERO 评测（2000 episodes），Overall **97.1%**，接近论文 **97.6%**
- 排查 LeRobot 集成 SR=0：定位图像双重归一化并修复，smoke **0%→100%**
- 熟悉 VLA vs WAM、FastWAM「训练有视频建模、推理不做未来想象」的设定
- （可选）探索 FastWAM→小 VLA 动作蒸馏管线；小规模纯 teacher 监督下 student SR ≈ 1%，验证对齐评测与数据瓶颈

**不要写：**

- 「提出 FastWAM / 达到 SOTA」
- 「我训出了与论文一致的模型」（实际是官方权重复现）
- 「复现了官方真机毛巾折叠」（本仓库未做）
- 「蒸馏达到高 SR / 接近 FastWAM」（当前仅 1%）

### 可直接粘贴的条目

**FastWAM / LIBERO 评测复现与链路排错**（2026.07–2026.08）  
- 基于官方权重与评测协议，在 LIBERO 四套件完成闭环评测（40 tasks × 50 episodes，共 2000 episodes），Overall 成功率约 **97.1%**，与论文 Fast-WAM（约 **97.6%**）基本对齐（Spatial/Object/Goal/Long：96.8/99.4/96.4/95.8）。  
- 排查 LeRobot 集成路径下 SR=0 问题：定位为图像 **双重归一化**（checkpoint `MEAN_STD(0.5,0.5)` 与模型内 `x*2-1` 叠加），修复后 smoke 由 **0% → 100%**。
