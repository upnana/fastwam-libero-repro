# FastWAM → SmolVLA 动作蒸馏：实验结果与分析

> 对照实现：`lerobot_alohamini/examples/alohamini/fastwam_distill/`  
> 原始输出：`/home/july/rxn/outputs/fastwam_distill/`  
> 机器：1× NVIDIA H100 80GB  
> 跑完时间：2026-08-20 ~14:11

**定位：** 这是 **工程探索 / MVP**，不是 SOTA。主简历数字仍是官方 FastWAM **97.1%**（2000 eps）。

---

## 1. 做了什么

```text
冻结官方 FastWAM (libero_uncond_2cam224)
  → 多 task env dump：(obs, a_teacher)
  → SmolVLA 离线拟合 a_teacher
  → 只在「dump 过的 task ids」上评 SR（aligned eval）
```

| 阶段 | 设置 |
|------|------|
| Teacher | 官方 `libero_uncond_2cam224.pt`，冻结；本地 Wan，无 Hub |
| Dump | `libero_spatial`，**task_ids = 0…9**，**10 eps/task**，stride=8 |
| Dump 样本 | **330** 条 action chunk，shape `[330, 32, 7]`（env-space） |
| Student | `lerobot/smolvla_base`，**5000** steps，batch=4，chunk=32 |
| Eval | **对齐 dump**：同一 10 个 task，**10 eps/task**（共 100 eps） |

机器可读汇总：[`../results/fastwam_distill_multitask_sr_summary.json`](../results/fastwam_distill_multitask_sr_summary.json)

---

## 2. 结果

| 指标 | 值 |
|------|-----|
| **Student pc_success** | **1.0%**（1/100） |
| 成功出现位置 | `libero_spatial` **task_id=7**，1 个 episode |
| 其余 task | 0/10 |

与对照：

| 链路 | SR | 说明 |
|------|-----|------|
| 官方 FastWAM 全量（同机） | **97.1%** | teacher 能力上界参考 |
| Distill smoke（早前：单 task dump + 全 suite eval） | ~1% | 评测不对齐，数字难解释 |
| Distill multi-task + **aligned** eval（本轮） | **1.0%** | 管线对齐了，结果仍接近随机成功 |

**结论一句话：** 多 task dump + 对齐 eval **管线已跑通**；在当前数据量 / 纯 teacher 标签设定下，**student 基本不会做任务**。

---

## 3. 为什么只有 ~1%

按可能性从高到低：

1. **监督数据极少**  
   330 chunks ≈ 每 task 几十条窗口；对比 LIBERO 人类 demo 有数十万帧量级。BC / distill 很难从这么少样本学到闭环策略。

2. **标签全是 `a_teacher`，且未筛成功轨迹**  
   Teacher rollout 也会失败或抖；失败轨迹进训练集会把 student 往「不会完成任务」的方向拉。

3. **训练预算有限**  
   5000 steps × batch 4，对 SmolVLA 全量微调偏短；loss 可降，不等于闭环 SR 升。

4. **Teacher–student 容量 / 表征差距**  
   FastWAM（WAM + Wan）→ SmolVLA，动作空间虽对齐到 env-space，但视觉–语言骨干不同，纯动作模仿难完全迁移。

5. **不是评测脚本坏了**  
   本轮已限制 `--env.task_ids=[0..9]`，与 dump 一致；`n_episodes=100` 与「10×10」对齐。低 SR 更像 **学不会**，不是「评了没 dump 的 task」。

---

## 4. 和「混 demo GT」的关系

LIBERO 是 **仿真** 数据；`HuggingFaceVLA/libero` 里的 `action` 是 **仿真 teleop 演示**（demo GT），不是真机。

| 标签 | 来源 |
|------|------|
| `a_teacher` | FastWAM 在 sim 里 rollout（本轮唯一监督） |
| `a_demo` / demo GT | 数据集里人类遥操作动作 |

「混一点 demo GT」= 训练时除了 teacher dump，再掺一部分 `(obs, a_demo)`。  
**不是**「蒸馏不能只靠 teacher」，而是「**只靠**少量 teacher rollout 往往不够；混成功演示通常更稳」。

本轮 **尚未实现** 混 GT。

---

## 5. 后续若要抬 SR（优先级）

1. **加大 dump**：更多 eps/task，或只保留成功 episode 再 dump  
2. **混 demo GT**（或先用纯 LIBERO BC 作对照 baseline）  
3. **加长 train**（更大 steps / 合理 lr）  
4. 保持 **aligned eval**；报数字时写清协议（哪些 task、多少 eps）

---

## 6. 简历 / 对外怎么写

**可以写：**

- 搭了 FastWAM→SmolVLA **action distillation** 管线（多 task dump、对齐闭环 eval）  
- 验证：在小 dump（330 samples）+ 纯 teacher 标签下 student SR ≈ **1%**；说明蒸馏需要更大 / 更干净监督  

**不要写：**

- 「蒸馏接近 / 超过 FastWAM」  
- 用 1% 当主成果；主数字仍是官方 **97.1%** 复现  

概念说明见 [`DISTILL_EXPLAINED.md`](DISTILL_EXPLAINED.md)，代码对照见 [`DISTILL_CODE_EXAMPLES.md`](DISTILL_CODE_EXAMPLES.md)。
