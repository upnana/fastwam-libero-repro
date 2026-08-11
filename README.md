# FastWAM LIBERO Reproduction Notes

> **复现 / 评测 / 排错记录**，不是 FastWAM 原作者仓库。  
> 官方代码：[yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM) · 论文：[arXiv:2603.16666](https://arxiv.org/abs/2603.16666)

本仓库记录在单卡 **NVIDIA H100 80GB** 上，用官方权重完整复现 FastWAM 在 **LIBERO 四套件**上的闭环评测，以及在 LeRobot 路径下 **SR=0%** 的根因分析与修复验证。

## 一句话结论

| 链路 | 结果 |
|------|------|
| 官方仓库全量 eval（2000 episodes） | **Overall 97.1%**（1942/2000），接近论文 ~97.6% |
| LeRobot 初评（未修 preprocessor） | **SR = 0%**（臂能动，任务全失败） |
| 根因 | 图像 **双重归一化**（Hub `MEAN_STD(0.5,0.5)` + 模型内 `x*2-1`） |
| 修复后 LeRobot smoke | **5/5 = 100%** |

## 复现成功率（官方权重 + 官方评测）

Checkpoint：`libero_uncond_2cam224.pt`  
协议：4 suites × 10 tasks × 50 episodes = **2000** episodes

| Suite | SR | Successes |
|-------|-----|-----------|
| libero_spatial | **96.8%** | 484/500 |
| libero_object | **99.4%** | 497/500 |
| libero_goal | **96.4%** | 482/500 |
| libero_10 | **95.8%** | 479/500 |
| **Overall** | **97.1%** | **1942/2000** |

逐 task 明细见 [`results/libero_uncond_2cam224_full_eval.json`](results/libero_uncond_2cam224_full_eval.json)。

## 文档导航

| 文档 | 内容 |
|------|------|
| [`docs/PROCESS.md`](docs/PROCESS.md) | 完整时间线：环境、命令、中间失败、官方全量 eval |
| [`docs/SR0_ANALYSIS.md`](docs/SR0_ANALYSIS.md) | SR=0 排查过程与双重归一化证明 |
| [`docs/RESULTS.md`](docs/RESULTS.md) | 数字汇总、与论文对照、简历写法注意 |
| [`docs/SETUP.md`](docs/SETUP.md) | 可复跑的环境与命令备忘 |
| [`scripts/run_libero_eval_sequential.sh`](scripts/run_libero_eval_sequential.sh) | 单卡顺序全量评测脚本（无 tmux） |

## 本机关键路径（记录用）

```text
官方仓库     /home/july/FastWAM
官方 conda   fastwam_official
LeRobot 集成 /home/july/lerobot_alohamini
LIBERO       /home/july/LIBERO
全量结果     /home/july/FastWAM/evaluate_results/libero/libero_uncond_2cam224_1e-4_sequential_20260731_181135/
```

## 声明

- 使用的是 **官方发布权重**，不是从零训练。
- 本仓库贡献是：**跑通评测协议、对齐论文量级数字、定位并修复 LeRobot 集成层双重归一化**。
- 写简历请写「复现并验证」，不要写「提出 FastWAM / 刷到 SOTA」。

## License

文档与脚本以 MIT 许可发布；FastWAM / LIBERO / 权重版权归各自原作者。
