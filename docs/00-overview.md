# 总览：这个项目到底在解决什么问题

## 一句话

在一个双核 650 MHz ARM + Artix-7 FPGA 的教学板上，把 Fashion-MNIST 分类从"训出
94.77% 的模型"推进到"在不同业务约束下，各自交出一个能跑的模型"。

前者很多人能做到。后者才是这个项目真正做过的事。

## 为什么这件事值得做

PYNQ-Z1 这类小型异构 SoC 有三种推理路径：纯 PS 侧 CPU、PL 侧 FPGA 定制加速器、
PS/PL 混合流水线。论文里有对它们的理论比较，但缺的是**工程参考**：每条路到底要
花多少天、踩哪些坑、在什么约束下该选哪条、失败时长什么样。

这个项目把三条路都走了一遍，其中两条走通、一条卡在最后一环，并且把全部中间决策
和失败原因记录了下来。

## 三条路线的最终状态

| 路线 | 精度 | 吞吐 | 周期 | 状态 |
|---|---:|---:|---|---|
| **CPU-only ONNX** | 92.21% | 445 img/s | 天级 | ✅ 完整闭环，四档交付 |
| **BNN-PYNQ (1W1A)** | 83.16% | 154 096 img/s | 周级 | ✅ 板端闭环，精度受限 |
| **FINN 高质量主线 (4W4A)** | 93.55% | 未达成 | 周级 | ⚠️ bitstream 可跑，数值正确性未闭环 |

第三条没有走完。它没有被包装成成功 —— 见
[05-fpga-finn-deployment.md](./05-fpga-finn-deployment.md)。

## 交付的不是"一个最好的模型"

最终交付是**一组分档模型**，因为边缘场景的需求在项目过程中会变，事先无法确定用户
最终要"最高精度"还是"最快速度"：

| 档位 | 模型 | 精度 | 板端吞吐 | 10000 张耗时 |
|---|---|---:|---:|---:|
| 精度优先 | `tinyplus_kd_fulltrain` | 92.21% | 186.8 img/s | 53.5 s |
| 中间档 | `tinyfast_xs` | — | ~310 img/s | — |
| 平衡档 | `tinyfast_xxs` | 91.09% | 375.99 img/s | 26.60 s |
| 速度优先 | `tinyfast_xxxs` | 90.45% | 445.19 img/s | 22.46 s |

从精度档到速度档：丢 1.76 个百分点，换 2.38 倍吞吐。这条曲线本身就是交付物。

## 核心方法论：部署约束前置

> 在资源受限平台上，模型设计必须从一开始就服从部署约束，而不能先训练后适配。

这不是口号，是这个项目用三次结构性重做换来的：

1. `q_md` 系列含 `GlobalAveragePool` 等算子 → 进不了 FINN 数据流化 → 重做为
   `finnconv_direct_head8`
2. `q_l` 系列权重存储预估超 BRAM 预算 → 提前放弃（省下几小时 full build）
3. EfficientNet-B0 直接上 ARM CPU → ~10 img/s → 重新设计极小 CNN

三者的共同点是：**如果第一步就知道约束，一次都不用重做。**

## 文档导航

| 文档 | 内容 |
|---|---|
| [01](./01-platform-and-path-selection.md) | 平台资源与 PS/PL 决策框架 |
| [02](./02-training-from-baseline-to-teacher.md) | 从基线 CNN 到 94.77% 教师 |
| [03](./03-distillation.md) | 蒸馏：为什么不只是一个提点技巧 |
| [04](./04-quantization-and-finn-friendly-design.md) | 量化与 FINN-friendly 结构设计 |
| [05](./05-fpga-finn-deployment.md) | FINN full build 与板端分层验证 |
| [06](./06-bnn-route.md) | BNN 二值网络路线 |
| [07](./07-cpu-only-route.md) | CPU-only 路线与板端调优 |
| [08](./08-scenario-playbook.md) | ★ 六场景最佳实践速查 |
| [09](./09-lessons-and-negative-results.md) | ★ 负结果与教训 |
| [10](./10-reproducing.md) | 复现步骤与预期耗时 |

如果只有十分钟，读 [08](./08-scenario-playbook.md) 和
[09](./09-lessons-and-negative-results.md)。

## 数据口径

所有数字以 [`benchmarks/results.csv`](../benchmarks/results.csv) 为唯一来源，
该文件逐行标注了论文出处。口径说明见
[`benchmarks/README.md`](../benchmarks/README.md)。
