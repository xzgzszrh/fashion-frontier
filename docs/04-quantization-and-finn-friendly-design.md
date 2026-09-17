# 量化与 FINN-Friendly 设计

## 为什么"精度够"还不够

确立部署目标为 PL 侧 FINN 数据流加速器后，遇到的关键挑战是：

> 普通量化 CNN 即使精度足够，也未必能顺利通过 Brevitas → QONNX → FINN 的构建链路。

完整链路：

```
PyTorch → Brevitas（QAT）→ QONNX 导出 → FINN 前端 → FINN full build → PYNQ-Z1 板端
```

链路里每一环都可能拒绝一个"精度很好"的模型。

## 量化模型设计

`QuantFPGAFashionCNN` 用 Brevitas 量化层（QuantConv2d / QuantLinear / QuantReLU）
替换标准层，原则：

- 可配置输入/权重/激活比特宽度（4W4A、2W2A、1W1A）
- **首层与末层允许更高精度**（保持数值稳定）
- 优先使用 Conv / BN / ReLU / Pool / Linear 等 FINN 易处理算子

## FINN-Friendly 结构约束（重点）

第一次真实 FINN estimate 就暴露出模型含以下**对数据流化不利**的算子：

```
MaxPoolNHWC、GlobalAveragePool、Flatten、MatMul、Mul、Add
```

结构因此被重构：

| 约束 | 改法 |
|---|---|
| `downsample_mode = stride_conv` | 用步长卷积替代 MaxPool |
| `head_type = conv` | 用卷积头替代 AdaptiveAvgPool + Linear |
| 架构更名 | `finnconv_direct_head8`（8 个 1×1 卷积通道输出 logits） |

**代价：精度 93.49% → 93.55%，几乎不损失。**

这是"部署约束前置"最直接的例证：**如果一开始就知道 FINN 不支持这些算子，就不用
重做一遍。**

注意这里精度的变化方向 —— 重构后反而高了 0.06 个点。所以"FINN-friendly 会牺牲
精度"这个直觉在本案中并不成立。真正的代价是工程时间，不是 accuracy。

## 资源粗估与候选筛选

在真正进入 FINN full build 之前，先用权重存储粗估筛候选：

| 模型变体 | 权重存储估计 | 相对 BRAM 压力 |
|---|---:|---|
| `q_s` | ~70.5 KiB | 低 |
| `q_m` | ~356.4 KiB | 中 |
| `q_md` | ~591.7 KiB | 中高 |
| `q_md_head8` | ~429.7 KiB | 中 |
| `q_l` | ~870.2 KiB | **高，超过 PYNQ-Z1 可接受范围** |

在约 630 KB 级别 BRAM 的背景下，`q_l` 被**提前排除**，`q_md` 系列成为主候选。

> **这套前置筛选避免了在注定不可行的候选上浪费数小时的 full build。**

这是本项目里投入产出比最高的一个做法：花几分钟做粗估，省下几小时综合。`q_l` 就是
这样被排除的 —— 一次 full build 都没浪费在它身上。

## 主候选实验结果

| 模型 | 量化位宽 | 测试精度 |
|---|---|---:|
| `q_md_w4a4_baseline` | 4W4A | 93.61% |
| `q_md_w4a4_kdkl_t4` | 4W4A + KD | 93.45% |
| `finnconv_direct_head8_baseline` | 4W4A | 93.49% |
| `finnconv_direct_head8_kd_noncheat` | 4W4A + KD | **93.55%** |
| `finnconv_direct_head8_kdkl_t2` | 4W4A + KD | 93.37% |
| `finnconv_direct_head8_w2a2_kdkl` | 2W2A + KD | — |
| `paper_bnnpynq_lfc_1w1a` | 1W1A | 83.16% |

### 三个可读出的结论

**1. KL 蒸馏在这个场景输给 Blend 两次。** `kdkl_t4`（93.45%）和 `kdkl_t2`（93.37%）
都低于 baseline（93.49%）和 Blend 版本（93.55%）。在量化学生上，KL 不是好选择。

**2. 4W4A 基本无损。** 与 FP32 教师 94.77% 相比，4W4A 量化学生 93.55%，差 1.22 个
百分点，而模型体积和算力需求降了一个量级。这是整条链路上性价比最高的一步。

**3. 位宽悬崖在 2W2A 与 1W1A 之间。** 4W4A 几乎无损，2W2A 开始疼（本项目没跑完），
1W1A 是赌博 —— 同在 1W1A 档位，一个 83.16%（成功），一个 ~10%（崩溃）。

## 复现

```bash
pip install -e ".[quant]"
frontier train --config configs/quant/finnconv_head8_w4a4.yaml
```

需要 Brevitas + QONNX 环境。这条路线是本项目**唯一不能纯 CPU 一键复现**的部分，
原因见 [10 - 复现](./10-reproducing.md)。

## 下一步

[05 - FPGA 部署与板端验证](./05-fpga-finn-deployment.md)
