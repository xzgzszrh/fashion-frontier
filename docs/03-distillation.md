# 知识蒸馏：不只是一个提点技巧

## 为什么需要它

目标从"高精度教师"转向"可部署学生"之后，蒸馏成为连接两者的桥梁。它的价值在三个
层面。

### 精度补偿

在严格结构约束下（轻量网络、量化、FINN-friendly 算子限制），直接训练学生往往比教师
低 2–4 个百分点。

蒸馏提供的软标签包含**类间相似性信息** —— 比如 T-shirt 与 Shirt 的混淆分布。这使
学生能在**不增加参数量**的前提下，学到比 one-hot 标签更丰富的类间关系。

Fashion-MNIST 恰好是个对这点敏感的数据集：T-shirt / Pullover / Coat / Shirt 四类
之间混淆严重，one-hot 标签浪费了这部分信息。

### 快速复制教师能力

一旦建立了教师和离线 soft target 文件（`teacher_targets_*.npz`），**后续任何结构
变体的学生都可以直接消费同一套软标签**，无需每次重新运行教师模型。

这才是蒸馏在这个项目里最重要的作用 —— 见下节。

### 架构无关性

本项目的蒸馏支持从 EfficientNet-B0 向任意结构学生（轻量 CNN、量化 CNN、BNN LFC）
传递知识。教师与学生架构可以完全不同，不需要像 online KD 那样同步运行大模型。

## 两种损失

**Blend distillation**

$$L = \alpha \cdot L_{CE}(\hat{y}, y) + (1-\alpha) \cdot L_{CE}(\hat{y}, p_T)$$

**KL distillation**

$$L = \alpha \cdot L_{CE}(\hat{y}, y) + (1-\alpha) \cdot T^2 \cdot KL(p_S^{(T)} \| p_T^{(T)})$$

其中 $p_T$ 为教师软标签，$T$ 为温度。

**选哪个：脆弱模型一律用 Blend。** 本项目唯一的蒸馏灾难（1W1A BNN 崩到 ~10%）用的
是 KL。事后判断是 KL 注入的梯度信号与 sign 激活冲突。详见
[09 - 负结果](./09-lessons-and-negative-results.md)。

## 效果

| 模型 | 蒸馏 | 备注 | 测试精度 |
|---|---|---|---:|
| `cnn_m_baseline` | 无 | 轻量 student 基线 | ~91.5% |
| `cnn_m_kd` | Blend | + 蒸馏 | ~92.3% |
| `q_l_w4a4_base` | 无 | 量化 student 基线 | 93.58% |
| `q_l_w4a4_kd` | Blend | + 蒸馏 | 93.69% |
| `finnconv_h8_base` | 无 | FINN-friendly 基线 | 93.49% |
| `finnconv_h8_kd` | Blend | + 蒸馏 | 93.55% |
| `tinyplus_kd_full` | Blend | CPU-only + 蒸馏 | 92.21% |

绝对增益 0.06%–0.8%，看起来不起眼。**但这个一致性本身就是价值所在**：

> 在模型结构和资源预算已经被部署约束锁定的情况下，蒸馏是在**不改变架构、不改变
> 部署开销**前提下稳定提升精度的手段。

换个说法：当结构已经不能再动时，蒸馏是唯一还能动的旋钮。

## 真正的杠杆：一次建立教师，多路复用软标签

这是本项目最值得复制的工作流。

建立高质量教师 + 离线 soft target 之后：

- **FPGA 高量化路线**：直接启动 QAT + 蒸馏，得到 FINN-friendly 量化学生
- **CPU-only 路线**：同一 soft target 文件，仅更换学生结构（TinyFashionCNN 族），
  快速得到多个速度/精度档位
- **BNN 路线**：Blend 蒸馏作为精度提升的保守选项（但要先稳定纯净训练）

边际开发成本因此大幅下降：**教师只跑一次，学生可以无限次。**

对需要多种速度/精度档位的边缘部署场景，这一条比蒸馏本身的 +0.5% 重要得多。

## 工程实现

```python
from fashionfrontier.distill.soft_targets import export_soft_targets, SoftTargetSet

# 主机端，一次
export_soft_targets(
    teacher, loader, device,
    "artifacts/teacher_targets_full_train.npz",
    indices=list(range(60000)),
    temperature=1.0,
)

# 之后任意学生训练，配置里加一段即可
# distillation:
#   teacher_targets_path: artifacts/teacher_targets_full_train.npz
#   method: blend
#   alpha: 0.35
```

NPZ 布局（`indices` / `labels` / `logits` / `probabilities` / `temperature`）：

```
indices       int64    数据集位置，与 loader 顺序无关地对齐
labels        int64    ground truth
logits        float32  教师 logits（未缩放）
probabilities float32  softmax(logits / temperature)
temperature   float32  上面用的温度
```

**为什么要存 `indices`**：训练 loader 是 shuffle 的，绝不能靠 batch 顺序对齐软标签。
代码里 `IndexedSubset` 让每个样本自带数据集位置，soft target 按位置查表。这个细节
处理错了会得到"看起来在蒸馏、实际在学噪声"的结果，而且很难 debug。

## 下一步

[04 - 量化与 FINN-friendly 设计](./04-quantization-and-finn-friendly-design.md)
