# CPU-Only ONNX Runtime：完整闭环的现实交付路线

## 路线收敛逻辑

CPU-only 路线最初**并非**本项目的主要目标。但在 FPGA 主线数值正确性尚未闭环的背景
下，它演化成了项目中最完整、最实用、最可交付的成果。

收敛的五个判断：

1. **放弃大模型** —— EfficientNet-B0 在 ARM A9 上约 10 img/s，完全不可接受
2. **聚焦极小 CNN** —— TinyFashionCNN：两层 Conv-BN-ReLU-MaxPool + 全连接隐藏层
3. **用蒸馏弥补结构缩小的精度损失**
4. **静态 INT8 量化**（ONNX Runtime `quantize_static`，QOperator 格式）进一步提速
5. **在真实 PYNQ-Z1 板端实测并调优运行参数**

值得强调的是第 5 点：这条路线是**在板子上调出来的**，不是推断出来的。

## 板端参数调优

在真实 PYNQ-Z1（双核 650 MHz）上，ONNX Runtime 参数对吞吐影响显著。收敛结果：

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2          # 利用双核
opts.inter_op_num_threads = 1
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "model_int8.onnx", opts, providers=["CPUExecutionProvider"]
)

batch = inputs[:256]                    # 大 batch
result = session.run(None, {"input": batch})
```

关键发现：

| 参数 | 效果 |
|---|---|
| `intra_op_num_threads=2` | 相比单线程 **+30–50%** 吞吐 |
| `batch_size=256` | 相比小 batch 明显优势（向量化指令利用率更高） |
| 静态 INT8 量化 | 相比 FP32 **1.5–2×** 吞吐，精度损失可控 |

配置固化在 [`configs/board/pynq_z1_ort.json`](../configs/board/pynq_z1_ort.json)。

## 分层交付结果

完整 10000 张测试集在真实板端的实测：

| 模型变体 | 主机精度 | 板端吞吐 | 完整耗时 | 定位 |
|---|---:|---:|---:|---|
| `tinyplus_kd_fulltrain` | 92.21% | 186.8 img/s | ~53.5 s | 精度优先 |
| `tinyfast_xs` | — | ~310 img/s | — | 中间档 |
| `tinyfast_xxs` | 91.09% | 375.99 img/s | 26.60 s | 速度/精度平衡 |
| `tinyfast_xxxs` | 90.45% | 445.19 img/s | 22.46 s | 速度优先 |

结构差异只在宽度：

| 变体 | conv_channels | hidden_dim | 参数量 |
|---|---|---|---:|
| `tinyplus` | [40, 80] | 160 | 658 490 |
| `tinyfast_xs` | [24, 48] | 96 | 237 658 |
| `tinyfast_xxs` | [22, 44] | 80 | 182 478 |
| `tinyfast_xxxs` | [20, 40] | 64 | 133 714 |

拓扑完全不变，只改宽度 —— 这保证了 ONNX 图结构和板端算子集一致，换档不需要重新
验证部署链路。

## ONNX 推理速度优化清单

### 模型结构

- 输入保持原生 28×28（避免上采样带来的额外计算）
- depthwise separable convolution 可在精度损失极小情况下进一步减少 MACs
- 移除不必要的 BN 层（推理时可融合到卷积权重中）

### 量化策略

- 静态 INT8（QOperator 格式）相比动态量化有 **10%–20%** 额外加速
- 校准集建议 200–500 样本，**本任务最优为约 256**
- 首层和末层保持 FP32 或更高精度可避免精度损失过大

### 运行时配置

- 图优化级别设为 `ORT_ENABLE_ALL`（启用 operator fusion）
- 批量推理时 batch_size 应匹配 NEON/SIMD 向量宽度
- **避免频繁重建 session**（初始化约 100–300 ms，应复用）

最后一条在小 batch 场景下影响巨大：如果每次请求都建 session，300 ms 的初始化会
完全淹没 2 ms 的推理。

## 为什么这条路线赢了

它同时满足了四件事，而另外两条路线各缺一件：

| | CPU-only | BNN | FINN 主线 |
|---|---|---|---|
| 精度可用（>90%） | ✅ 92.21% | ❌ 83.16% | ✅ 93.55% |
| 吞吐可用（>100 img/s） | ✅ 445 | ✅ 154 096 | ❌ 未闭环 |
| 部署周期短 | ✅ 天级 | ❌ 周级 | ❌ 周级 |
| 数值正确 | ✅ | ✅ | ❌ |

**在"什么时候要交付"这个约束下，能跑通的次优解优于跑不通的最优解。**

## 复现

```bash
frontier train --config configs/cpu/04_tinyfast_xxs.yaml
frontier export-onnx --model tiny_fashion_cnn --variant tinyfast_xxs \
    --checkpoint outputs/tinyfast_xxs/best_model.pt --out artifacts/tinyfast_xxs_fp32.onnx
frontier quantize --model artifacts/tinyfast_xxs_fp32.onnx --out artifacts/tinyfast_xxs_int8.onnx
```

## 下一步

[08 - 六场景最佳实践](./08-scenario-playbook.md)
