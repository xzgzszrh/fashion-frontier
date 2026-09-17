# 复现步骤

## 环境

```bash
git clone https://github.com/xzgzszrh/fashion-frontier.git && cd fashion-frontier
python -m venv .venv && source .venv/bin/activate
pip install -e .              # CPU-only 路线
# pip install -e ".[quant]"   # 额外：Brevitas 量化 / BNN 路线
# pip install -e ".[dev]"     # 额外：测试与 lint
```

| 依赖 | 版本要求 | 用途 |
|---|---|---|
| Python | ≥ 3.10 | 板端 PYNQ 自带 3.10 |
| torch / torchvision | ≥ 2.0 | 训练 |
| onnx / onnxruntime | ≥ 1.16 | 导出、量化、测速 |
| brevitas + qonnx | 可选 | `[quant]` extra，量化与 BNN 路线 |

## 一键跑通 CPU-only 交付路线

```bash
make cpu-route
```

等价于以下步骤：

```bash
# 1. 基线（可选，用于验证环境）        ~2 min  (Apple MPS)
frontier train --config configs/teacher/01_baseline_cnn.yaml

# 2. 教师：EfficientNet-B0 迁移学习      ~40 min (GPU) / 数小时 (CPU)
frontier train --config configs/teacher/02_efficientnet_b0_transfer.yaml

# 3. 导出软标签（教师只跑这一次）        ~3 min
frontier export-soft-targets \
    --config configs/teacher/04_export_soft_targets.yaml \
    --checkpoint outputs/efficientnet_b0_transfer_95/best_model.pt \
    --out artifacts/teacher_targets_full_train.npz \
    --split full_train --temperature 1.0

# 4. 学生：四档各训一次                 每档 10–30 min
frontier train --config configs/cpu/00_tinyplus_base.yaml
frontier train --config configs/cpu/01_tinyplus_kd.yaml
frontier train --config configs/cpu/02_tinyplus_kd_fulltrain.yaml
frontier train --config configs/cpu/04_tinyfast_xxs.yaml
frontier train --config configs/cpu/05_tinyfast_xxxs.yaml

# 5. 导出 ONNX（带 torch↔ORT 往返校验）
frontier export-onnx --model tiny_fashion_cnn --variant tinyfast_xxs \
    --checkpoint outputs/tinyfast_xxs/best_model.pt \
    --out artifacts/tinyfast_xxs_fp32.onnx

# 6. 静态 INT8 量化（256 样本校准）
frontier quantize --model artifacts/tinyfast_xxs_fp32.onnx \
    --out artifacts/tinyfast_xxs_int8.onnx --calib-size 256
```

## 板端测速

```bash
# 主机端：导出测试集 NPZ
frontier export-test-set --out artifacts/fashion_mnist_test.npz

# 拷到板子
scp artifacts/fashion_mnist_test.npz artifacts/*_int8.onnx xilinx@<board>:~/fashionfrontier/

# 板端（只需 numpy + onnxruntime）
pip install -r requirements/board.txt
frontier bench --model-dir artifacts --test-set artifacts/fashion_mnist_test.npz \
    --out benchmarks/board_raw/$(date +%Y%m%d)_pynq.json
```

推荐直接用脚本：

```bash
./scripts/deploy_to_board.sh <board-ip>
```

## 预期结果

**精度**（主机端测试集）：

| 模型 | 论文值 | 复现预期 |
|---|---:|---|
| `baseline_cnn` | 91.74% | ±0.5 |
| `efficientnet_b0_transfer_95` | 94.77% | ±0.3 |
| `tinyplus_kd_fulltrain` | 92.21% | ±0.5 |
| `tinyfast_xxs` | 91.09% | ±0.5 |
| `tinyfast_xxxs` | 90.45% | ±0.5 |

**吞吐**：只有在 PYNQ-Z1 上才有意义。主机端测出的数字与论文不可比。

### 关于复现精度的诚实说明

本仓库的代码是**结构等价的重建**，不是原实验环境的逐行复刻。影响复现精度的因素：

- 原始训练在 Apple MPS（主机）与 Windows + CUDA（量化）上完成，本仓库代码主要路径
  在 Linux/macOS/CPU 与 CUDA 上验证
- Brevitas / QONNX / FINN 版本演进会改变量化行为
- 随机种子固定了，但不同 torch 版本的算子实现差异会累积

**因此：精度数字以论文为准，代码用于让别人能接着往下走，而不是用于逐位复现。**

`benchmarks/results.csv` 是论文的转录，不会因为本地重跑而改变。

## 测试

```bash
make test          # pytest
make lint          # ruff
```

测试覆盖：模型结构契约、蒸馏损失数值、ONNX 往返一致性、benchmark 自洽性。全部不依赖
网络或硬件，CI 里能跑。

## 量化与 BNN 路线

```bash
pip install -e ".[quant]"
frontier train --config configs/quant/finnconv_head8_w4a4.yaml
frontier train --config configs/quant/bnn_lfc_w1a1.yaml
```

⚠️ 这两条路线的**训练**可在普通环境跑，但要走到板端还需要 FINN 工具链 +
Vivado/Vitis，只能在 Linux + Docker（官方推荐）或打了补丁的 Windows 环境下完整
执行。相关经验见 [05](./05-fpga-finn-deployment.md)。

## Docker

```bash
docker build -t fashion-frontier -f docker/Dockerfile .
docker run --rm -it fashion-frontier frontier results
```
