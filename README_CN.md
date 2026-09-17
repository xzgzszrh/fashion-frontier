# Fashion Frontier

面向 **Fashion-MNIST 和 PYNQ-Z1** 的模型训练与部署实验。仓库包含模型结构、训练配置、知识蒸馏、ONNX 导出、量化工具及实验记录。

[English](README.md) · [在线演示](https://xzgzszrh.github.io/fashion-frontier/) · [复现指南](docs/10-reproducing.md) · [PYNQ Runner](https://github.com/xzgzszrh/pynq-runner)

![Fashion-MNIST 模型测试页面](assets/demo.png)

## 包含什么

- 基线 CNN 和 EfficientNet-B0 教师模型。
- 使用知识蒸馏训练的轻量 CNN，以及 FP32、INT8 导出流程。
- PYNQ CPU 测速工具，BNN-PYNQ 和 FINN 路线的实验记录。
- 五个可直接在浏览器运行的 ONNX 权重。

板端界面独立维护在 **[PYNQ Runner](https://github.com/xzgzszrh/pynq-runner)**：加载本地 ONNX 模型、输入张量或图片、查看输出和测量耗时，不依赖本数据集。

## 实验结果

以下 CPU 结果转录自论文表 8。吞吐测试使用 PYNQ-Z1、ONNX Runtime、batch size 256、两个 intra-op 线程和一个 inter-op 线程。

| INT8 学生模型 | 测试准确率 | 板端吞吐 | 10,000 张耗时 |
|---|---:|---:|---:|
| `tinyplus_kd_fulltrain` | 92.21% | 186.8 img/s | 53.5 s |
| `tinyfast_xs` | — | ~310 img/s | — |
| `tinyfast_xxs` | 91.09% | 375.99 img/s | 26.60 s |
| `tinyfast_xxxs` | 90.45% | 445.19 img/s | 22.46 s |

教师模型的测试准确率为 94.77%。轻量学生模型牺牲部分准确率，降低了 CPU 推理开销。

FPGA 实验使用不同的计时范围，单独列出：

- **BNN-PYNQ LFC 1W1A：** 测试准确率 83.16%；batch 256 下加速器吞吐为 154,096 img/s，不包含 Python 打包开销。详见 [BNN 实验](docs/06-bnn-route.md)。
- **FINN 4W4A：** 主机端准确率 93.55%。已生成并运行 bitstream，但板端输出与参考结果不一致。详见 [FINN 部署记录](docs/05-fpga-finn-deployment.md)。

[完整结果表](benchmarks/results.csv)保留了每项结果的来源。这些是历史实验记录，并非重新运行本仓库得到的结果。随仓库提供的 `tinyfast_xxs` 和 `tinyplus` 权重复测记录分别为 91.01% 和 92.19%，与上表的论文值分开保留。网页上的实时耗时则来自访问者自己的设备。

## 运行演示

```bash
git clone https://github.com/xzgzszrh/fashion-frontier.git
cd fashion-frontier
python3 -m http.server 8000
```

打开 **http://localhost:8000/web/**，选择模型和图片，点击 **Predict** 或 **Compare all five**。ONNX Runtime Web 从 CDN 加载，模型和样例图片已包含在仓库中。图片在浏览器内处理。

## 训练与导出

在训练电脑上使用 Python 3.10 或更高版本：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

frontier models
frontier train --config configs/teacher/01_baseline_cnn.yaml
```

运行教师训练、软标签导出、学生训练、ONNX 导出及 INT8 量化：

```bash
make cpu-route
```

分步命令、数据准备和板端测速见[复现指南](docs/10-reproducing.md)。训练代码依据实验整理重建，重新训练不保证得到相同的历史准确率。量化训练可选安装 `pip install -e ".[quant]"`；FPGA 综合需要单独配置工具链。

## 目录

| 目录 | 内容 |
|---|---|
| `src/fashionfrontier/` | 模型、训练、蒸馏、导出和测速代码 |
| `configs/` | 教师、学生、量化和板端配置 |
| `benchmarks/` | 历史结果与来源 |
| `web/` | 浏览器演示、五个 ONNX 权重及样例图片 |
| `docs/` | 实验记录和复现说明 |

相关记录：[教师训练](docs/02-training-from-baseline-to-teacher.md)、[知识蒸馏](docs/03-distillation.md)、[CPU 部署](docs/07-cpu-only-route.md)、[未成功的实验](docs/09-lessons-and-negative-results.md)。

## 开发

```bash
make test
make lint
make web-data     # 从结果表重新生成网页数据
```

测试覆盖模型形状、数据切分、蒸馏损失、ONNX 导出和测速行为。在演示网址后添加 `?selftest=1`，可让五个模型分别运行十张样例；它用于检查推理链路，不等同于完整测试集评估。

采用 MIT 许可证，见 [LICENSE](LICENSE)。引用信息见 [CITATION.cff](CITATION.cff)。
