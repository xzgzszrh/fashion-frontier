<div align="center">

<img src="assets/shot-hero.png" alt="Fashion Frontier —— 五个权重，一块 PYNQ-Z1，每个数字都实测" width="100%">

<h1>Fashion Frontier</h1>

<p><b>同一个数据集，同一块开发板，五种不同的答案 —— 以及支撑每个答案的实测数据。</b></p>

<p>
  <img alt="license" src="https://img.shields.io/badge/license-MIT-2563eb?style=flat-square">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-2563eb?style=flat-square">
  <img alt="tests" src="https://img.shields.io/badge/tests-35%20passing-16a34a?style=flat-square">
  <img alt="dataset" src="https://img.shields.io/badge/dataset-Fashion--MNIST-ea580c?style=flat-square">
  <img alt="target" src="https://img.shields.io/badge/target-PYNQ--Z1-0f172a?style=flat-square">
</p>

<p>
  <a href="https://xzgzszrh.github.io/fashion-frontier/"><b>▶ 在线演示</b></a> ·
  <a href="#直接跑起来看">本地运行</a> ·
  <a href="#交付档位">交付档位</a> ·
  <a href="#六个场景">场景手册</a> ·
  <a href="#失败案例">失败复盘</a> ·
  <a href="README.md"><b>English</b></a>
</p>

</div>

---

## 目录

| | |
|---|---|
| [为什么有这个仓库](#为什么有这个仓库) | 交付的是曲线，不是模型 |
| [流水线](#流水线) | 从 Fashion-MNIST 到上板，端到端 |
| [交付档位](#交付档位) | 四条 CPU 档位，每一条都实测 |
| [三条路线](#三条路线) | 两个闭环，一个诚实的不闭环 |
| [六个场景](#六个场景) | 给定约束，选出模型 |
| [唯一真正重要的方法论](#唯一真正重要的方法论) | 约束优先的设计 |
| [直接跑起来看](#直接跑起来看) | 权重在你浏览器里真跑 |
| [快速开始](#快速开始) | 安装、复现、测速 |
| [仓库结构](#仓库结构) | 目录顺序**就是**流水线顺序 |
| [关于数字的说明](#关于数字的说明) | 为什么只有一个结果文件 |
| [失败案例](#失败案例) | 六个负结果 |
| [诚实的边界](#诚实的边界) | 这个项目不声称什么 |
| [文档索引](#文档索引) · [硬件参考](#硬件参考) · [引用](#引用) | |

---

## 为什么有这个仓库

Fashion-MNIST 是个简单的数据集。在它上面刷到 94.77 % 并不是这个项目的价值。

价值在于：在一块 **双核 650 MHz ARM Cortex-A9 + Artix-7 FPGA** 的板子上，四条 CPU 交付档位和两条加速器路线被推到了可测量的终点 —— 而途中的每一个决策、每一次取舍、每一处走不通，都**在真实硬件上**留下了记录。

所以这里交付的不是一个模型，而是一条**实测的精度-吞吐曲线，以及产生它的完整推理链条**。

<img src="assets/frontier.svg" alt="PYNQ-Z1 上的精度-吞吐权衡曲线：四条 INT8 CPU 档位与一个 1W1A FPGA fabric 点" width="100%">

这条曲线横跨大约**五个数量级的吞吐**。沿着它移动从来不是免费的，而知道**每一步在什么地方不再划算**，就是这个项目的工程含量所在。

## 流水线

六个阶段，每个阶段都有对应的 `configs/` 配置和 `docs/` 文档。下方那条加速器分支是对同一个问题的第二种回答 —— 而它的两条臂中有一条从未闭环。

<img src="assets/pipeline.svg" alt="流水线：Fashion-MNIST → 教师 → 软标签 → 学生 → ONNX/INT8 → PYNQ-Z1，以及 Brevitas QAT 分出的 BNN-PYNQ 与 FINN 两条臂" width="100%">

最值得注意的其实是第二个框：教师**只训一次**，之后每一档学生都从**同一份缓存软标签**蒸馏而来。这正是一次新的学生实验只需 15–40 分钟、而不是重跑一整轮训练的原因。

## 交付档位

四条 CPU 侧档位，全部在真实 PYNQ-Z1 上跑完**完整 10000 张测试集**。静态 INT8，`onnxruntime` 1.16.0，`intra_op_num_threads=2`，`batch_size=256`。

<img src="assets/ladder.svg" alt="四条 CPU 档位按参数量、测试准确率与板端吞吐的对照" width="100%">

| 档位 | 模型 | 测试准确率 | 吞吐 | 10000 张耗时 | 参数量 |
|---|---|---:|---:|---:|---:|
| 精度优先 | `tinyplus_kd_fulltrain` | **92.21 %** | 186.8 img/s | 53.5 s | 658 490 |
| 中间档 | `tinyfast_xs` | — | ~310 img/s | — | 237 658 |
| 均衡档 | `tinyfast_xxs` | 91.09 % | 375.99 img/s | 26.60 s | 182 478 |
| 速度优先 | `tinyfast_xxxs` | 90.45 % | **445.19 img/s** | **22.46 s** | 133 714 |

**放弃 1.76 个准确率点，换回 2.38 倍吞吐。** 这笔交易在什么地方开始不划算，就是本仓库的全部主题。

这四档学生蒸馏自的教师是 **4 020 358 参数** —— 最大的学生档的 6.1 倍，也正是流水线里必须有蒸馏的原因。

两处空缺是刻意留着的，而不是从别处凑一个数填上：

- `tinyfast_xs` **没有公开准确率** —— 论文只记录了它的吞吐。
- `params` 一列对四个 TinyFashionCNN 变体是**由结构算出来的**（并有测试锁定），对 EfficientNet 与 Brevitas 模型则留空，不做估算。

## 三条路线

<img src="assets/routes.svg" alt="三条部署路线及其状态：CPU-only ONNX 与 BNN-PYNQ 闭环，FINN 主线未闭环" width="100%">

| 路线 | 准确率 | 吞吐 | 周期 | 状态 |
|---|---:|---:|---|---|
| **CPU-only ONNX** | 92.21 % | 445 img/s | 数天 | 闭环，四条档位全部交付 |
| **BNN-PYNQ (1W1A)** | 83.16 % | 154 096 img/s | 数周 | 闭环，受准确率限制 |
| **FINN 主线 (4W4A)** | 93.55 %（主机端） | 未取得 | 数周 | bitstream 可运行，数值未闭环 |

第三条路线没有走完。它被如实记录为未完成，而不是包装成一次成功 —— 见 [`docs/05-fpga-finn-deployment.md`](docs/05-fpga-finn-deployment.md)。

## 六个场景

每一行都是真实的约束组合，不是假设。逐场景的完整配置与注意事项：[`docs/08-scenario-playbook.md`](docs/08-scenario-playbook.md)。

<img src="assets/shot-playbook.png" alt="在线演示中渲染出的场景手册与决策树" width="100%">

| # | 场景 | 约束 | 选型 | 预期 |
|---|---|---|---|---|
| 1 | 有交付期限的原型 | > 90 %，1–3 天 | `tinyfast_xxs` | 91 %+ / 半天 |
| 2 | 精度优先 | 尽可能高 | `tinyplus_kd_fulltrain` | 92.21 % / 186.8 img/s |
| 3 | 高吞吐批处理 | 吞吐为王，> 83 % | `bnn_lfc_1w1a` | 83.16 % / 154 096 img/s |
| 4 | 精度**与**吞吐兼得 | > 90 % **且** > 200 img/s | `tinyfast_xxs` | 91.09 % / 375.99 img/s |
| 5 | 快速迭代 | 模型几小时就变一次 | CPU-only + 缓存软标签 | 每轮更新 15–40 分钟 |
| 6 | 未来的高质量主线 | > 93 % 且要快，1–2 周 | FINN 4W4A | 理论 1 000–10 000 img/s |

## 唯一真正重要的方法论

> **把部署约束放在最前面。** 在资源受限的平台上，模型必须被设计成适配部署环境 —— 而不是先训好、再想办法塞进去。

这个项目里有三次结构性重做，都是因为没做到这一点：

| 发生了什么 | 代价 |
|---|---|
| `q_md` 用了 `GlobalAveragePool` / `Flatten` / `MatMul` → 被 FINN 数据流前端拒绝 | 整个拓扑重做 |
| `q_l` 需要约 870 KiB 权重 → 超出 BRAM 预算 | 靠五分钟的资源粗估提前排除，省下数小时的完整构建 |
| EfficientNet-B0 直接部署到 ARM CPU → 约 10 img/s | 整条路线围绕 tiny CNN 宽度重新设计 |

**这三次本来都能避免，只要做一次廉价的预检查。先估算，再构建。**

还有第四条经验，价值和前三条一样高：**如果一条"本该更快"的部署路径反而更慢了，先怀疑链路，不要怀疑模型。** 那个在 FINN fabric 上只跑到 1.227 img/s（比纯 CPU 还慢）的模型，不是模型慢，是链路坏了。

## 直接跑起来看

[`web/`](web/) 是一个静态站点，把本项目交付的五个 ONNX 权重**直接跑在你的浏览器里**（ONNX Runtime Web，WASM，单线程）。不需要构建、没有服务端推理、也不是录屏截图：

```bash
python -m http.server 8000          # 在仓库根目录执行
# 然后打开 http://localhost:8000/web/
```

选一个权重、选一张图片，数字就来自你这台机器 —— 十个类别概率、置信度，以及背后的延迟分布：

<img src="assets/shot-demo.png" alt="在线演示中单张预测后的效果：十个类别的概率条与五张延迟指标卡" width="100%">

点 **Compare all five** 会用同一张图跑完全部五个模型。下面这一屏就是整个项目的论点 —— 94.77 % 的教师是这里最慢的东西，而 90.45 % 的那一档是最快的：

<img src="assets/shot-compare.png" alt="五个权重跑同一张图的对比表，含逐行延迟与正确性" width="100%">

页面读取 `web/data/results.json`，该文件由 `benchmarks/results.csv` 生成，因此数字不可能和权威结果表脱节：

```bash
python scripts/build_web_data.py    # results.csv -> web/data/results.json
```

在 URL 后加 `?selftest=1`，会让每个权重依次跑完十张样例图并打印逐条通过/失败记录。这个页面的核心主张就是"跑的是论文真权重、且在浏览器里依然分类正确"，所以这件事值得能一键自证：

<details>
<summary>本仓库一次 Chromium 运行的真实输出（有删节）</summary>

```text
ort 1.22.0 models 5 samples 10
images loaded 10
load ok tinyfast_xxxs 2306ms
run tinyfast_xxxs s0 OK   truth=T-shirt/top pred=T-shirt/top conf=0.974 ms=0.3
run tinyfast_xxxs s1 OK   truth=Trouser pred=Trouser conf=1.000 ms=0.3
# ... 此处省略 44 行：每个权重对十张样例图的完整结果 ...
load ok efficientnet_b0_transfer_95 91ms
run efficientnet_b0_transfer_95 s6 OK   truth=Shirt pred=Shirt conf=0.513 ms=13.5
run efficientnet_b0_transfer_95 s9 OK   truth=Ankle boot pred=Ankle boot conf=0.979 ms=13.4
TOTAL 50/50 correct in 7.1s
SELFTEST DONE
```

</details>

亚毫秒的模型，会在每个计时样本内部连续跑多次再除以次数 —— 否则 `performance.now()` 的精度比模型本身还粗，中位数报的其实是时钟而不是模型。

## 快速开始

```bash
pip install -e .                       # CPU-only 路线；量化路线用 ".[quant]"

frontier models                        # 宽度阶梯与参数量
frontier results --only-deployed       # 权威结果表
```

端到端复现整条 CPU 交付路线 —— 教师、软标签、四档学生、ONNX 导出、静态 INT8：

```bash
make cpu-route
```

或者逐步执行：

```bash
frontier train --config configs/teacher/02_efficientnet_b0_transfer.yaml

# 教师只跑一次，把软标签缓存下来，之后所有学生共用
frontier export-soft-targets \
    --config configs/teacher/04_export_soft_targets.yaml \
    --checkpoint outputs/efficientnet_b0_transfer_95/best_model.pt \
    --out artifacts/teacher_targets_full_train.npz --split full_train

frontier train --config configs/cpu/04_tinyfast_xxs.yaml
frontier export-onnx --model tiny_fashion_cnn --variant tinyfast_xxs \
    --checkpoint outputs/tinyfast_xxs/best_model.pt --out artifacts/xxs_fp32.onnx
frontier quantize --model artifacts/xxs_fp32.onnx --out artifacts/xxs_int8.onnx
```

在板子本身上（只需 `numpy` 和 `onnxruntime`）：

```bash
pip install -r requirements/board.txt
frontier export-test-set --out artifacts/fashion_mnist_test.npz   # 在主机上跑一次
frontier bench --model-dir artifacts --test-set artifacts/fashion_mnist_test.npz
```

验证 —— 不需要联网、不需要硬件、不需要下载数据集：

```bash
make test      # 35 个测试
make lint
```

其他常用目标：`make demo`（启动浏览器演示）、`make web-data`（由 `results.csv` 重新生成 `web/data/`）、`make readme-figures`（重绘 `assets/` 中的插图）、`make help`（完整列表）。

## 仓库结构

目录顺序**就是**流水线顺序：

```
configs/
  teacher/   01 基线 -> 02 EB0 迁移 -> 03 全量定型 -> 04 导出软标签
  cpu/       00 base -> 01 kd -> 02 fulltrain，然后 03/04/05 三个快速档
  quant/     finnconv_head8_w4a4, bnn_lfc_w1a1        （需 .[quant]）
  board/     pynq_z1_ort.json  —— 实测确定的运行时配置
src/fashionfrontier/
  data       可复现的切分、非作弊守卫、独立的切分随机种子
  models     教师 / TinyFashionCNN 家族 /（可选）Brevitas QAT + BNN
  distill    离线软标签、blend + KL 两种蒸馏损失
  engine     训练循环、EMA、评估
  export     带往返校验的 ONNX 导出、静态 INT8 量化
  bench      ORT 会话工厂、主机端测速
  board      PYNQ-Z1 板端测速（只依赖 numpy + onnxruntime）
benchmarks/  results.csv —— 全仓唯一的权威数字来源
docs/        00..10 —— 每个文件对应流水线的一个阶段
assets/      本 README 中的插图与运行截图
scripts/     build_web_data.py, build_readme_figures.py
web/         浏览器端演示：真实 ONNX 权重、真实推理（见上）
```

## 关于数字的说明

[`benchmarks/results.csv`](benchmarks/results.csv) 是本仓库**唯一**的数字来源文件，每一行都标注了它来自论文的哪一节。它是**转录**，不是重跑。

<img src="assets/shot-results.png" alt="在线演示中渲染出的权威结果表：28 行，每行一次实验，含来源列" width="100%">

这一点很重要，因为项目原本把结果散落在四个地方（训练 `metrics.json`、板端测速 JSON、演示清单、论文），而它们互相矛盾：样本数不一致、同一个模型有两个准确率、还有一次板端数据是由文档里没写的另一个运行时跑出来的。

处理方式：**以论文为唯一权威。** 与论文冲突的数字一律不在此发布。你自己测出来的数据写进 `benchmarks/board_raw/`，并且刻意**不**合并进 `results.csv`。

所以引用数字时请始终带上四要素 —— **设备、样本数、运行时设置、精度**。没有设备标注的吞吐数字没有意义。例如 154 096 img/s 是 `batch_size=256` 下排除 Python 侧打包开销的加速器吞吐，它不是单张流式的速率。

## 失败案例

六个失败案例的完整复盘（含根因）在 [`docs/09-lessons-and-negative-results.md`](docs/09-lessons-and-negative-results.md)。最值得先读的三个：

- **一个崩到约 10 % 的二值学生网络。** 根因不是"蒸馏对二值网络没用"，而是二值拓扑、KL 蒸馏、以及一个被换错形式的交叉熵**同时**被改动。一次只改一个变量。
- **一个 93.5 % 的模型上板只剩 5 %。** 一张分层验证表把故障精确定位到 stitched-IP 集成路径 —— 训练、导出、单节点 RTL 全部 golden。没有这张表，搜索范围会是六个互不相关的子系统。
- **三次冲击 94.77 % 全部未果。** 天花板来自数据集与数据管线，不是网络结构。在花一周做架构搜索之前，这件事值得先知道。

## 诚实的边界

- **FPGA 路线始终没有完成数值闭环。** bitstream 能生成、能加载、能执行；输出与 golden 参考不一致，根因已收缩到 stitched-IP 集成路径。
- **Brevitas/QONNX 训练代码是结构性重建，** 不是原始实验环境的逐行复刻。准确率数字来自论文。代码存在的意义是让这条路线可以被继续走下去，而不是声称在这里复现了原始数字。
- **2W2A 从未跑完，** 也没有为它编造任何数字。
- **Fashion-MNIST 很简单。** 这里真正的价值是跨场景的方法与决策链条，不是数据集本身。

## 文档索引

| | |
|---|---|
| [00](docs/00-overview.md) | 总览 |
| [01](docs/01-platform-and-path-selection.md) | 平台资源、PS/PL 决策框架 |
| [02](docs/02-training-from-baseline-to-teacher.md) | 基线 CNN → 94.77 % 教师 |
| [03](docs/03-distillation.md) | 为什么蒸馏是杠杆，而不只是涨点 |
| [04](docs/04-quantization-and-finn-friendly-design.md) | QAT 与面向 FINN 的结构设计 |
| [05](docs/05-fpga-finn-deployment.md) | FINN 完整构建、分层验证 |
| [06](docs/06-bnn-route.md) | 1W1A 二值路线 |
| [07](docs/07-cpu-only-route.md) | CPU-only 路线与板端调优 |
| [08](docs/08-scenario-playbook.md) | **六场景选型手册** |
| [09](docs/09-lessons-and-negative-results.md) | **负结果与教训** |
| [10](docs/10-reproducing.md) | 复现步骤与预期耗时 |

只有十分钟？读 **08** 和 **09**。

## 硬件参考

| | |
|---|---|
| SoC | Xilinx Zynq-7020 |
| PS | 双核 ARM Cortex-A9 @ 650 MHz |
| PL | Artix-7 XC7Z020（53 200 LUT，106 400 FF，140 BRAM36K，220 DSP） |
| 内存 | 512 MB DDR3，256 KB 片上 SRAM |
| 系统 | PYNQ Linux 3.0，Python 3.10，ONNX Runtime 1.16.0 |

## 引用

如果你使用了本仓库，请引用这个软件；如果你引用了 `benchmarks/results.csv` 中的数字，请引用它所转录自的论文。机器可读的元数据在 [`CITATION.cff`](CITATION.cff)。

```bibtex
@software{fashion_frontier_2026,
  title  = {Fashion Frontier: End-to-End Inference Engineering on PYNQ-Z1},
  author = {xzgzszrh},
  year   = {2026},
  url    = {https://github.com/xzgzszrh/fashion-frontier},
  note   = {面向 Fashion-MNIST 的实测精度/吞吐前沿：四条 CPU 档位与两条加速器路线}
}
```

## 许可证

MIT —— 见 [LICENSE](LICENSE)。
