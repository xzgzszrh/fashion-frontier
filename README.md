<div align="center">

<img src="assets/shot-hero.png" alt="Fashion Frontier — five checkpoints, one PYNQ-Z1, training and deployment experiments" width="100%">

<h1>Fashion Frontier</h1>

<p><b>Fashion-MNIST on PYNQ-Z1: training, compression and measured deployment trade-offs.</b></p>

<p>
  <img alt="license" src="https://img.shields.io/badge/license-MIT-2563eb?style=flat-square">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-2563eb?style=flat-square">
  <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/xzgzszrh/fashion-frontier/ci.yml?branch=main&label=CI&style=flat-square">
  <img alt="dataset" src="https://img.shields.io/badge/dataset-Fashion--MNIST-ea580c?style=flat-square">
  <img alt="target" src="https://img.shields.io/badge/target-PYNQ--Z1-0f172a?style=flat-square">
</p>

<p>
  <a href="https://xzgzszrh.github.io/fashion-frontier/"><b>Live demo</b></a> ·
  <a href="#see-it-run">Run it locally</a> ·
  <a href="#model-tiers">Delivered tiers</a> ·
  <a href="#scenario-selection">Scenario playbook</a> ·
  <a href="#unsuccessful-experiments">What failed</a> ·
  <a href="README_CN.md"><b>中文</b></a>
</p>

</div>

---

## Contents

| | |
|---|---|
| [Overview](#overview) | Accuracy and throughput across deployment routes |
| [The pipeline](#the-pipeline) | Training, distillation, quantization and deployment |
| [Model tiers](#model-tiers) | Four INT8 CPU students |
| [Three routes](#three-routes) | CPU, BNN-PYNQ and FINN |
| [Scenario selection](#scenario-selection) | Accuracy, throughput and iteration constraints |
| [See it run](#see-it-run) | Five ONNX checkpoints in the browser |
| [Quick start](#quick-start) | Install, train, export and benchmark |
| [Measurement notes](#measurement-notes) · [Unsuccessful experiments](#unsuccessful-experiments) · [Documentation](#documentation) | Sources and experiment records |

---

## Overview

Fashion Frontier asks a practical question: **which Fashion-MNIST models meet an accuracy target while fitting the inference budget of a PYNQ-Z1?**

The project trains an EfficientNet-B0 teacher, then uses knowledge distillation and quantization to produce smaller students. CPU and FPGA experiments compare accuracy, throughput and deployment effort.

<img src="assets/frontier.svg" alt="Accuracy and throughput across INT8 CPU students, an FP32 reference and a binary FPGA route" width="100%">

The teacher reaches 94.77% test accuracy. CPU students cover 90.45%–92.21%, with throughput up to 445.19 img/s. The binary BNN-PYNQ model reaches higher accelerator throughput at 83.16% accuracy. CPU and FPGA timings have different scopes; see [measurement notes](#measurement-notes).

## The pipeline

The CPU route covers teacher training, soft-target export, student training, ONNX export and INT8 quantization. The FPGA branches explore binary networks and quantized architectures that fit FINN's dataflow toolchain.

<img src="assets/pipeline.svg" alt="Fashion-MNIST training and CPU deployment, with BNN-PYNQ and FINN branches" width="100%">

Cached teacher outputs can be reused across student experiments. Changing a student then requires retraining and exporting that student, without rerunning the teacher. Stage-specific configurations live in `configs/`; commands are listed in the [reproduction guide](docs/10-reproducing.md).

## Model tiers

Four CPU students use different network widths. The figure compares their parameter counts and the board results recorded in the paper.

<img src="assets/ladder.svg" alt="Parameter counts, accuracy and throughput of the four CPU tiers" width="100%">

| Tier | Model | Test accuracy | Throughput | 10,000-image time | Parameters |
|---|---|---:|---:|---:|---:|
| Accuracy | `tinyplus_kd_fulltrain` | **92.21%** | 186.8 img/s | 53.5 s | 658,490 |
| Middle | `tinyfast_xs` | — | ~310 img/s | — | 237,658 |
| Balanced | `tinyfast_xxs` | 91.09% | 375.99 img/s | 26.60 s | 182,478 |
| Speed | `tinyfast_xxxs` | 90.45% | **445.19 img/s** | **22.46 s** | 133,714 |

CPU timing used ONNX Runtime 1.16.0, `intra_op=2`, `inter_op=1`, batch 256 and the 10,000-image test set. Accuracy and throughput are transcribed from Table 8; parameter counts are computed from model definitions. No accuracy was published for `tinyfast_xs`.

TinyPlus and XXXS differ by **1.76 accuracy points** and approximately **2.38× throughput**. TinyPlus is a useful starting point when accuracy matters more; XXS and XXXS offer smaller CPU inference costs.

## Three routes

<img src="assets/routes.svg" alt="Steps and completion status of the CPU ONNX, BNN-PYNQ and FINN routes" width="100%">

| Route | Accuracy | Throughput | Result |
|---|---:|---:|---|
| **CPU-only ONNX · INT8** | 90.45%–92.21% | 186.8–445.19 img/s | Board records for multiple students |
| **BNN-PYNQ · 1W1A** | 83.16% | 154,096 img/s | Batch-256 accelerator execution recorded |
| **FINN · 4W4A** | 93.55% on host | No validated result | Bitstream runs; outputs differ from the reference |

The CPU maxima in the figure belong to different models and should not be combined into one model's performance. The FINN build and numerical checks are documented in the [deployment notes](docs/05-fpga-finn-deployment.md).

## Scenario selection

A useful starting point is the workload: its accuracy requirement, acceptable batch size and model update frequency. The original demo presented these choices in a scenario panel and decision tree.

<img src="assets/shot-playbook.png" alt="Original demo scenario panel and model-selection decision tree" width="100%">

| Requirement | Route to evaluate | What to check |
|---|---|---|
| A working board prototype | CPU ONNX · `tinyfast_xxs` | Runtime availability and input preprocessing |
| Higher CPU-route accuracy | `tinyplus_kd_fulltrain` | Whether its recorded 186.8 img/s is sufficient |
| High batch throughput | BNN-PYNQ · 1W1A | Whether 83.16% accuracy and batch 256 fit the task |
| >90% accuracy and >200 img/s | `tinyfast_xxs` or `tinyfast_xxxs` | Actual batch size, end-to-end latency and error distribution |
| Frequent model updates | CPU ONNX and cached soft targets | Student training, export and quantization costs |
| Further high-accuracy FPGA work | FINN · 4W4A | Board numerical validation remains unresolved |

The screenshot contains early planning estimates, not delivery guarantees. Its projected FINN throughput has not been measured. See the [scenario guide](docs/08-scenario-playbook.md) for configurations and discussion.

## Deployment lessons

Several problems affected the deployment work:

| Observation | Follow-up |
|---|---|
| Some pooling, flattening and matrix operations did not map directly to FINN dataflow | Redesign the network for the supported operator set |
| Larger weight memories exceeded the BRAM budget | Estimate resources before full synthesis |
| The teacher reached only about 10 img/s on the ARM CPU | Use smaller students for CPU deployment |
| A runnable bitstream returned outputs that differed from the reference | Compare export, node RTL, stitched-IP and board execution separately |

The [experiment notes](docs/09-lessons-and-negative-results.md) retain these investigations for related debugging work.

## See it run

**[Open the live demo](https://xzgzszrh.github.io/fashion-frontier/)** and select a checkpoint and image to inspect predictions, class scores and local inference timings. All five [ONNX checkpoints](web/models/) are included in the repository.

<img src="assets/shot-demo.png" alt="Original demo: a single prediction, ten class scores and inference timing" width="100%">

**Compare all five** runs the same image through each model in sequence. The teacher and students use their own image size and normalization settings.

<img src="assets/shot-compare.png" alt="Original demo: five checkpoints evaluated on the same image" width="100%">

These screenshots preserve the original interface; the live page can change between versions. Inference uses ONNX Runtime Web with single-threaded WASM. Timings come from the visitor's device, not the PYNQ board.

To serve the demo locally from the repository root:

```bash
python3 -m http.server 8000
```

Open <http://localhost:8000/web/>. Checkpoints and samples are served locally; ONNX Runtime Web loads from a CDN. Append `?selftest=1` to run all five models over the ten bundled samples as an inference smoke test.

## Quick start

### Install

Use Python 3.10+ on the training computer. These commands use a macOS or Linux shell:

```bash
git clone https://github.com/xzgzszrh/fashion-frontier.git
cd fashion-frontier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

frontier models
frontier results --only-deployed
```

### Train, distill and export

Run the full CPU workflow:

```bash
make cpu-route
```

Or run teacher training, soft-target export and one student experiment separately:

```bash
frontier train --config configs/teacher/02_efficientnet_b0_transfer.yaml

frontier export-soft-targets \
  --config configs/teacher/04_export_soft_targets.yaml \
  --checkpoint outputs/efficientnet_b0_transfer_95/best_model.pt \
  --out artifacts/teacher_targets_full_train.npz --split full_train

frontier train --config configs/cpu/04_tinyfast_xxs.yaml
frontier export-onnx --model tiny_fashion_cnn --variant tinyfast_xxs \
  --checkpoint outputs/tinyfast_xxs/best_model.pt --out artifacts/xxs_fp32.onnx
frontier quantize --model artifacts/xxs_fp32.onnx --out artifacts/xxs_int8.onnx
```

### Benchmark on the board

Export the test set on the host, then deploy to a board environment that already has NumPy and ONNX Runtime:

```bash
frontier export-test-set --out artifacts/fashion_mnist_test.npz
BOARD_PYTHON=/path/to/board/python ./scripts/deploy_to_board.sh <board-ip>
```

`BOARD_PYTHON` selects the interpreter on the board. The script checks dependencies, copies inference code, ONNX models and test data, then retrieves results into `benchmarks/board_raw/`. See the [setup guide](docs/10-reproducing.md) for ARMv7 runtime preparation.

For a browser interface to board-side inference, use the separate **[PYNQ Runner](https://github.com/xzgzszrh/pynq-runner)** project.

### Development checks

```bash
make test
make lint
make web-data
```

Tests cover model structure, distillation losses, ONNX export and benchmark behavior. Run `make help` for additional commands.

## Repository layout

```text
configs/
  teacher/   Baselines, teacher training and soft-target export
  cpu/       Compact students and distillation configurations
  quant/     Brevitas QAT and BNN configurations
  board/     Board runtime settings
src/fashionfrontier/
  data/      Dataset loading and train/validation splits
  models/    Baseline, teacher, student and quantized networks
  distill/   Soft targets, blend and KL distillation
  engine/    Training loop, EMA and evaluation
  export/    ONNX export and INT8 quantization
  bench/     Runtime sessions and host measurement
  board/     Board evaluation and reports
benchmarks/  Historical results and sources
web/         Browser inference, ONNX checkpoints and samples
assets/      Figures and interface screenshots
docs/        Experiment records and reproduction guide
```

## Measurement notes

[`benchmarks/results.csv`](benchmarks/results.csv) transcribes the paper's results, with a source reference for each row. The original demo displayed them in this table:

<img src="assets/shot-results.png" alt="Original demo results table with model, precision, device and source columns" width="100%">

Three kinds of measurements are kept separate:

- **Paper records:** historical results in this README and `benchmarks/results.csv`.
- **Checkpoint verification:** the bundled TinyPlus and XXS artifacts were recorded at 92.19% and 91.01%, compared with paper values of 92.21% and 91.09%. Their metadata is in `scripts/build_web_data.py`.
- **New runs:** browser timings belong to the current computer; new board reports go to `benchmarks/board_raw/`.

BNN's 154,096 img/s is batch-256 accelerator throughput excluding Python packing overhead. Its full-test-set accuracy is 83.16%; 214 correct predictions in an early 256-image board bundle give 83.59%. Those evaluations use different sample sets.

## Unsuccessful experiments

The repository also records experiments that did not reach their targets:

- **A binary student fell to about 10% accuracy.** Topology, distillation and loss formulation had changed together; they need to be tested independently.
- **FINN board outputs differed from host results.** Layered tests narrowed further investigation to the stitched-IP integration path.
- **Several teacher refinements did not exceed 94.77%.** Validation gains did not consistently carry over to the test set; configurations and results are retained.

See [unsuccessful experiments and debugging notes](docs/09-lessons-and-negative-results.md).

## Reproduction scope

CPU, quantized and BNN training code was reconstructed from the experiments. New training runs are not guaranteed to reproduce historical results exactly. FINN 4W4A has not completed board numerical validation; the 2W2A experiment also has no complete result. FPGA synthesis requires a separate FINN and Vivado/Vitis toolchain.

## Documentation

| | |
|---|---|
| [00](docs/00-overview.md) | Project overview |
| [01](docs/01-platform-and-path-selection.md) | Platform resources and PS/PL route selection |
| [02](docs/02-training-from-baseline-to-teacher.md) | Baseline and teacher training |
| [03](docs/03-distillation.md) | Knowledge distillation experiments |
| [04](docs/04-quantization-and-finn-friendly-design.md) | QAT and FINN-compatible design |
| [05](docs/05-fpga-finn-deployment.md) | FINN build and layered validation |
| [06](docs/06-bnn-route.md) | BNN-PYNQ binary route |
| [07](docs/07-cpu-only-route.md) | CPU deployment and tuning |
| [08](docs/08-scenario-playbook.md) | Scenario selection |
| [09](docs/09-lessons-and-negative-results.md) | Unsuccessful experiments and debugging |
| [10](docs/10-reproducing.md) | Reproduction steps |

## Hardware reference

| | |
|---|---|
| SoC | Xilinx Zynq-7020 |
| PS | dual-core ARM Cortex-A9 @ 650 MHz |
| PL | Artix-7 XC7Z020 (53 200 LUT, 106 400 FF, 140 BRAM36K, 220 DSP) |
| Memory | 512 MB DDR3, 256 KB on-chip SRAM |
| OS | PYNQ Linux 3.0, Python 3.10, ONNX Runtime 1.16.0 |

## Citation

If you use this repository, cite the software; if you take a number from
`benchmarks/results.csv`, cite the paper it was transcribed from. Machine-readable
metadata is in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{fashion_frontier_2026,
  title  = {Fashion Frontier: End-to-End Inference Engineering on PYNQ-Z1},
  author = {xzgzszrh},
  year   = {2026},
  url    = {https://github.com/xzgzszrh/fashion-frontier},
  note   = {Measured accuracy/throughput frontier for Fashion-MNIST across four CPU
            tiers and two accelerator routes}
}
```

## License

MIT. See [LICENSE](LICENSE).
