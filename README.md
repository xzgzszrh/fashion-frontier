<div align="center">

<img src="assets/shot-hero.png" alt="Fashion Frontier — five checkpoints, one PYNQ-Z1, every number measured" width="100%">

<h1>Fashion Frontier</h1>

<p><b>Same dataset. Same board. Five different answers — and the measurements to back them up.</b></p>

<p>
  <img alt="license" src="https://img.shields.io/badge/license-MIT-2563eb?style=flat-square">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-2563eb?style=flat-square">
  <img alt="tests" src="https://img.shields.io/badge/tests-35%20passing-16a34a?style=flat-square">
  <img alt="dataset" src="https://img.shields.io/badge/dataset-Fashion--MNIST-ea580c?style=flat-square">
  <img alt="target" src="https://img.shields.io/badge/target-PYNQ--Z1-0f172a?style=flat-square">
</p>

<p>
  <a href="https://xzgzszrh.github.io/fashion-frontier/"><b>▶ Live demo</b></a> ·
  <a href="#see-it-run">Run it locally</a> ·
  <a href="#the-delivered-tiers">Delivered tiers</a> ·
  <a href="#six-scenarios">Scenario playbook</a> ·
  <a href="#what-did-not-work">What failed</a> ·
  <a href="README_CN.md"><b>中文</b></a>
</p>

</div>

---

## Contents

| | |
|---|---|
| [Why this repository exists](#why-this-repository-exists) | the curve, not the model |
| [The pipeline](#the-pipeline) | Fashion-MNIST → board, end to end |
| [The delivered tiers](#the-delivered-tiers) | four CPU tiers, every one measured |
| [Three routes](#three-routes) | two closed loops and one honest failure |
| [Six scenarios](#six-scenarios) | given a constraint, pick a model |
| [The one method that matters](#the-one-method-that-matters) | constraint-first design |
| [See it run](#see-it-run) | the checkpoints run in your browser |
| [Quick start](#quick-start) | install, reproduce, benchmark |
| [Repository layout](#repository-layout) | the directory order *is* the pipeline order |
| [A note on the numbers](#a-note-on-the-numbers) | why there is only one results file |
| [What did not work](#what-did-not-work) | six negative results |
| [Honest boundaries](#honest-boundaries) | what this project does not claim |
| [Documentation](#documentation) · [Hardware](#hardware-reference) · [Citation](#citation) | |

---

## Why this repository exists

Fashion-MNIST is an easy dataset. Getting 94.77 % on it is not the achievement.

The achievement is this: on a board with a **dual-core 650 MHz ARM Cortex-A9 and an
Artix-7 FPGA**, four CPU delivery tiers and two accelerator routes were pushed to a
measured endpoint — and every decision, trade-off and dead end along the way was
recorded **on the actual hardware**.

The result is not one model. It is a **measured accuracy/throughput curve, and the
reasoning that produced it**.

<img src="assets/frontier.svg" alt="Accuracy versus throughput on PYNQ-Z1: four INT8 CPU tiers and one 1W1A FPGA fabric point" width="100%">

That curve spans roughly **five orders of magnitude in throughput**. Moving along it is
never free, and knowing *where* each step stops paying off is the engineering content of
this repository.

## The pipeline

Six stages, each with a config under `configs/` and a document under `docs/`. The
accelerator branch below it is a second answer to the same question — and one of its two
arms never closed.

<img src="assets/pipeline.svg" alt="Pipeline: Fashion-MNIST to teacher to soft targets to students to ONNX/INT8 to the PYNQ-Z1, plus the Brevitas QAT branch into BNN-PYNQ and FINN" width="100%">

The thing worth noticing is the second box: the teacher is trained **once**, and every
student tier is distilled from the *same cached soft targets*. That is what makes a new
student a 15–40 minute experiment instead of a full retraining run.

## The delivered tiers

Four CPU-side tiers, every one measured on a real PYNQ-Z1 over the **full 10 000-image
test set**. Static INT8, `onnxruntime` 1.16.0, `intra_op_num_threads=2`, `batch_size=256`.

<img src="assets/ladder.svg" alt="The four CPU tiers compared by parameter count, test accuracy and board throughput" width="100%">

| tier | model | test accuracy | throughput | 10 000 images | params |
|---|---|---:|---:|---:|---:|
| accuracy-first | `tinyplus_kd_fulltrain` | **92.21 %** | 186.8 img/s | 53.5 s | 658 490 |
| middle | `tinyfast_xs` | — | ~310 img/s | — | 237 658 |
| balanced | `tinyfast_xxs` | 91.09 % | 375.99 img/s | 26.60 s | 182 478 |
| speed-first | `tinyfast_xxxs` | 90.45 % | **445.19 img/s** | **22.46 s** | 133 714 |

**Trading 1.76 accuracy points buys 2.38× throughput.** Where that trade stops being
worth making is the whole subject here.

The teacher all four were distilled from is **4 020 358 parameters** — 6.1× the largest
student tier, and the reason distillation is in the pipeline at all.

Two deliberate gaps, left visible rather than filled in from a convenient source:

- `tinyfast_xs` has **no published accuracy** — the paper records its throughput only.
- The `params` column is **computed from the architecture** for the four TinyFashionCNN
  variants (and locked by a test), and left blank for the EfficientNet and Brevitas
  models rather than estimated.

## Three routes

<img src="assets/routes.svg" alt="Three deployment routes: CPU-only ONNX and BNN-PYNQ closed, FINN mainline not closed" width="100%">

| route | accuracy | throughput | cycle | status |
|---|---:|---:|---|---|
| **CPU-only ONNX** | 92.21 % | 445 img/s | days | closed loop, four tiers delivered |
| **BNN-PYNQ (1W1A)** | 83.16 % | 154 096 img/s | weeks | closed loop, accuracy-limited |
| **FINN mainline (4W4A)** | 93.55 % (host) | not reached | weeks | bitstream runs, numerics never closed |

The third route did not finish. It is documented as unfinished rather than packaged as a
success — see [`docs/05-fpga-finn-deployment.md`](docs/05-fpga-finn-deployment.md).

## Six scenarios

Each row is a real constraint set, not a hypothetical. Full configuration and caveats per
scenario: [`docs/08-scenario-playbook.md`](docs/08-scenario-playbook.md).

<img src="assets/shot-playbook.png" alt="The scenario playbook and the decision tree as rendered in the live demo" width="100%">

| # | scenario | constraint | pick | expected |
|---|---|---|---|---|
| 1 | prototype on a deadline | > 90 %, 1–3 days | `tinyfast_xxs` | 91 %+ / half a day |
| 2 | accuracy-first | as high as possible | `tinyplus_kd_fulltrain` | 92.21 % / 186.8 img/s |
| 3 | high-throughput batch | throughput is king, > 83 % | `bnn_lfc_1w1a` | 83.16 % / 154 096 img/s |
| 4 | accuracy **and** throughput | > 90 % **and** > 200 img/s | `tinyfast_xxs` | 91.09 % / 375.99 img/s |
| 5 | rapid iteration | model changes every few hours | CPU-only + cached soft targets | 15–40 min per update |
| 6 | future high-quality mainline | > 93 % and fast, 1–2 weeks | FINN 4W4A | theoretical 1 000–10 000 img/s |

## The one method that matters

> **Put the deployment constraint first.** On a resource-limited platform the model must
> be designed to fit the deployment — not trained first and adapted to it later.

Three structural rewrites in this project came from learning that the hard way:

| what happened | cost |
|---|---|
| `q_md` used `GlobalAveragePool` / `Flatten` / `MatMul` → rejected by the FINN dataflow frontend | full topology rebuild |
| `q_l` needed ~870 KiB of weights → over the BRAM budget | avoided by a five-minute estimate, saving hours of builds |
| EfficientNet-B0 deployed straight to the ARM CPU → ~10 img/s | a whole route redesigned around tiny CNN widths |

**Every one of them was avoidable with a cheap pre-check. Estimate before you build.**

A fourth lesson is worth as much as the other three: **if a deployment path that should be
faster turns out slower, suspect the pipeline before the model.** The FINN-fabric model
that ran at 1.227 img/s — slower than plain CPU — was not a slow model, it was a broken
link.

## See it run

[`web/`](web/) is a static site that runs the project's five shipped ONNX checkpoints **in
your browser**, on ONNX Runtime Web (WASM, single-threaded). No build step, no server-side
inference, no screenshots:

```bash
python -m http.server 8000          # from the repository root
# then open http://localhost:8000/web/
```

Pick a checkpoint, pick an image, and the numbers come from your machine. Ten
probabilities, a confidence, and the latency distribution behind it:

<img src="assets/shot-demo.png" alt="The demo page after a single prediction: probability bars for all ten classes and five latency cards" width="100%">

**Compare all five** runs that same image through every model. This table is the entire
argument of the project in one screen — the 94.77 % teacher is the slowest thing here by a
wide margin, and a 90.45 % tier is the fastest:

<img src="assets/shot-compare.png" alt="All five checkpoints run on one image, with per-row latency and correctness" width="100%">

The page loads `web/data/results.json`, which is generated from `benchmarks/results.csv`
so the figures cannot drift from the authoritative table:

```bash
python scripts/build_web_data.py    # results.csv -> web/data/results.json
```

Append `?selftest=1` to the URL to run every checkpoint against all ten sample images and
print a pass/fail transcript. The page's central claim is that these are the paper's real
weights and that they still classify correctly in a browser, so it is worth being able to
check that without clicking anything:

<details>
<summary>Real transcript from a Chromium run in this repository (abridged)</summary>

```text
ort 1.22.0 models 5 samples 10
images loaded 10
load ok tinyfast_xxxs 2306ms
run tinyfast_xxxs s0 OK   truth=T-shirt/top pred=T-shirt/top conf=0.974 ms=0.3
run tinyfast_xxxs s1 OK   truth=Trouser pred=Trouser conf=1.000 ms=0.3
# ... 44 lines omitted: every checkpoint against all ten samples ...
load ok efficientnet_b0_transfer_95 91ms
run efficientnet_b0_transfer_95 s6 OK   truth=Shirt pred=Shirt conf=0.513 ms=13.5
run efficientnet_b0_transfer_95 s9 OK   truth=Ankle boot pred=Ankle boot conf=0.979 ms=13.4
TOTAL 50/50 correct in 7.1s
SELFTEST DONE
```

</details>

Sub-millisecond models are timed by repeating the inference inside each sample and
dividing — `performance.now()` is coarser than the work otherwise, and the median would
report the clock rather than the model.

## Quick start

```bash
pip install -e .                       # CPU-only route; add ".[quant]" for Brevitas/QONNX

frontier models                        # the width ladder and parameter counts
frontier results --only-deployed       # the authoritative table
```

Reproduce the whole CPU delivery route end to end — teacher, soft targets, four student
tiers, ONNX export, static INT8:

```bash
make cpu-route
```

Or step by step:

```bash
frontier train --config configs/teacher/02_efficientnet_b0_transfer.yaml

# run the teacher ONCE, cache its soft labels, reuse them for every student
frontier export-soft-targets \
    --config configs/teacher/04_export_soft_targets.yaml \
    --checkpoint outputs/efficientnet_b0_transfer_95/best_model.pt \
    --out artifacts/teacher_targets_full_train.npz --split full_train

frontier train --config configs/cpu/04_tinyfast_xxs.yaml
frontier export-onnx --model tiny_fashion_cnn --variant tinyfast_xxs \
    --checkpoint outputs/tinyfast_xxs/best_model.pt --out artifacts/xxs_fp32.onnx
frontier quantize --model artifacts/xxs_fp32.onnx --out artifacts/xxs_int8.onnx
```

On the board itself — which needs nothing but `numpy` and `onnxruntime`:

```bash
pip install -r requirements/board.txt
frontier export-test-set --out artifacts/fashion_mnist_test.npz   # on the host, once
frontier bench --model-dir artifacts --test-set artifacts/fashion_mnist_test.npz
```

Verification — no network, no hardware, no dataset download:

```bash
make test      # 35 tests
make lint
```

Other useful targets: `make demo` (serve the browser demo), `make web-data` (regenerate
`web/data/` from `results.csv`), `make readme-figures` (redraw the diagrams in `assets/`),
`make help` (the full list).

## Repository layout

The directory order *is* the pipeline order:

```
configs/
  teacher/   01 baseline -> 02 EB0 transfer -> 03 fulltrain -> 04 export soft targets
  cpu/       00 base -> 01 kd -> 02 fulltrain, then 03/04/05 the fast tiers
  quant/     finnconv_head8_w4a4, bnn_lfc_w1a1        (needs .[quant])
  board/     pynq_z1_ort.json  -- the measured runtime configuration
src/fashionfrontier/
  data       reproducible splits, non-cheat guard, decoupled split seed
  models     teacher / TinyFashionCNN family / (optional) Brevitas QAT + BNN
  distill    offline soft targets, blend + KL losses
  engine     training loop, EMA, evaluation
  export     ONNX export with round-trip check, static INT8 quantisation
  bench      ORT session factory, host measurement
  board      PYNQ-Z1 board measurement (numpy + onnxruntime only)
benchmarks/  results.csv -- the single authoritative numbers file
docs/        00..10 -- one document per stage of the pipeline
assets/      the figures and screenshots in this README
scripts/     build_web_data.py, build_readme_figures.py
web/         the in-browser demo: real ONNX checkpoints, real inference (see above)
```

## A note on the numbers

[`benchmarks/results.csv`](benchmarks/results.csv) is the **only** numbers file in this
repository, and every row cites the paper section it came from. It is a transcription,
not a regeneration.

<img src="assets/shot-results.png" alt="The authoritative table as rendered in the live demo: 28 rows, one per experiment, with source column" width="100%">

This matters because the project originally accumulated results in four places
(training `metrics.json`, board benchmark JSONs, a demo manifest, and the paper) — and
they disagreed: different sample counts, two accuracy figures for the same model, and
one board run produced by a different runtime than documented.

The resolution: **the paper is authoritative.** Nothing that contradicts it is
published here. Numbers you measure yourself land in `benchmarks/board_raw/` and are
deliberately *not* merged into `results.csv`.

So always quote all four together — **device, sample count, runtime settings,
precision**. A throughput figure without a device label is a number without a meaning.
The 154 096 img/s figure, for instance, is a `batch_size=256` accelerator measurement
with Python-side packing excluded; it is not a streaming single-image rate.

## What did not work

Six failure cases are written up in full, with root causes, in
[`docs/09-lessons-and-negative-results.md`](docs/09-lessons-and-negative-results.md).
The three worth reading first:

- **A binary student that collapsed to ~10 %.** Not because distillation fails on
  binary networks, but because a binary topology, KL distillation and a swapped
  cross-entropy formulation were all changed **at once**. Change one variable at a time.
- **A 93.5 % model that scored 5 % on the board.** A layer-by-layer verification table
  localised the fault to the stitched-IP integration path — training, export and
  single-node RTL were all golden. Without that table the search space would have been
  six unrelated subsystems.
- **Three separate attempts to beat 94.77 % that all fell short.** The ceiling was the
  dataset and the data pipeline, not the architectures. That is worth knowing before
  spending a week on architecture search.

## Honest boundaries

- **The FPGA route never closed numerically.** The bitstream generates, loads and
  executes; its output does not match the golden reference. Root cause isolated to the
  stitched-IP integration path.
- **The Brevitas/QONNX training code is a structural reconstruction,** not a
  line-by-line replica of the original experiment environment. Accuracy figures come
  from the paper. The code exists so the route can be continued, not to claim the
  original numbers were regenerated here.
- **2W2A was never completed,** and no number is invented for it.
- **Fashion-MNIST is easy.** The value here is the cross-scenario method and the
  decision chain, not the dataset.

## Documentation

| | |
|---|---|
| [00](docs/00-overview.md) | overview |
| [01](docs/01-platform-and-path-selection.md) | platform resources, the PS/PL decision framework |
| [02](docs/02-training-from-baseline-to-teacher.md) | baseline CNN → 94.77 % teacher |
| [03](docs/03-distillation.md) | why distillation is leverage, not just accuracy |
| [04](docs/04-quantization-and-finn-friendly-design.md) | QAT and FINN-friendly design |
| [05](docs/05-fpga-finn-deployment.md) | FINN full build, layered verification |
| [06](docs/06-bnn-route.md) | the 1W1A binary route |
| [07](docs/07-cpu-only-route.md) | the CPU-only route and board tuning |
| [08](docs/08-scenario-playbook.md) | **the six-scenario playbook** |
| [09](docs/09-lessons-and-negative-results.md) | **negative results and lessons** |
| [10](docs/10-reproducing.md) | reproduction steps and expected runtime |

Ten minutes to spare? Read **08** and **09**.

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

MIT — see [LICENSE](LICENSE).
