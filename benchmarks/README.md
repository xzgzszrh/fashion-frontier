# Results: the single authoritative table

`results.csv` is the **only** numbers file in this repository. Every figure in
the README, the docs and the web demo is sourced from it, and every row cites the
paper section it came from.

If a number appears anywhere else in the repo and disagrees with this file, this
file wins.

## Why one table

This project accumulated results in four places over its lifetime: training
`metrics.json` files, board-side benchmark JSONs, a demo dashboard manifest, and
the paper. They did not agree with each other. Sample counts differed (256-image
bundles vs the full 10 000-image test set), the same model carried two different
accuracy figures in two documents, and one board run turned out to have been
produced by a different runtime than the paper described.

The resolution was to stop treating all sources as equal. **The paper is the
authoritative record.** This table is a transcription of it, and nothing that
contradicts the paper is published here.

## Column meanings

| column | meaning |
|---|---|
| `route` | `teacher` / `distill` / `quant` / `cpu` / `bnn` / `fpga` |
| `precision` | `FP32`, `INT8` (post-training static), or `WxAy` (quantisation-aware training) |
| `params` | trainable parameter count. Computed from the model definition, **not** copied from a source that did not report it. Blank where the architecture is not in this repo. |
| `val_accuracy` | best validation accuracy on the 6 000-image val split |
| `test_accuracy` | accuracy on the official 10 000-image Fashion-MNIST test split |
| `board_throughput_img_s` | measured on PYNQ-Z1 unless `device` says otherwise |
| `board_total_time_s` | wall clock for the full 10 000-image test set |
| `device` | `host` (training workstation) or `PYNQ-Z1` |
| `deliverable` | `yes` if this model was actually shipped as a scenario recommendation |
| `source` | paper table or section the row is transcribed from |
| `notes` | the one thing worth knowing about the row |

## Measurement protocol

Any number produced by this repository must state all four of these, or it does
not belong in this table:

1. **Device.** Host and board throughput differ by orders of magnitude. A
   throughput figure without a device label is meaningless.
2. **Sample count.** Screening may use the 256-image bundle; published accuracy
   must use the full 10 000-image test set. It costs 22–54 s on the board —
   there is no reason to shortcut it.
3. **Runtime and settings.** `onnxruntime`, `intra_op=2`, `inter_op=1`,
   `ORT_ENABLE_ALL`, `batch_size=256`. See `configs/board/pynq_z1_ort.json`.
4. **Precision.** FP32 / INT8-QOperator / 4W4A / 1W1A.

## Known gaps, stated plainly

- **`tinyfast_xs` has no published accuracy.** The paper's Table 8 lists
  throughput (~310 img/s) only. The field is left empty rather than filled in
  from another source.
- **`finnconv_direct_head8_w2a2_kdkl` has no result.** The 2W2A branch was not
  carried to completion; the paper records the row without a number.
- **The FPGA route never closed numerically.** See
  [`docs/05-fpga-finn-deployment.md`](../docs/05-fpga-finn-deployment.md). The
  bitstream builds and runs; its output does not match the golden reference.
  That row is in the table marked `deliverable: no`.
- **Parameter counts for the Brevitas models are blank.** The paper does not
  report them and they were not recomputed from a checkpoint. The EfficientNet-B0
  rows *are* filled in (4 020 358) because that architecture lives in this
  repository — the figure comes from `build_model`, not from a checkpoint. Note
  that reading the same model out of its ONNX graph gives 3 999 350, because
  graph folding removes folded-batch-norm constants; the definition is the number
  we publish.

## Regenerating

```bash
frontier results                      # print the table
frontier results --only-deployed      # just the shipped models
frontier results --route cpu          # one route
```

Board numbers can be re-measured with:

```bash
frontier bench --model-dir artifacts/onnx --test-set artifacts/fashion_mnist_test.npz
```

Results land in `benchmarks/board_raw/` as JSON with the device label, Python
version and onnxruntime version attached. They are **not** automatically merged
into `results.csv` — that file only changes when the paper does.
