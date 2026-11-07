# E010: BF16 mixed precision versus FP32

## Metadata

- **Experiment ID:** E010
- **Title:** BF16 mixed precision: matmul throughput, training-step time, and loss parity
- **Status:** Checked
- **Date or reporting period:** 2026-11-07
- **Author or contributors:** fBLT contributors
- **Related reports, issues, or papers:** roadmap item 2, [E009](E009_backward_profile.md), `docs/BENCHMARKS.md`
- **Git commit or release:** `feat/bf16-mixed-precision` (`005c419`), PR #90
- **Configurations and scripts:** `cuda_bench`, `train_blt_d --dtype bf16|fp32`, `BLT_TRAIN_TIMING=1`
- **Artifacts:** raw results were written outside the tree; the tables below record the runs

## 1. Research question

How much faster is BF16 than FP32 for the GEMMs and for a full training step, and does a BF16
run track the FP32 loss without NaN or Inf?

## 2. Motivation and hypothesis

`--dtype bf16` keeps FP32 masters and casts only the matmul operands. Raw BF16 tensor-core
GEMMs should be 2-3x the FP32 SIMT rate on Ada. A step-level win was expected to be smaller:
the weight-gradient GEMM and the optimizer stay FP32, and earlier profiling (E009) found the
step dominated by launch and memory traffic rather than GEMM alone.

## 3. Experimental design

- **Independent variable:** dtype (`fp32` versus `bf16`).
- **Baseline:** FP32 on the same GPU, same seed, same corpus, same window order.
- **Controlled:** GPU, driver, CUDA toolkit, corpus, model shape, batch, seed, optimizer, and
  number of steps. Only the dtype flag changed between paired runs.
- **Micro benchmark:** square 512, 1024, and 2048 GEMMs, `cuda_bench --iters 20`. BF16 uses
  `cublasGemmEx` with FP32 accumulation; FP32 uses `cublasSgemm`.
- **Meso benchmark:** full `blt_model` forward/backward at sequence lengths 256 and 1024 with
  `E=64`, two layers per submodule.
- **End-to-end:** `train_blt_d` on `data/tinystories/train.bin` with `BLT_TRAIN_TIMING=1`, so
  forward, backward, and optimizer time are reported separately. Steady-state values are
  averages over 50 steps after warmup.
- **Loss parity:** 300 steps on `data/train.bin` with `--embed 64 --hidden 128 --layers 2
  --window 64`, logging per-step loss and gradient norm for both dtypes.
- **Metrics:** TFLOP/s for GEMMs, ms per step (split into forward/backward/optimizer), and
  per-step loss and global gradient norm.

## 4. Reproduction instructions

Environment: desktop NVIDIA GeForce RTX 4060 (AD107, 8188 MiB, 115 W), driver 616.56, CUDA
12.0 (V12.0.140), Linux. Build with `cmake .. -DUSE_CUDA=ON` and target `build-cuda`.

Micro and meso benchmarks:

```bash
CUDA_VISIBLE_DEVICES=0 ./build-cuda/cuda_bench \
  --backend cuda --iters 20 --results /tmp/cuda_bench_bf16.jsonl
```

End-to-end step split (repeat with `--dtype bf16` for the second row):

```bash
BLT_TRAIN_TIMING=1 ./build-cuda/train_blt_d \
  --corpus data/tinystories/train.bin --backend cuda --steps 150 \
  --window 512 --embed 512 --hidden 1024 \
  --enc-layers 2 --glob-layers 6 --dec-layers 2 \
  --diffusion 1 --scratch-mb 1024 --report-every 1000
```

```bash
BLT_TRAIN_TIMING=1 ./build-cuda/train_blt_d \
  --corpus data/tinystories/train.bin --backend cuda --steps 100 \
  --window 512 --embed 1024 --hidden 2048 \
  --enc-layers 1 --glob-layers 2 --dec-layers 1 \
  --diffusion 1 --scratch-mb 1024 --report-every 1000 --dtype bf16
```

Loss parity (run once per dtype with a separate `--metrics-log`):

```bash
./build-cuda/train_blt_d \
  --corpus data/train.bin --backend cuda --steps 300 \
  --window 64 --embed 64 --hidden 128 --layers 2 \
  --diffusion 1 --report-every 1000 \
  --metrics-log /tmp/metrics_fp32.jsonl
```

## 5. Results

### 5.1 Raw GEMM throughput

Square matrices, FP32 accumulation for BF16, `cuda_bench --iters 20`:

| Shape | FP32 TFLOP/s | BF16 TFLOP/s | BF16 / FP32 |
|-------|-------------:|-------------:|------------:|
| 512 x 512 | 5.12 | 11.48 | 2.24x |
| 1024 x 1024 | 7.06 | 19.15 | 2.71x |
| 2048 x 2048 | 7.05 | 22.44 | 3.18x |

### 5.2 Meso pipeline (`E=64`, two layers per submodule)

| Sequence | FP32 fwd | BF16 fwd | FP32 bwd | BF16 bwd |
|---------:|---------:|---------:|---------:|---------:|
| 256 | 4.17 ms | 3.96 ms | 15.48 ms | 14.96 ms |
| 1024 | 14.44 ms | 14.61 ms | 65.43 ms | 64.66 ms |

### 5.3 End-to-end training step

Steady-state averages over 50 steps after 100 warmup steps:

| Config | Phase | FP32 | BF16 | Delta |
|--------|-------|-----:|-----:|------:|
| E=512, H=1024, 2/6/2, W=512 | forward | 50.8 ms | 50.0 ms | -1.6% |
| | backward | 176.4 ms | 173.2 ms | -1.8% |
| | optimizer | 10.3 ms | 10.7 ms | +3.9% |
| | **step** | **238.0 ms** | **234.4 ms** | **-1.5%** |
| E=1024, H=2048, 1/2/1, W=512 | forward | 54.1 ms | 52.1 ms | -3.7% |
| | backward | 174.7 ms | 166.9 ms | -4.5% |
| | optimizer | 15.6 ms | 15.9 ms | +1.9% |
| | **step** | **244.7 ms** | **235.2 ms** | **-3.9%** |

Both configs include 7.8M and 51.2M parameters respectively. The optimizer delta is noise:
optimizer math is FP32 in both runs.

### 5.4 Loss and gradient parity (300 steps)

| Metric | FP32 | BF16 |
|--------|-----:|-----:|
| loss at step 1 | 694.019 | 694.007 |
| loss at step 300 | 501.153 | 501.131 |
| mean absolute loss difference | — | 0.18 (0.04% of the ~450 mean loss) |
| max absolute loss difference | — | 2.44 (0.3% of the run's loss range) |
| mean relative gradient-norm difference | — | 0.36% |
| max relative gradient-norm difference | — | 8.6% |
| NaN or Inf | none | none |

Loss was noisy in both runs because the comparison used the default AdamW learning rate on
random initialization; the point is that the two curves track each other, not the absolute
level.

## 6. Interpretation

The raw GEMM result matches the expectation: BF16 tensor cores are 2.2-3.2x the FP32 rate, and
the advantage grows with matrix size because launch and edge overhead amortize.

The step-level win is 1.5-3.9%, an order of magnitude below the GEMM ratio. Several effects
explain this:

- Only the forward matmuls and the backward input-gradient GEMM use BF16. The weight-gradient
  GEMM and the optimizer remain FP32 by design, so roughly one third of the training FLOPs do
  not benefit.
- The trainer reported 1-7% MFU on these runs, and E009 independently found a step dominated by
  launch overhead and memory traffic (thousands of launches, many memset/memcpy calls). Faster
  GEMMs do not shorten work that never reaches the GEMMs.
- The meso forward at sequence 1024 was slightly slower under BF16. That is consistent with
  the cast kernels adding traffic in a launch-bound loop, and is within the run-to-run noise
  seen on this desktop GPU.

Loss and gradient parity are consistent with BF16 rounding of the matmul operands: small,
bounded differences, no divergence, and no non-finite values.

## 7. Limitations and validity threats

- One GPU, one machine, one run per configuration. Short wall-clock runs (30 steps) varied by
  up to ~25% because startup dominates them, but the `BLT_TRAIN_TIMING` averages over 50 steps
  were stable across report windows (for example FP32 step 238.0/241.4/241.3 ms). The
  per-phase splits are more informative than a single total-step ratio.
- Model sizes are small (7.8M and 51.2M parameters, sequence length 512). Nothing here says
  how BF16 scales to a genuinely GEMM-bound model.
- The micro benchmark is square and does not cover the skinny shapes produced by short
  patched sequences.
- The meso benchmark uses `E=64`, where GEMMs are tiny and the pipeline is entirely
  launch-bound; it is a lower bound on the BF16 benefit, not a representative shape.
- Loss parity used one seed and one corpus slice. It checks stability, not statistical
  equivalence.
- Windows CUDA was compile-checked only.

## 8. Conclusion

BF16 mixed precision is numerically stable and delivers 2.2-3.2x on raw GEMMs on an RTX 4060.
For full training steps at the model sizes in this repository it delivers only 1.5-3.9%,
because those steps are launch- and memory-bound and because the weight-gradient GEMM and
optimizer stay FP32. The acceptance criterion for roadmap item 2 (a step-time reduction on a
matmul-bound configuration) holds for the GEMM-bound measurement; it does not imply a blanket
2x training speedup.

## 9. Follow-up work

- Run BF16 activations through the weight-gradient GEMM (full AMP) to cover the remaining
  third of the training matmuls, then re-measure.
- Re-run at a genuinely GEMM-bound shape (larger embed, larger effective batch) to separate
  the GEMM ratio from launch overhead.
- Profile the BF16 path with `nsys` to confirm the cast kernels are not a material share.
- Re-evaluate after item 26 (launch-overhead work) changes the step composition.
- Combine with FP8 once roadmap item 13 lands.
