# Benchmarks

Inference verification on the 7.8M-parameter TinyStories BLT-D checkpoint (embed=256, hidden=512,
2 encoder / 6 global / 2 decoder layers, 600k steps, AdamW, `block_size=4`, entropy patching).

## Headline finding

**At this scale BLT-DV is a net win, but only at the trained block size.** At `B=4`, with the
entropy LM the model was trained with, BLT-DV accepts 65% of drafted bytes and produces output
byte-identical to greedy, at 0.85x the memory bandwidth and 0.89x the wall-clock of greedy
(350 ms vs 394 ms per 64-byte prompt). Adaptive-B, which starts at `B=8` and backs off, recovers
most of that: 35.2% acceptance at 0.99x bandwidth.

Every configuration at `B=8` or above loses badly, because block cells past the trained `B` form a
bidirectional block that L_mask never supervised. BLT-S (self-speculation) is a net loss at every
window size tested. So the honest summary is: verified speculation helps at this scale only where
the draft matches what the model was trained to draft.

## Setup

| | |
|---|---|
| Checkpoints | `runs/tinystories_600k/tinystories_600k.fblt` (7.83M params), `runs/tinystories_300k/tinystories_paper_360k.fblt` |
| Entropy LM | `runs/entropylm/entropy_lm.fblt` (required, see below) |
| Corpus | TinyStories, 171.6 MB train (`data/tinystories/train.bin`), 9.1 MB held out (`data/tinystories/heldout.bin`) |
| Prompts | 8 held-out prompts at offsets `200000 + p*100000`, 64 bytes each, 64 new bytes |
| Quality metrics | causal BPB, masked-cell accuracy |
| Speed metrics | NFE/byte, bandwidth (Eq. 8), wall-clock |

### Passing the entropy LM is not optional

`--entropy-lm` is **required** whenever entropy patching is used. A random-init entropy LM puts
per-byte entropy near `ln(256) = 5.54`, above the 2.5 patch threshold at every position, so every
byte starts its own patch and the model runs on an all-1-byte-patch layout it never saw in
training. `bin/infer` and `infer_bench` both fail hard rather than measure this; `--fixed-patches`
is the one legitimate case that needs no LM.

## Model quality

Two checkpoints from the same run, evaluated on the held-out split
(`train_blt_d --steps 0 --eval-windows 200`, 102,400 bytes):

| checkpoint | causal BPB (held out) |
|---|---:|
| 30k probe | 1.7536 |
| 360k | 1.1611 |
| 600k | **0.9978** |

Quality improves monotonically with training. There is no overfitting: the whole 600k-step run is
1.07 epochs over 171.6 MB, so the model has seen each byte roughly once.

## Inference methods

| Method | How it generates | Cost |
|---|---|---|
| **greedy** (baseline) | one full encoder+global+decoder forward per byte | 1.0 enc and 1.0 dec NFE/byte |
| **BLT-S** (self-speculation) | draft k bytes decoder-only, verify all k with one full forward, accept the longest matching prefix | more decoder NFEs, fewer encoder NFEs |
| **BLT-D** (block diffusion) | start a block of B masked positions, iteratively unmask the most confident ones | cheapest per byte, output diverges from greedy |
| **BLT-DV** (diffusion + verification) | BLT-D drafts, then a causal forward verifies them like BLT-S | output identical to greedy; cheap only when acceptance is high |

## Results

### Bandwidth analysis (Eq. 8)

Memory bandwidth = `b * [N_dec * P_dec + N_enc * (P_enc + P_glob)] / 10^9` GB with `b=2` (fp16).

For this checkpoint, `P_dec = 1,967,872` and `P_enc + P_glob = 5,862,912`, giving a greedy baseline
of **15.66 MB/byte**. Both counts come from `blt_model_visit_params` using the visitor's
`is_dec` / `is_enc || is_glob` flags. (The method reproduces the previously published 2.97M
constants exactly, which is how it was validated.)

#### Entropy patching (default)

| Method | dec/byte | enc/byte | acceptance | agree | BW (MB) | BW vs greedy |
|---|---:|---:|---:|---:|---:|---:|
| greedy | 1.000 | 1.000 | – | 1.000 | 15.66 | 1.000x |
| selfspec k=4 | 2.336 | 0.957 | 28.0% | 1.000 | 20.42 | 1.304x |
| selfspec k=8 | 4.074 | 0.957 | 14.5% | 1.000 | 27.26 | 1.740x |
| selfspec k=16 | 7.189 | 0.957 | 7.7% | 1.000 | 39.52 | 2.523x |
| blockdiff B=4 | 0.838 | 0.508 | – | 0.182 | 9.25 | 0.591x |
| blockdiff B=8 | 0.908 | 0.400 | – | 0.150 | 8.27 | 0.528x |
| blockdiff B=16 | 0.906 | 0.240 | – | 0.135 | 6.38 | 0.408x |
| **blockdv B=4** | **1.043** | **0.787** | **65.0%** | **1.000** | **13.33** | **0.851x** |
| blockdv B=8 | 2.045 | 1.031 | 10.8% | 1.000 | 20.14 | 1.286x |
| blockdv B=16 | 3.283 | 1.055 | 4.9% | 1.000 | 25.29 | 1.615x |
| blockdv onestep (B=8) | 1.109 | 1.109 | 5.6% | 1.000 | 17.37 | 1.109x |
| blockdv eb γ=1.0 (B=8) | 2.145 | 1.031 | 10.9% | 1.000 | 20.53 | 1.311x |
| blockdv eb γ=2.0 (B=8) | 2.004 | 1.033 | 10.8% | 1.000 | 20.00 | 1.277x |
| blockdiff eb (B=8) | 0.984 | 0.357 | – | 0.113 | 8.07 | 0.515x |
| blockdv hetv (B=8) | 2.045 | 1.031 | 10.8% | 1.000 | 20.14 | 1.286x |
| blockdv boundary-aligned (B=8) | 2.219 | 1.082 | 16.8% | 1.000 | 21.42 | 1.368x |
| blockdv adaptive (from B=8) | 1.359 | 0.861 | 35.2% | 1.000 | 15.45 | 0.986x |
| blockdv adaptive + aligned | 1.385 | 0.846 | 36.9% | 1.000 | 15.37 | 0.981x |

Raw BLT-D (`blockdiff`) is the cheapest thing here at 0.41-0.59x bandwidth, but its output
diverges from greedy: agreement is only 0.135-0.182, so it is not a drop-in substitute.

### Wall-clock latency

Mean ms per prompt, 64 new bytes, 8 prompts, RTX 4060.

| Method | entropy patching | fixed-stride-4 |
|---|---:|---:|
| greedy | 394 | 424 |
| selfspec k=4 | 504 | 515 |
| selfspec k=16 | 950 | 986 |
| blockdiff B=4 | 251 | 197 |
| **blockdv B=4** | **350** | 479 |
| blockdv B=8 | 551 | 626 |
| blockdv B=16 | 720 | 861 |
| blockdv adaptive | 408 | 500 |

BLT-DV at `B=4` is the only verified method that beats greedy on both bandwidth and wall-clock.
Wall-clock is not a clean comparison across methods since KV-cache usage and encoder passes
differ, but the direction matches the NFE counts.

![Speculative acceptance rates by method](../graphs/acceptance.png)
*Drafted-byte acceptance by method. BLT-DV at the trained `B=4` leads at 65%; everything at
`B>=8` collapses because those block cells were never supervised by L_mask.*

![NFE vs quality frontier](../graphs/nfe_quality_frontier.png)
*Decoder NFEs/byte against agreement with greedy. `blockdv B=4` sits below and left of greedy:
fewer encoder passes and slightly more decoder passes, for identical output.*

![Encoder/decoder NFE map](../graphs/enc_dec_map.png)
*Encoder and decoder NFEs per byte. Verified methods trade encoder passes for decoder passes.*

![Mean wall-clock latency per prompt](../graphs/latency.png)
*Mean wall-clock ms per prompt (64 new bytes).*

## Acceptance vs the paper

The Fast-BLT paper reports acceptance at 1B-3B params on FR→EN translation. This is 7.8M params on
English story prose, so both scale and domain differ, and the gap below is not attributable to
either alone.

| Config | Paper 3B (FR→EN) | This repro (7.8M, TinyStories) |
|---|---:|---:|
| B=4 | 94.4% | **65.0%** |
| B=8 | 86.3% | 10.8% |
| B=16 | 67.2% | 4.9% |
| B=8, one-step | 84.6% | 5.6% |

**BLT-S acceptance (no paper-comparable B axis):**

| k | This repro |
|---|---:|
| 4 | 28.0% |
| 8 | 14.5% |
| 16 | 7.7% |

`B=4` is the only setting where the two are in the same regime, and there the model reaches 65%
against the paper's 94.4%. The `B=8` and `B=16` rows are not a scale result: the model was
trained at `B=4`, so those rows measure a train/inference mismatch, not a capability limit. A
`B=8` comparison would need a model trained at `B=8`.

## Entropy vs fixed-stride patching

| Patching | blockdv B=4 acceptance | agree | BW vs greedy |
|---|---:|---:|---:|
| Entropy (matches training) | **65.0%** | 1.000 | 0.851x |
| Fixed-stride-4 | 28.7% | 1.000 | 1.107x |

Entropy patching more than doubles acceptance and is the difference between BLT-DV being cheaper
than greedy and being more expensive. This model was trained with entropy patching, so fixed-stride
commit points land mid-patch and every draft byte has to be re-verified.

Note that with fixed-stride patching `blockdiff` agreement drops to 0.088-0.115, versus 0.135-0.182
with entropy patching.

## What changed from the previous numbers

The previous version of this document reported BLT-DV acceptance of 1.6-5.2% and concluded that
"acceptance at this scale is 18-42x lower than the paper, root cause likely the 340x scale gap".
That conclusion was wrong, and the numbers were measuring a broken configuration rather than the
model.

`bench/infer_bench.c` defaulted `--entropy-lm` to NULL, which builds a random-init entropy LM, so
every byte became its own patch. The old rows are still in `bench/results.jsonl` under phase
`6_infer` and match the old tables digit for digit (`bltd_blockdv_B4_a0.70` = 0.0523). Re-running
the same checkpoint with its trained entropy LM gives 0.650.

Two code bugs also contributed and are fixed in PRs #26 (draft pass cross-attended a stale latent
for the patch-closing byte) and #29 (inference defaulted `--block-size` to 8 against a model
trained at 4). Both `bin/infer` and `infer_bench` now hard-fail on a missing entropy LM.

The old checkpoint (`s101.fblt`, 2.97M params) no longer exists, so those specific numbers can
never be re-verified bit-for-bit. This section reports the checkpoint that does exist.

## Reproducing

Build and run:

```bash
cmake -S . -B build-cuda -DUSE_CUDA=ON
cmake --build build-cuda --target infer_bench -j$(nproc)

source scripts/cuda_env.sh   # WSL2 only

./build-cuda/infer_bench --backend cuda \
  --plain runs/tinystories_600k/tinystories_600k.fblt \
  --bltd runs/tinystories_600k/tinystories_600k.fblt \
  --entropy-lm runs/entropylm/entropy_lm.fblt \
  --heldout data/tinystories/heldout.bin \
  --embed 256 --hidden 512 --enc-layers 2 --glob-layers 6 --dec-layers 2 --cross-attn all \
  --prompts 8 --new-bytes 64 \
  --results bench/results_tinystories_10m.jsonl
```

Add `--fixed-patches` for the fixed-stride comparison. `--entropy-lm` and the shape flags are both
mandatory: without them the tool aborts rather than measure a model on a layout it never saw.

Held-out quality:

```bash
./build-cuda/train_blt_d --corpus data/tinystories/train.bin \
  --load-weights runs/tinystories_600k/tinystories_600k.fblt --steps 0 \
  --embed 256 --hidden 512 --enc-layers 2 --glob-layers 6 --dec-layers 2 \
  --entropy-lm runs/entropylm/entropy_lm.fblt --entropy-patches \
  --eval-corpus data/tinystories/heldout.bin --eval-windows 200 --backend cuda
```

Plots:

```bash
pip install matplotlib
python fblt/scripts/bench_plots.py --results bench/results_tinystories_10m.jsonl --out graphs
```

Raw measurements are committed at `bench/results_tinystories_10m.jsonl` (44 rows, phase
`6_infer`). `bench/results.jsonl` is gitignored and holds the full untracked history, including the
superseded broken-configuration run.

## CUDA benchmarks

All numbers from `bench/cuda_bench` on a **desktop NVIDIA GeForce RTX 4060** (AD107, 3072 fp32 cores, 115 W, 8188 MiB GDDR6, driver 616.56). CPU baseline is **single-threaded naive loops** (no OpenMP, no BLAS so really weak).

### Hardware identification

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 (AD107, CC 8.9) |
| VRAM | 8188 MiB GDDR6 |
| Power | 115 W |
| Max SM clock | 3105 MHz |
| fp32 peak (NVIDIA spec) | 15.11 TFLOP/s (3072 cores × 2460 MHz boost × 2) |
| Harness constant (corrected) | 15.11 TFLOP/s (`cuda_bench.c`) |

### Micro benchmarks (bench config: E=64, H=128, L=2)

Pure-op throughput, 3 runs averaged.

| Op | Size | CUDA (mean) | Unit |
|---|---|---|---|---|
| fp32 matmul | 512×512 | 5.87 | TFLOP/s |
| fp32 matmul | 1024×1024 | 6.77 | TFLOP/s | |
| fp32 matmul | 2048×2048 | 6.57 | TFLOP/s | |
| bf16 matmul | 512×512 | 11.2 | TFLOP/s |
| bf16 matmul | 1024×1024 | 18.73 | TFLOP/s | |
| bf16 matmul | 2048×2048 | 20.48 | TFLOP/s | |

fp32 matmul range: 5.87–6.77 TFLOP/s across sizes. bf16/fp32 ratio ranges from 1.9x (512) to 3.1x (2048).

### Meso benchmarks (bench config: E=64, L=2, full forward+backward pipeline)

All times in milliseconds (ms), 3 runs averaged.

| Config | Size | fwd (ms) | bwd (ms) |
|---|---|---|---|
| fp32 | 256 tokens | 4.28 | 15.76 |
| fp32 | 1024 tokens | 13.75 | 62.07 |
| bf16 | 256 tokens | 4.13 | 15.65 |
| bf16 | 1024 tokens | 13.68 | 62.11 |

![CPU vs CUDA fp32 vs CUDA bf16 pipeline timing](../graphs/cuda_speedup.png)
*CPU vs CUDA fp32 vs CUDA bf16 pipeline timing (E=64, H=128, L=2, fixed stride-4 patching). Seq=256: ~10x speedup. Seq=1024: ~24x speedup. Values averaged across multiple benchmark runs.*

### Macro benchmarks, training cell and generation cell

`macro_training_cell`: embed + 2×matmul + layernorm + SwiGLU + add (one full decoder layer, fwd+bwd). `macro_generation_cell`: autoregressive byte generation with KV-cache (naive greedy, 48-byte window). Both measured with `--iters 10`, 3 runs.

The training cell is synthetic and needs no checkpoint. The generation cell does load one, and its
default `--ckpt` path (`runs/followup1_base_s104.fblt`) no longer exists in this repo, so these rows
cannot be reproduced as-is. Its shape is hardcoded to E=192 / H=384 / 2-2-2 with no shape flags, so
it also cannot load the 7.8M TinyStories checkpoint. Any checkpoint of that shape works, since the
measurement is timing-only and weight-independent.

#### Bench config (E=64, H=128)

| Metric | CUDA (mean) | CPU (mean) | Speedup |
|---|---|---|---|
| macro_train_1024 (tokens/s) | 14,872 | 496 | **30.0x** |
| macro_gen_greedy (passes/s) | 221.7 | 6.3 | **35.2x** |

#### Production config (E=192, H=384, 2.97M params)

| Metric | CUDA (mean) | CPU (mean) | Speedup |
|---|---|---|---|
| macro_train_1024 (tokens/s) | 7,024 | 93 | **75.8x** |
| macro_gen_greedy (passes/s) | 217.7 | 6.3 | **34.4x** |

Per-run raw data (production config):

| Run | CUDA train | CUDA gen | CPU train | CPU gen |
|---|---:|---:|---:|---:|
| 1 | 7,004 | 216.2 | 92 | 6.3 |
| 2 | 7,005 | 222.4 | 93 | 6.3 |
| 3 | 7,062 | 214.5 | 93 | 6.4 |

Training speedup scales with model size (30x → 76x from E=64 to E=192) because the larger model better saturates GPU cores. Generation speedup stays flat (~34-35x).

### MFU analysis

| Metric | Value | Against |
|---|---|---|
| fp32 matmul mean (512/1024/2048) | 6.40 TFLOP/s | |
| **fp32 MFU** | **42.4%** | Desktop RTX 4060 spec peak (15.11 TFLOP/s) |
| bf16 matmul (1024+2048 avg) | 19.61 TFLOP/s | Tensor-core path |
| bf16/fp32 ratio at 512 | 1.9x | |
| bf16/fp32 ratio at 1024 | 2.8x | |
| bf16/fp32 ratio at 2048 | 3.1x | |

### Training-step and gradient-clipping overhead

The research config (`W=64`, `E=64`, `H=128`, 2/2/2 layers, AdamW) went from
26.2 ms/step to 15.5-16.1 ms/step on the RTX 4060. The optimizer block is no
longer dominated by gradient bookkeeping:

| Stage | Before (E009) | Current |
|---|---:|---:|
| forward | ~4.8-7.8 ms | 4.0-4.3 ms |
| backward | 12.9 ms | 9.9-10.3 ms |
| optimizer + clipping | 4.9 ms | 1.0-1.1 ms |
| **step** | **26.2 ms** | **15.5-16.1 ms** |

Steady-state numbers from 150-step runs after warmup:

```bash
BLT_TRAIN_TIMING=1 ./build-cuda/train_blt_d \
  --corpus data/train.bin --backend cuda --steps 150 \
  --window 64 --embed 64 --hidden 128 --layers 2 --report-every 1000
```

What changed:

- `blt_cuda_memset` is asynchronous on the default stream.
- `--grad-accum 1` skips the accumulator zero/add/copy/scale entirely, and the
  second gradient buffer is only allocated when accumulation is enabled.
- Gradient clipping computes every per-tensor dot into one device array and
  downloads it once. The array is sized from a counting pass over the parameter
  visitor before any dots launch. The old grow-on-demand path reallocated at
  the 65th parameter tensor (the default model has 67), freed the array holding
  the first 64 sums, and then summed the uninitialized replacement. With 64 of
  the 67 squared contributions dropped, the reported norm and clip factor were
  wrong. After the fix the CUDA norm matches the CPU ground truth: both
  backends report per-step norms in the 100-240 range on this config.
- The clip factor is folded into `sgd_all`/`adamw_all` instead of running a
  separate per-tensor scale pass. Forcing clipping with `--max-norm 0.5`
  raises the optimizer block to about 1.5-1.6 ms/step; without the fold that
  threshold also paid a full extra parameter pass.

### BF16 mixed precision (`--dtype bf16`)

Measured on the same desktop RTX 4060, driver 616.56, CUDA 12.0. `--dtype bf16` keeps FP32
masters and casts only the matmul operand; see `docs/CLI_REFERENCE.md` for the exact scope.

Raw `blt_matmul` throughput (`cuda_bench --backend cuda --iters 20`, square GEMMs, FP32
accumulation for BF16):

| shape | FP32 TFLOP/s | BF16 TFLOP/s | speedup |
|-------|-------------:|-------------:|--------:|
| 512 x 512 | 5.12 | 11.48 | 2.24x |
| 1024 x 1024 | 7.06 | 19.15 | 2.71x |
| 2048 x 2048 | 7.05 | 22.44 | 3.18x |

End-to-end trainer (`BLT_TRAIN_TIMING=1`, steady-state average after 50 warmup steps):

| config | FP32 step | BF16 step | delta |
|--------|----------:|----------:|------:|
| E=512, H=1024, enc/glob/dec 2/6/2, W=512 | 238.0 ms | 234.4 ms | -1.5% |
| E=1024, H=2048, enc/glob/dec 1/2/1, W=512 | 244.7 ms | 235.2 ms | -3.9% |

Commands:

```bash
BLT_TRAIN_TIMING=1 ./build-cuda/train_blt_d \
  --corpus data/tinystories/train.bin --backend cuda --steps 150 \
  --window 512 --embed 1024 --hidden 2048 \
  --enc-layers 1 --glob-layers 2 --dec-layers 1 \
  --scratch-mb 1024 --dtype bf16
```

The forward and input-gradient GEMMs use BF16 tensor cores; the weight-gradient GEMM and the
optimizer stay FP32. At these model sizes the step is launch- and elementwise-bound (1-7% MFU),
so raw GEMM throughput does not translate into a proportional step-time win. The CPU backend
rejects `--dtype bf16`.

Loss parity over 300 steps (same seed and window order, E=64/H=128/2 layers, `data/train.bin`):
mean absolute per-step loss difference 0.18 (max 2.44 on losses in 203-903), no NaN/Inf.

Full design, per-phase numbers, meso results, and limitations: [E010](experiments/E010_bf16_mixed_precision.md).

### FP16 and FP8 (`--dtype fp16|fp8`)

Measured in one run on the same RTX 4060 with CUDA 13.3 / cuBLASLt 13.6 (the BF16 table above
was taken with CUDA 12.0, so do not mix columns between the two). `cuda_bench` default
`--iters 10`, square GEMMs, FP32 accumulation throughout:

| n | FP32 | BF16 | FP16 | FP8-E4M3 | FP8-E5M2 |
|---|-----:|-----:|-----:|---------:|---------:|
| 512 | 5.71 | 11.63 | 11.09 | 4.04 | 4.05 |
| 1024 | 6.88 | 18.60 | 18.70 | 7.43 | 7.86 |
| 2048 | 6.54 | 19.24 | 20.45 | 11.46 | 11.57 |

The FP8 cells run the training recipe, not a bare kernel benchmark: the activation stays FP32
and is quantized on every call, and the weight shadow is stored `[K,N]` and transposed into
scratch for the TN-only cuBLASLt kernel. At n=512 the conversion dominates and the cell is
tens of microseconds long, so run-to-run spread is 20-40%. E5M2 tracks E4M3 because only the
weight encoding differs.

FP8 therefore beats FP32 by ~1.8x at n=2048 and loses to BF16 on Ada. Given that end-to-end
steps at the model sizes here are launch- and elementwise-bound, treat `--dtype fp8` as a
supported capability rather than a speed feature. Scaling rules and the E5M2 x E5M2 constraint
that forces E4M3 activation operands: [MIXED_PRECISION.md](MIXED_PRECISION.md).

## Device table and MFU/MBU

Peak FLOPS and HBM bandwidth come from `csrc/core/device_info.c`. The trainer
looks the running CUDA device name up in that table at startup and reports MFU
per report window plus a final MBU in `RESULT`. `infer_bench` uses the same table
for inference MBU.

Formulas:

- MFU = achieved_tflops / peak_tflops
- MBU = (bytes_moved_gb / time_s) / bandwidth_gbps

### Reference peaks

All values are theoretical dense tensor peaks at boost clock, not measured
throughput. FP8 is only defined for Ada/Hopper (and later); Turing has no FP8
tensor cores, so `NVIDIA T4` carries `fp8 = 0`.

| Device | FP32 TFLOP/s | BF16 TFLOP/s | FP8 TFLOP/s | HBM GB/s | Source |
|---|---:|---:|---:|---:|---|
| NVIDIA GeForce RTX 4060 | 15.1 | 30.2 | 60.4 | 272.0 | [Ada product page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/) |
| NVIDIA GeForce RTX 4060 Ti | 22.1 | 44.1 | 88.2 | 288.0 | [Ada product page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/) |
| NVIDIA GeForce RTX 4070 | 29.1 | 58.3 | 116.5 | 504.0 | [Ada product page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/) |
| NVIDIA GeForce RTX 4070 Ti | 40.1 | 80.1 | 160.1 | 504.0 | [Ada product page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/) |
| NVIDIA GeForce RTX 4080 | 48.7 | 97.5 | 194.9 | 716.8 | [Ada product page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/) |
| NVIDIA GeForce RTX 4090 | 82.6 | 165.2 | 330.3 | 1008.0 | [Ada product page](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/) |
| NVIDIA A100 (SXM4-40GB) | 19.5 | 312.0 | 624.0 | 1555.0 | [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |
| NVIDIA A100-SXM4-80GB | 19.5 | 312.0 | 624.0 | 2039.0 | [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |
| NVIDIA A100-PCIE-40GB | 19.5 | 312.0 | 624.0 | 1555.0 | [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |
| NVIDIA A100-PCIE-80GB / A100 80GB PCIe / generic `A100-PCIE` | 19.5 | 312.0 | 624.0 | 1935.0 | [A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |
| NVIDIA H100 (SXM) | 67.0 | 989.0 | 1979.0 | 3350.0 | [H100 datasheet](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet) |
| NVIDIA H100 PCIe | 51.0 | 756.0 | 1513.0 | 2000.0 | [H100 datasheet](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet) |
| NVIDIA L4 | 30.3 | 121.0 | 242.0 | 300.0 | [L4 product page](https://www.nvidia.com/en-us/data-center/l4/) |
| NVIDIA L40 | 90.5 | 181.0 | 362.0 | 864.0 | [L40 product page](https://www.nvidia.com/en-us/data-center/l40/) |
| NVIDIA L40S | 91.6 | 366.0 | 733.0 | 864.0 | [L40S product page](https://www.nvidia.com/en-us/data-center/l40s/) |
| NVIDIA T4 | 8.1 | 65.0 | 0.0 | 300.0 | [T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf) |

Assumptions:

- Values assume the vendor boost clock and a well-cooled card. Laptop and
  thermally limited SKUs run below these peaks.
- MIG and vGPU instances expose different names (and a fraction of the
  bandwidth/FLOPS). They will usually fall through to "unknown device" rather
  than silently inheriting the full-card peak.
- Power limits and memory clocks move real throughput well below the table, so
  MFU/MBU are upper-bound-normalized ratios, not absolute efficiency.

### Lookup and overrides

`blt_device_lookup` picks the **longest** table name that is a substring of the
reported device name. This is what keeps variants apart: `NVIDIA GeForce RTX
4060 Ti` must not match the shorter `NVIDIA GeForce RTX 4060`, `NVIDIA H100
PCIe` must not match `NVIDIA H100`, and `NVIDIA L40S` must not match `NVIDIA
L4`.

`blt_device_resolve(name, dtype, tflops_override, bw_override)` then resolves
the effective peaks:

1. A positive `--peak-tflops` / `--peak-bandwidth` always wins (even for an
   unknown device).
2. Otherwise the matched entry's dtype column (`fp32`, `bf16` or `fp8`,
   selected from `--fp8` / `--bf16`) and its HBM bandwidth are used.
3. Unknown device with no override → all-zero peaks; the trainer prints
   `n/a` and disables MFU/MBU instead of reporting a misleading `0.0`.

### MBU models

MBU is a **lower bound**: both models count mandatory weight traffic plus a
minimal I/O term, and ignore activations, allocator scratch and KV cache.

- **Training** (`train_blt_d`): per optimizer step the weights are read on the
  forward and backward passes and touched again by the update, gradients are
  written then read, and AdamW additionally reads/writes its two moments. That
  is `param_bytes * 6` for SGD and `param_bytes * 10` for AdamW, multiplied by
  the number of steps. Elapsed time is the wall clock over the training loop.
- **Inference** (`infer_bench`): per prompt,
  `(model_weight_bytes + entropy_lm_weight_bytes) * (decoder_nfe + encoder_nfe)
  + (prompt_len + new_bytes)`. Weights are counted once per forward pass; the
  I/O term covers reading the prompt and writing the generated bytes.

The trainer excludes the optimizer's own state allocation from `param_bytes`
(it counts weights only) and both tools exclude KV-cache growth.

### Adding a device

1. Append an entry to `BLT_DEVICE_TABLE` in `csrc/core/device_info.c`, longest
   distinctive name first is not required (lookup is longest-match), but keep
   variants grouped.
2. Add a row to the table above with the datasheet/product-page link.
3. Add a lookup/resolve assertion in `tests/unit/core/unit_device_info.c`
   covering any prefix ambiguity (e.g. a `Ti` or `PCIe` suffix).
