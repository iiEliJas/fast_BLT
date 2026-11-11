# Mixed precision

`--dtype` narrows GEMM operands only. Master weights, gradients, optimizer state, activations,
normalization, softmax, cross-entropy, and checkpoints stay FP32 in every mode. A run is
therefore numerically "FP32 with rounded weights" or "FP32 with rounded GEMM inputs", never a
fully narrow pipeline.

## Supported dtypes

| `--dtype` | Encoding | Operator requirement | Weight shadow |
|-----------|----------|----------------------|---------------|
| `fp32` | IEEE FP32 | any | none (no shadows allocated) |
| `bf16` | bfloat16 | CUDA, compute capability >= 8.0 | 2 bytes/weight |
| `fp16` | IEEE FP16 | CUDA, compute capability >= 7.0 | 2 bytes/weight |
| `fp8` | E4M3 (`--fp8-format e4m3`, default) | CUDA, compute capability >= 8.9 | 1 byte/weight + one FP32 scale per tensor |
| `fp8` | E5M2 (`--fp8-format e5m2`) | CUDA, compute capability >= 8.9 | 1 byte/weight + one FP32 scale per tensor |

`--bf16` and `--fp8` remain as aliases. The gates are checked before the first step and the
trainer exits with an error rather than falling back to FP32: a silent fallback would report
MFU against the wrong peak. `--fp8-format` is rejected unless an FP8 dtype is selected.

The CPU backend has no low-precision GEMM, so `train_blt_d` rejects every non-FP32 dtype
there. The CPU op implementations do understand all dtypes bit-exactly, which is what the
parity tests use as ground truth.

## What gets narrowed

The local encoder, the global transformer stack, and the local decoder each keep one shadow
per matmul weight: QKV, attention projection, FFN up/gate/down, the encoder-side query and the
decoder-side key/value/output cross-attention projections, and the decoder `lm_head`. Every
other weight (embeddings, norms, the d0 embedding) is consumed by elementwise or gather code
and stays FP32.

The forward GEMM multiplies FP32 activations by the shadow weight. In the backward pass the
input-gradient GEMM reads the same shadow, and the weight-gradient GEMM narrows the FP32
activation it multiplies; the incoming gradient is staged in the shadow dtype for both. Every
accumulator and every gradient buffer stays FP32 -- only the multiplicands are rounded -- so
the optimizer still updates FP32 masters. `blt_model_refresh_mixed` rebuilds every shadow from
the masters once after model creation or `--load-weights` and again after each optimizer step.

`bf16`/`fp16` narrow both operands of a mixed GEMM: the CUDA backend casts the FP32 activation
or gradient into scratch and issues `cublasGemmEx` with FP32 accumulation.

## FP8 scaling

FP8 is per-tensor symmetric. A weight shadow stores one FP32 scale next to its bytes:

```
scale = max(|W|) / fp8_max            fp8_max = 448 (E4M3), 57344 (E5M2)
q     = fp8(round(W / scale))         dequant(q) = q * scale
```

The scale is recomputed at every refresh from the current FP32 master. Activations and the
incoming gradient do not have a persistent scale: `blt_matmul` computes a fresh amax for each
call, quantizes into scratch, and passes both per-tensor scales to cuBLASLt through
`CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` / `B_SCALE_POINTER` with `CUBLAS_COMPUTE_32F`. The kernel
applies the product of the scales to the FP32 accumulator, so no rescaling pass is needed. Do
not set `CUBLASLT_MATMUL_DESC_AMAX_D_POINTER` with an FP32 destination on sm_89: every call
returns `CUBLAS_STATUS_INVALID_VALUE`.

## E5M2 pairing constraint

cuBLASLt on Ada has no E5M2 x E5M2 kernel. A probe across 240 combinations (four dtype
pairings, four orientations, five scale configurations, three shapes) on an RTX 4060 /
cuBLASLt 13.6 found:

- `E5M2 x E5M2` — `CUBLAS_STATUS_NOT_SUPPORTED` for every orientation, shape, and scale setup
  (60/60).
- `E4M3 x E5M2` and `E5M2 x E4M3` — supported in the TN orientation at every shape tested.
- `NT` and `NN` — never supported for any FP8 pairing; `TT` only at `m == 1`, where it is the
  same kernel as TN.

So an E5M2 weight is paired with an E4M3 activation and an E4M3 gradient: `blt_fp8_pair_fmt`
maps the weight encoding to the dynamic operand encoding (`E4M3 -> E4M3`, `E5M2 -> E4M3`).
Consequences:

- `--dtype fp8 --fp8-format e5m2` stores and uses E5M2 **weights**; activations and gradients
  are E4M3. There is no configuration that puts E5M2 on both operands.
- E5M2 weights carry two mantissa bits. Use them to widen dynamic range, not as a faster E4M3:
  expect a visibly different loss trajectory.
- If a caller hands both operands in E5M2 (only reachable through the raw ops, not through the
  trainer), both backends fall back to dequantize + SGEMM. Results stay correct, the run gets
  slower.

## Fallbacks

These are correctness fallbacks, not performance paths. All of them produce the same numbers
as the tensor-core route up to FP32 accumulation order.

- Shape: `k % 16 == 0`, `m >= 16`, `n >= 16`, and at least 16384 MACs. Smaller or odd shapes
  dequantize and call SGEMM.
- Alignment: cuBLASLt needs 16-byte-aligned FP8/FP32 operands. A misaligned GEMM output (a view
  into a packed buffer at an arbitrary offset) is staged through aligned scratch and copied
  back; a misaligned weight falls back to dequantize + SGEMM.
- Architecture: FP8 GEMM below compute capability 8.9 is not attempted; the trainer refuses the
  dtype up front, other callers get the fallback.

## Loss scaling

There is no `--loss-scale` flag, and adding one would be decorative in this codebase:

- Gradients, master weights, and optimizer state are FP32. Nothing that carries range-sensitive
  values is stored in FP16, so the usual reason to scale the loss (preventing FP16 gradient
  underflow) does not apply.
- `blt_model_backward` takes no upstream scale: the loss gradient is produced by the
  cross-entropy / block-diffusion head and applied directly. A multiplier applied to the loss
  would be exactly cancelled by the corresponding division before the optimizer step, and
  without that division the effective learning rate would change instead of the numerics.

If a future variant narrows gradients or the optimizer state, the change has to land in the
loss head (multiply) and the optimizer (unscale before the update, check for Inf/NaN, adjust
the scale), not in a CLI flag that only moves a constant around.

## CPU ground truth and expected agreement

Every CUDA path is checked against the CPU implementation: `blt_fp8_dequantize` +
`cpu_fp8_roundtrip` reproduce exactly the operands cuBLASLt sees, so the reference and the
device result differ only by accumulation order and the FP8 datapath itself. On Ada the FP8
datapath costs about 2e-4 relative versus exact FP32 accumulation (single products are exact;
the loss is in the accumulation network), so FP8 parity tolerances are around 1e-2 absolute on
the test shapes, not exact equality. BF16/FP16 parity is exact for the operand rounding and
differs only in accumulation.

On the model integration test (6 SGD steps, 5.5-nat loss), FP8 runs stay within 0.01 of the
FP32 loss trajectory and FP16 is indistinguishable at that scale. Run-to-run loss noise from
random timesteps is larger than any of this.

## Throughput

`cuda_bench` micro cells, RTX 4060, n x n x n, TFLOP/s:

| n | FP32 | BF16 | FP16 | FP8-E4M3 | FP8-E5M2 |
|---|------|------|------|----------|----------|
| 512 | 5.71 | 11.63 | 11.09 | 4.04 | 4.05 |
| 1024 | 6.88 | 18.60 | 18.70 | 7.43 | 7.86 |
| 2048 | 6.54 | 19.24 | 20.45 | 11.46 | 11.57 |

The FP8 cells measure the training recipe: the activation stays FP32 and is quantized on every
call, and the weight shadow is stored `[K,N]` and transposed into scratch for the TN-only
cuBLASLt kernel. The bare kernel rate is higher; at n=512 the conversion dominates and run-to-run
variance is 20-40% because the kernel is only tens of microseconds long.

With these numbers, FP8 is a capability, not a speed feature on Ada: it beats FP32 by ~1.8x at
n=2048 and loses to BF16. End-to-end step time at the model sizes in `docs/BENCHMARKS.md` is
dominated by launch overhead, elementwise work, and the FP32 weight-gradient GEMM, so expect
single-digit percentages from any of the narrow dtypes.

## Reproducing

```
# dtype sweep on a CUDA build
for dt in fp32 bf16 fp16 fp8; do
  ./build-cuda/train_blt_d --corpus data/corpus.bin --backend cuda --diffusion 1 \
    --embed 64 --hidden 128 --layers 2 --block-size 8 --window 64 --steps 20 --dtype "$dt"
done

# FP8 E5M2 weights
./build-cuda/train_blt_d ... --dtype fp8 --fp8-format e5m2

# GEMM rates per dtype
./build-cuda/cuda_bench --results /tmp/bench.jsonl
```

## Limitations

- The trainer does not expose the dtypes on the CPU backend; only the ops support them.
- FP8 tensor-core GEMM supports the TN orientation only before Blackwell, so a weight shadow is
  transposed into scratch on every call rather than stored transposed.
- CUDA graph capture of the FP8 path is untested; the GEMM allocates scratch per call.
- The inference binaries (`blt_generate`, the Fast-BLT modes in `blt_bench_infer`) construct
  FP32 models; there is no flag to load a model with shadows and decode against them.
