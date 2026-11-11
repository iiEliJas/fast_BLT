# E011: FP16 and FP8 mixed precision

Status: Checked.

## Question

Can `train_blt_d` run its GEMMs in FP16 and in both FP8 encodings on Ada, does the result track
FP32, and what does the hardware actually support?

## Setup

- RTX 4060, 8 GB, compute capability 8.9, 24 SMs.
- CUDA 13.3, cuBLASLt 13.6.0.2, driver 616.x.
- Corpus `data/train.bin` for the trainer runs; a 4 KB random corpus for the shape checks.
- Model: E=64, H=128, enc/glob/dec 2/2/2, window 64, batch 8, block size 8, AdamW, lr 1e-4.

Scope of the implementation: FP32 masters, gradients, optimizer state, activations, norms,
softmax, and cross-entropy in every mode. Only GEMM operands are narrowed. FP8 weight shadows
carry one FP32 scale each, recomputed from the master at every refresh; activations and
incoming gradients get a fresh amax-derived scale per call.

## cuBLASLt support matrix (fp8)

A standalone probe enumerated four dtype pairings x four orientations x five scale
configurations x three shapes (240 cases) and checked every success against a host reference
built from the same FP8 bytes.

| pairing | TN | NT | NN | TT |
|---------|----|----|----|----|
| E4M3 x E4M3 | supported | no | no | only at m == 1 (same kernel as TN) |
| E4M3 x E5M2 | supported | no | no | only at m == 1 |
| E5M2 x E4M3 | supported | no | no | only at m == 1 |
| E5M2 x E5M2 | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` | `NOT_SUPPORTED` |

Additional findings:

- The TN orientation is the only real recipe on sm_89, which is why the weight shadow is
  transposed into scratch per call.
- Per-tensor scale pointers are optional; an unset pointer behaves as 1.0. A one-sided scale is
  therefore silently accepted and gives a plausible but wrong result, so both are always set.
- `CUBLASLT_MATMUL_DESC_AMAX_D_POINTER` with an FP32 destination returns
  `CUBLAS_STATUS_INVALID_VALUE` for every case; it is not set.
- Dyadic inputs are reproduced bit-exactly, and all 254 finite E4M3 / 248 finite E5M2 byte codes
  decode identically to `csrc/ops/fp8_convert.h`, but full k=64 sums differ from an exact FP32
  accumulation by ~2.2e-4 of max|C| (5.4e-3 absolute at |C| ~ 25). The error is in the Ada FP8
  datapath, not in accumulation order: single products are exact and FP32 sequential/pairwise
  references are themselves exact on the same data.

Consequence in the code: an E5M2 weight is paired with an E4M3 activation and an E4M3
incoming gradient (`blt_fp8_pair_fmt`). `--dtype fp8 --fp8-format e5m2` therefore narrows the
weights to E5M2 and leaves the dynamic operands at E4M3.

## Results

GEMM throughput, `cuda_bench`, default `--iters 10`, TFLOP/s:

| n | FP32 | BF16 | FP16 | FP8-E4M3 | FP8-E5M2 |
|---|-----:|-----:|-----:|---------:|---------:|
| 512 | 5.71 | 11.63 | 11.09 | 4.04 | 4.05 |
| 1024 | 6.88 | 18.60 | 18.70 | 7.43 | 7.86 |
| 2048 | 6.54 | 19.24 | 20.45 | 11.46 | 11.57 |

FP8 includes the per-call activation quantization and the weight transpose that the training
recipe requires, so it is not a bare kernel rate. At n=512 the cell is tens of microseconds
and run-to-run spread is 20-40%.

Trainer, 20 steps, same seed and corpus:

| dtype | final avg_loss |
|-------|---------------:|
| fp32 | 967.2496 |
| bf16 | 967.2494 |
| fp16 | 967.2494 |
| fp8-e4m3 | 967.2528 |
| fp8-e5m2 (60 steps) | 969.2890 |

The 60-step E5M2 run is a separate invocation; it stays finite and monotone at the same lr.

Model integration test (6 SGD steps, lr 0.05, 5.5-nat starting loss, CPU ground truth): FP16 is
indistinguishable from FP32 at 4 decimal places, FP8-E4M3 stays within 0.005, FP8-E5M2 within
0.008 of the FP32 trajectory.

## Conclusions

- FP16 and FP8 both run end to end on sm_89 with no numerical surprises at these scales.
- FP8 is a capability, not a speed feature on Ada: 1.75x FP32 at n=2048, 0.6x BF16. The
  cuBLASLt TN-only recipe and the per-call activation quantization eat the theoretical 2x over
  BF16.
- E5M2 weights cost almost nothing in loss here because the test weights are small, but two
  mantissa bits is a real precision floor; E4M3 remains the default.
- No loss scaling is implemented: gradients are FP32, and a loss multiplier that is not undone
  in the optimizer would only change the effective learning rate.

## Reproducing

```bash
CUDA_VISIBLE_DEVICES=0 ./build-cuda/cuda_bench --results /tmp/bench.jsonl
for dt in fp32 bf16 fp16 fp8; do
  CUDA_VISIBLE_DEVICES=0 ./build-cuda/train_blt_d --corpus data/train.bin --backend cuda \
    --diffusion 1 --embed 64 --hidden 128 --layers 2 --block-size 8 --window 64 \
    --steps 20 --dtype "$dt"
done
CUDA_VISIBLE_DEVICES=0 ./build-cuda/train_blt_d ... --dtype fp8 --fp8-format e5m2
```

Design notes and limitations: [../MIXED_PRECISION.md](../MIXED_PRECISION.md).
