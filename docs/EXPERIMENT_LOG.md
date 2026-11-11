# Experiment Log

Compact index of significant fBLT research activity. This is an index, not a chronological
dump of commands or measurements. Detailed evidence lives in the linked reports.

## Status definitions

| Status | Meaning |
|--------|---------|
| Planned | Defined, but not yet run. |
| Exploratory | Initial observations exist, but the result has not been sufficiently checked. |
| Checked | The setup and results have been examined and the report contains the available evidence. This does not imply independent reproduction. |
| Independently reproduced | A separate rerun has confirmed the reported result within the documented conditions and tolerances. |
| Superseded | A later experiment or correction has replaced the result as the current reference. |

These are evidence and workflow statuses, not quality rankings. An exploratory or inconclusive
report can still be useful.

## Experiment registry

| ID | Research question | Status | Main finding | Report |
|----|-------------------|--------|--------------|--------|
| E001 | Why did a checkpoint with 97.76% teacher-forced top-1 accuracy produce repetitive garbage under greedy generation? | Checked | Four implementation deviations from the papers were found and fixed: causal loss coverage, decoder cross-attention, global attention boundary scoping, and BLT-D train/inference block attention. | [docs/experiments/E001_last_row_gap.md](experiments/E001_last_row_gap.md) |
| E002 | How does `train_blt_d`'s objective relate to Fast-BLT Eq. 5-7? | Checked | The code deviation is that `L_clean` is a mean rather than a sum; `L_mask` already uses the paper's `1/t` sum. The paper form changes the clean:mask balance and gradient scale. | [docs/experiments/E002_loss_normalization.md](experiments/E002_loss_normalization.md) |
| E003 | Does the paper-exact loss improve masked accuracy at 30k steps? | Checked | The 30k probe reaches causal BPB 1.7536 and train-layout masked accuracy 0.518, versus 5.3614 / 0.089 for the older mean-form run. It also reverses two undertrained-model conclusions from E002. | [docs/experiments/E003_paper_loss_probe.md](experiments/E003_paper_loss_probe.md) |
| E004 | Do forced training-window boundaries weaken final-byte prediction? | Checked | Boundary-final accuracy is 50.00%, versus 99.38% for natural closures; the gap is driven by closure type rather than patch length. The effect appears across all four measured checkpoints. | [docs/experiments/E004_forced_boundary_patches.md](experiments/E004_forced_boundary_patches.md) |
| E005 | Is the 128-patch cap justified? | Exploratory | No measurements yet. The cap is an implementation limit, not a paper limit. | [docs/experiments/E005_patch_cap_limit.md](experiments/E005_patch_cap_limit.md) |
| E006 | How does held-out causal BPB improve over 600k training steps? | Checked | BPB improves from 1.7536 at 30k steps to 0.9978 at 600k steps, with no observed overfitting. | [docs/experiments/E006_tinystories_quality.md](experiments/E006_tinystories_quality.md) |
| E007 | How well does BLT-D work on source code? | Planned | Config created; training not started. | [docs/experiments/E007_stack_smol_code.md](experiments/E007_stack_smol_code.md) |
| E006 | Which architecture and schedule choices matter at 2.97M parameters? | Checked | High-t warmup improves BPB by 0.33; mask-late shows no benefit; deeper decoder does not help; deeper encoder is harmful. Results use 3-6 seeds at 40k steps. | [docs/ABLATIONS.md](ABLATIONS.md) |
| E007 | Does BLT-DV improve over greedy at the 7.8M-parameter scale? | Checked | BLT-DV is a net win only at the trained block size `B=4`: 65% acceptance, output byte-identical to greedy, 0.85x memory bandwidth, 0.89x wall-clock. Configurations at `B>=8` lose badly. | [docs/BENCHMARKS.md](BENCHMARKS.md) |
| E008 | What CUDA speedup and determinism cost does fBLT have? | Checked | CUDA provides 76x training and 34x generation speedup over the single-threaded CPU baseline at the production config. CUDA is non-deterministic unless `--deterministic` is used. | [docs/BENCHMARKS.md](BENCHMARKS.md) |
| E009 | Where does training time go, and what should be optimized first? | Checked | Backward pass is ~50% of step time on CUDA (FP32 and BF16); top kernels are sgemm, patch-pool backward, and many memset/memcpy calls. | [docs/experiments/E009_backward_profile.md](experiments/E009_backward_profile.md) |
| E010 | How much faster is BF16 than FP32 for GEMMs and full training steps, and does it stay numerically stable? | Checked | BF16 GEMMs are 2.2-3.2x FP32; end-to-end steps improve 1.5-3.9% because those steps are launch-bound and the weight-gradient GEMM and optimizer stay FP32. Loss tracks FP32 with no NaN/Inf. | [docs/experiments/E010_bf16_mixed_precision.md](experiments/E010_bf16_mixed_precision.md) |
| E011 | Which FP8 GEMM recipes does cuBLASLt support on sm_89, and do FP16/FP8 training runs track FP32? | Checked | TN is the only FP8 orientation below Blackwell and E5M2 x E5M2 is not supported at all, so E5M2 weights run against E4M3 activations. FP8 reaches 1.75x FP32 GEMM throughput at n=2048 but 0.6x BF16; FP16 and FP8 loss track FP32 within 0.01 over short runs. | [docs/experiments/E011_fp8_mixed_precision.md](experiments/E011_fp8_mixed_precision.md) |

## Notes

- E003 supersedes the undertrained-model conclusions recorded in E002.
- E004 is not a deviation from either paper; both papers leave training-window boundary
  handling unspecified.
- E004's 7.20pp stage-2 gain is confounded by simultaneous learning-rate and mask-scale
  changes, as stated in its report.
- No experiment in this log has been marked independently reproduced.

## Historical evidence gaps

E001-E004 predate this log and may not include complete commands, seeds, hardware details, or
artifact references. Missing information is noted rather than inferred. Future reports should use
[docs/experiments/TEMPLATE.md](experiments/TEMPLATE.md) so the reproduction record is complete.

## Maintenance

Add or update an entry when a significant experiment, benchmark finding, negative result, or
investigation referenced elsewhere is completed. A temporary debugging run does not need a
formal entry. Do not copy full numerical tables into this index; link to the report instead.
