# fBLT TODO

Public roadmap for fBLT. Version 1.0 is released, including the 7.8M-parameter TinyStories
checkpoint and the matching entropy LM:

- <https://github.com/iiEliJas/fBLT/releases/tag/v1.0.0>

If you want to contribute:

1. Open an issue or comment on an existing one before starting, so work does not overlap.
2. Create a feature branch. Do not commit directly to `main`.
3. Keep the change scoped to one item where possible. Add tests and update the docs.
4. Run the CPU suite from the repository root:

   ```bash
   cmake --build build --target test
   pytest tests/
   ```

   Run the CUDA suite too if your change touches `csrc/ops/*.cu`.
5. Include the command, hardware, and `resolved_config.yaml` with any benchmark or
   training result.

Negative results are useful. If an idea does not work, record it in
[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) or mention it in the issue. For the
full workflow, see [docs/RESEARCH_GUIDE.md](docs/RESEARCH_GUIDE.md).

## Status

| # | Item | Area | Status |
|---|------|------|--------|
| 1 | Gradient accumulation / batch > 1 | Training | Done (PR #66) |
| 2 | BF16 mixed-precision training | Training | Done (global transformer, PR #68) |
| 3 | Multi-GPU data parallel | Training | Design draft |
| 4 | Evaluation harness (BPB + CORE tasks) | Evaluation | Open |
| 5 | MuonAdamW and optimizer parameter groups | Training | Open |
| 6 | Compute-optimal scaling dial | Training | Open |
| 7 | Document-aware data pipeline | Data | Open |
| 8 | Batched inference and KV cache | Inference | Partial (storage + helper, PRs #69/#70) |
| 9 | Checkpoint ergonomics | Checkpoints | Open |
| 10 | Byte-level SFT and chat | Post-training | Open |
| 11 | Sandboxed tool execution | Post-training | Open |
| 12 | RL / GRPO | Post-training | Open |
| 13 | FP8 / FP16 training | Training | Done (primitives + integration, --fp8 flag) |
| 14 | Dataset download and cache | Data | Open |
| 15 | Reproducible speedrun and leaderboard | Tooling | Open |
| 16 | Peak FLOPS/BW table and MBU | Tooling | Open (E009 profiling landed) |
| 17 | Interactive chat CLI | Inference | Open |
| 18 | Experiment log and negative results | Docs | Done (E009 added) |
| 19 | Test coverage for new subsystems | Testing | Open |
| 20 | Architecture experiments | Research | Exploratory |
| 21 | Hugging Face checkpoint and model integration | Integration | Open |
| 22 | FineWeb/DCLM corpus pipeline | Data | Open |
| 23 | Tokenizer/BPE hybrid patching modes | Data | Open |
| 24 | Official-scale configuration and scaling matrix | Training | Open |
| 25 | Tensor parallelism and multi-node training | Training | Open |
| 26 | Backward-pass and launch-overhead optimization | Training (CUDA) | In progress (async memset, redundant passes, batched clip norm, uninit tensors) |

---

## 1. Gradient accumulation / batch > 1

- **Status:** Open
- **Area:** Training
- **Depends on:** nothing, but 5 makes the optimizer recipe more useful

### Why

fBLT trains one window per optimizer step. That keeps the loop simple, but it limits the
effective batch size and makes gradient noise high. It also means large-batch recipes from
the literature cannot be reproduced without changing the trainer.

Prior art: nanochat exposes `grad_accum_steps` and normalizes the accumulated gradient before
clipping and the optimizer step (`scripts/base_train.py`, `nanochat/optim.py`).

### Current state

- `csrc/train_blt_d.c` selects one window with `train_window_index(step, num_windows, seed)`
  and runs one forward, one backward, clipping, and one optimizer step.
- Gradients live in a `blt_model_grad` allocated from `model_arena`.
- The norm-gradient zeroing is explicit in the training loop. The fused-RMSNorm weight
  gradients are easy to miss when the loop is restructured.
- `args_t` has no batch-size or accumulation field.
- The backward entry points are `blt_local_encoder_backward`, `blt_global_transformer_backward`,
  `blt_local_decoder_backward_diffusion`, and `blt_model_backward`.

### Implementation

1. Add `--grad-accum N` to `args_t`, default `1`. Validate `N >= 1`.
2. Keep one gradient buffer for the micro-batch and add a second accumulator buffer, or
   accumulate in place if the backward contract allows it. Document the extra memory.
3. Add reusable helpers in `csrc/models/`:

   ```c
   void blt_model_grad_zero(const blt_model *model, blt_model_grad *g);
   void blt_model_grad_add(const blt_model *model, blt_model_grad *dst,
                           const blt_model_grad *src);
   void blt_model_grad_scale(const blt_model *model, blt_model_grad *g, float scale);
   ```

   Use `blt_model_visit_params` / `blt_model_visit_param_pairs` so every tensor is covered.
   Verify that the visitor includes the encoder embedding, n-gram tables, decoder d0
   embedding, and all norm gradients before removing the explicit zeroing.

4. Restructure the step:

   ```c
   blt_model_grad_zero(model, grad_accum);
   double step_loss = 0.0;

   for (size_t j = 0; j < a.grad_accum; j++) {
       const size_t data_step = step * a.grad_accum + j;
       const size_t w = train_window_index(data_step, num_windows, a.seed);

       blt_arena_reset(scratch);
       blt_model_grad_zero(model, grad);

       // forward and backward into grad
       step_loss += loss;

       blt_model_grad_add(model, grad_accum, grad);
   }

   blt_model_grad_scale(model, grad_accum, 1.0f / (float)a.grad_accum);
   const float pre_clip_norm = clip_all(model, grad_accum, a.max_norm);
   adamw_all(model, grad_accum, aw, &cfg);
   ```

   Scale before clipping so the clip threshold applies to the averaged gradient.

5. Keep schedules on the optimizer step, not the micro-batch:

   - Mask warmup, `--t-warmup-hi`, and LR schedules use `step`.
   - Window selection, diffusion timestep RNG, and batch-build RNG use `data_step`.
   - Epoch and ETA counts use optimizer steps and account for `grad_accum`.

6. Resume:

   - `--start-step` is an optimizer step.
   - The first micro-batch is `start_step * grad_accum`.
   - Verify that resuming reproduces the same window/RNG stream as an uninterrupted run.

7. Logging:

   - `--loss-log` writes the averaged loss once per optimizer step.
   - Gradient/update norm logs use the accumulated gradient after scaling.
   - `[MONITOR]`, `[EVAL]`, and `[TIMING]` run once per optimizer step.

### Acceptance

- `--grad-accum 1` is bit-identical to the current trainer for the same seed and config.
- `--grad-accum 2` produces the average of two single-window gradients before the update.
- A saved/resumed run with accumulation matches an uninterrupted run.
- Epoch, ETA, and report counts are correct after `steps * grad_accum` windows.
- The extra gradient buffer is documented, and an arena overflow fails with a clear error.

### Tests and docs

- Add a unit test for `zero`/`add`/`scale` covering every tensor in `blt_model_grad`.
- Add a CLI test that compares accumulated gradients to a manually averaged reference.
- Add a resume test.
- Document the flag, data-order semantics, schedule semantics, and memory cost in
  `docs/CLI_REFERENCE.md`.
- Add a production example config with accumulation enabled.

### Risks

- If a backward function accumulates instead of assigning, the micro-batch buffer must be
  zeroed at the right point. Audit all four backward entry points.
- Missing norm gradients produce silently wrong updates that still look plausible.
- `model_arena` may need to grow for the extra buffer and optimizer state.
- Future dropout must use a per-micro-batch RNG stream, not a per-optimizer-step stream.

---

## 2. BF16 mixed-precision training

- **Status:** Partially wired, not usable end-to-end
- **Area:** Training
- **Depends on:** nothing

### Why

The BF16 matmul path is already faster than FP32 on CUDA, but the trainer cannot enable it
end-to-end. The global transformer has BF16 weight copies, while the local encoder and decoder
do not. The refresh function that copies updated FP32 master weights into the BF16 views is
never called, so enabling the existing flag would train on stale or uninitialized data.

### Current state

- `BLT_DTYPE_BF16` exists in `csrc/core/dtype.h`.
- `blt_cast` supports FP32 <-> BF16 on CPU and CUDA.
- `blt_matmul` supports BF16 inputs with FP32 accumulation through `cublasGemmEx` on CUDA.
- `blt_transformer_stack_config` has `use_bf16`; `blt_transformer_stack_init` allocates BF16
  copies of the matmul weights and uses them in the forward pass.
- `blt_mixed_weight_refresh` exists in `csrc/models/transformer_stack.c` but is never called.
- Only `blt_global_transformer_config` exposes `use_bf16`.
- The model defaults zero the config, so `use_bf16` is always off.
- The trainer has no `--bf16` or `--dtype` flag.
- Optimizer state and checkpoints remain FP32.

### Implementation

#### Global transformer

1. Add `--bf16` (or `--dtype bf16|fp32`) to `args_t`, `TrainConfig`, and the CLI docs.
2. Before `blt_model_create`, set `cfg.global_config.use_bf16 = a.use_bf16`.
3. After model creation and weight loading, call a new `blt_model_refresh_bf16(model)` that
   walks every FP32/BF16 pair and refreshes the BF16 view.
4. After every optimizer step, refresh the BF16 views from the FP32 masters. If a refresh
   interval is added, document the approximation and keep correctness as the default.
5. Require CUDA:

   ```c
   BLT_REQUIRE(!a.use_bf16 || a.use_cuda, "--bf16 requires --backend cuda");
   ```

   The CPU BF16 matmul path converts to FP32 temporaries, so it is not a speed path.

#### Local encoder and decoder

6. Add `use_bf16` to `blt_local_encoder_config` and `blt_local_decoder_config`, and propagate
   it to their internal transformer stacks.
7. If the local modules use their own layer storage rather than
   `blt_transformer_stack_init`, add BF16 copy tensors and the same refresh pattern.
8. Extend `blt_model_refresh_bf16` to cover encoder and decoder layers.
9. Re-run the numerical parity and loss-curve checks. The local layers dominate the parameter
   count and compute, so this is where the speedup should appear.

#### Training loop and checkpoints

10. Print the active dtype in the startup line and `RESULT`.
11. Keep norms, softmax, cross-entropy, and optimizer state in FP32. Cast only matmul inputs.
12. Serialize FP32 masters only. Reconstruct BF16 views after loading. A BF16-trained
    checkpoint must load into an FP32 model and produce the same outputs within tolerance.
13. Document the `--model-mb` impact of the extra BF16 copies.

### Acceptance

- `--bf16` on CUDA reduces step time on a matmul-bound config.
- The loss curve stays within a documented tolerance of FP32, with no NaN/Inf over a long run.
- BF16 views are refreshed after optimizer steps; a unit test mutates a master weight and
  checks that the view changes.
- `--bf16` on CPU exits with a clear error.
- Checkpoints remain FP32 and load into both FP32 and BF16 models.

### Tests and docs

- Unit test for `blt_mixed_weight_refresh` and `blt_model_refresh_bf16` across every layer type.
- Integration test comparing several steps of FP32 and BF16 with the same seed/data.
- Add a CUDA benchmark entry with dtype and hardware metadata.
- Document the flag, numerical expectations, and CPU limitation in `docs/CLI_REFERENCE.md`
  and the README.

### Risks

- Stale BF16 views are the most likely failure mode. Enabling `use_bf16` without a refresh
  trains on garbage.
- BF16 has a narrow mantissa. Keep normalization, softmax, cross-entropy, and optimizer state
  in FP32.
- Do not serialize BF16 copies; that breaks checkpoint compatibility and wastes space.
- Windows CUDA is compile-checked only. Runtime validation needs a real GPU.

---

## 3. Multi-GPU data parallel

- **Status:** Design draft in `MULTI_GPU.md`
- **Area:** Training
- **Depends on:** 1 (gradient accumulation), 5 (optimizer parameter groups)

### Why

fBLT is single-GPU and single-process. `MULTI_GPU.md` describes a data-parallel design, but
there is no NCCL/MPI layer, no rank/world-size flags, and no distributed checkpoint or eval
handling. Scaling beyond one device needs all of those.

Prior art: nanochat runs DDP with `torchrun`, performs gradient all-reduce, and shards
optimizer state across ranks in `MuonAdamW` (`nanochat/optim.py`).

### Current state

- `MULTI_GPU.md` documents the design and the current single-GPU limits.
- `csrc/train_blt_d.c` has no rank or world-size fields.
- `csrc/core/` has no `distributed.*`.
- `CMakeLists.txt` has no NCCL option.
- Checkpoint saves and evaluation are unconditional; there is no rank guard.

### Implementation

1. Add a CMake option:

   ```cmake
   option(USE_NCCL "Enable NCCL multi-GPU support" OFF)
   if(USE_NCCL)
     find_library(NCCL_LIBRARY nccl REQUIRED)
     find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)
     target_compile_definitions(blt_core PRIVATE BLT_WITH_NCCL)
     target_link_libraries(blt_core PRIVATE ${NCCL_LIBRARY})
     target_include_directories(blt_core PRIVATE ${NCCL_INCLUDE_DIR})
   endif()
   ```

   Gate to Linux + CUDA. Keep single-GPU builds working with stubs.

2. Add `csrc/core/distributed.h` / `.c`:

   ```c
   int  blt_dist_init(int rank, int world_size, const char *master_addr, int master_port);
   void blt_dist_finalize(void);
   int  blt_dist_rank(void);
   int  blt_dist_world_size(void);
   void blt_dist_allreduce_sum(float *buf, size_t n, int backend);
   void blt_dist_broadcast(float *buf, size_t n, int root, int backend);
   void blt_dist_barrier(void);
   ```

3. Add flags and environment support:

   ```text
   --world-size N
   --rank R
   --master-addr IP
   --master-port P
   --local-rank R
   ```

   Also accept `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, and `LOCAL_RANK` so a
   `torchrun`-style launcher works. Explicit flags win.

4. Initialize before the CUDA arena/model and select the device with `cudaSetDevice`.

5. Initialize the model consistently across ranks. Either use the same seed everywhere or
   initialize on rank 0 and broadcast every tensor once at startup. The broadcast is safer if
   any init path becomes rank-dependent.

6. Shard data:

   ```c
   const size_t global_index = (step * a.grad_accum + j) * a.world_size + (size_t)a.rank;
   const size_t w = train_window_index(global_index, num_windows, a.seed);
   ```

   The effective batch is `world_size * grad_accum` windows per optimizer step.

7. All-reduce gradients after accumulation and before clipping:

   - Walk `blt_model_visit_param_pairs` and call `ncclAllReduce` with `ncclFloat`, `ncclSum`,
     and the active stream.
   - Scale by `1.0 / world_size`, or use `ncclAvg` if the NCCL version supports it.
   - Fatal if a gradient is not on CUDA. A CPU multi-process backend is a separate item.

8. Logging and evaluation:

   - All-reduce the scalar loss, or print local loss per rank and the global average on rank 0.
   - Rank 0 runs held-out evaluation and monitoring. Other ranks wait at a barrier.
   - Optionally shard evaluation windows by rank and all-reduce the CE sum later.

9. Checkpointing:

   - Rank 0 writes weights and optimizer state; other ranks wait at a barrier.
   - All ranks load from shared storage, or rank 0 loads and broadcasts.
   - Periodic and best-checkpoint paths must be identical across ranks.

10. Optimizer state sharding (advanced):

    - The first milestone replicates the full optimizer on every rank.
    - A second milestone shards optimizer state across ranks with reduce-scatter/all-gather,
      following the approach in nanochat's `MuonAdamW`.
    - This depends on the grouped optimizer in item 5 and should not block basic DDP.

### Acceptance

- `--world-size 1` with `USE_NCCL=ON` is bit-identical to the single-GPU trainer.
- A 2-GPU run with `world_size=2, grad_accum=1` consumes twice the data per step and produces
  a loss curve comparable to a single-GPU run with `grad_accum=2`.
- Gradients after all-reduce match a CPU reference average in a unit test.
- Only rank 0 writes checkpoints/logs; no interleaved or corrupt files.
- Rank 0's evaluation uses the global model.
- A failed rank terminates the job instead of hanging indefinitely.

### Tests and docs

- Compile-check the NCCL path in CI even when no multi-GPU runner is available.
- Add a single-process NCCL test (`world_size=1`) that is a no-op.
- Add a 2-rank manual validation script under `scripts/`.
- Update `MULTI_GPU.md` from design to implementation notes, including launch commands.
- Document the effective batch formula and shared-filesystem requirement.

### Risks

- NCCL is unavailable on CPU-only and Windows machines. The build must remain clean without it.
- A rank that exits early can leave the others blocked. Add timeouts and clear errors.
- Each rank holds a full model, gradients, optimizer state, and scratch.
- A wrong data-index formula makes ranks train on the same windows.
- Concurrent checkpoint writes corrupt files. Enforce rank-0 writes and barriers.
- Do not promise bit-identical runs across different world sizes or topologies.

---

## 4. Evaluation harness (BPB + CORE tasks)

- **Status:** Open
- **Area:** Evaluation
- **Depends on:** nothing; task generation needs 7 and 10 for best results

### Why

fBLT has strong internal diagnostics (`sanity_check`, `masked_acc`, patch splits, leak probes)
and a corpus BPB evaluation, but no single command that evaluates a checkpoint on a named
benchmark and writes a comparable record. The BLT paper reports downstream results; the
repository cannot currently reproduce them.

Prior art: nanochat implements the DCLM CORE benchmark with multiple-choice, schema, and
language-model tasks, few-shot prompts, centered accuracy, distributed evaluation, and CSV
output (`nanochat/core_eval.py`, `scripts/base_eval.py`, `tasks/common.py`).

### Current state

- The trainer computes causal BPB and masked accuracy on a held-out corpus.
- `bench/harness.c` runs benchmark configurations.
- `sanity_check` and the experiment tools cover internal diagnostics.
- There is no task registry, few-shot prompt renderer, or downstream score output.
- `bench/eval_suite.py` does not exist.

### Implementation

#### Corpus BPB tool

1. Add `tools/eval_bpb.c` or extend `sanity_check` with:

   ```text
   --checkpoint FILE
   --embed E --hidden H --enc-layers N --glob-layers N --dec-layers N
   --corpus FILE
   --window W --windows N --skip S
   --patcher fixed|entropy --entropy-lm FILE
   --block-size B --layout paper|novel --t F
   --output-json FILE
   ```

2. Write a JSON record with `checkpoint`, `corpus`, `corpus_sha256`, `window`, `windows`,
   `patcher`, `causal_bpb`, `masked_bpb`, `masked_acc`, `params`, `git_commit`, and
   `timestamp`.

3. Store the evaluation protocol in the record. BPB is only comparable when byte boundaries
   and windowing match.

#### Task harness

4. Add a task registry under `fblt/eval/`:

   - `mc`: multiple-choice; score each continuation and pick the lowest byte NLL.
   - `schema`: constrained answer selection from a schema.
   - `lm`: free-form generation with a task-specific scorer.
   - `code`: execute generated code in a sandbox (link item 11).

5. Start with public, reproducible tasks: ARC, MMLU, GSM8K, and HumanEval. Add HellaSwag,
   PIQA, Winogrande, and SQuAD when their prompt formats are stable.

6. Render byte-level prompts. Because fBLT has no tokenizer, every candidate continuation is
   scored in bytes. Normalize by continuation byte count when the task requires it.

7. Support few-shot examples, answer extraction, centered accuracy, and per-task metadata.

8. Add `--eval core,bpb,tasks` to the training runner and a standalone
   `fblt-eval` entry point.

9. Add `bench/eval_suite.py` to run a matrix of checkpoints and tasks and aggregate JSONL or
   CSV. Reuse `bench_plot.py` where possible.

10. Add a corpus prep script for WikiText-103 and enwik8 with SHA-256 checksums and a
    deterministic split. Reuse the TinyStories manifest format.

### Acceptance

- One command produces a reproducible JSON record for a checkpoint and corpus.
- The same checkpoint evaluates identically on CPU and CUDA within tolerance.
- Task scores are printed in a table and written to CSV/JSONL.
- Few-shot and zero-shot modes are both selectable.
- Entropy-patching evaluations store the entropy LM, threshold, block size, and seed.
- A CI smoke test runs the BPB tool on the sample corpus with a tiny checkpoint.

### Tests and docs

- Unit tests for prompt rendering, answer extraction, and byte-NLL scoring.
- A fixture task with known answers to verify centered accuracy.
- Integration test that runs one task on a tiny model and validates the JSON schema.
- Document each task, prompt format, split, and metric in `docs/EVALUATION.md`.
- Document that token-level benchmark numbers are not directly comparable to byte-level
  scores without the same protocol.

### Risks

- Byte-level scores are not interchangeable with token-level scores from other papers.
- Entropy patching makes evaluation stochastic unless the LM and thresholds are fixed.
- Benchmark contamination and prompt-format sensitivity can dominate small-model results.
- HumanEval requires a sandbox; do not run untrusted code in-process.

---

## 5. MuonAdamW and optimizer parameter groups

- **Status:** Open
- **Area:** Training
- **Depends on:** nothing, but 1 benefits from the final optimizer semantics

### Why

fBLT supports SGD and AdamW with one global learning rate. Muon is a strong optimizer for 2D
weight matrices and is used by nanochat alongside AdamW for embeddings and scalar parameters.
Even without adopting Muon, per-group learning rates, betas, and weight decay are useful for
matching published recipes and for stabilizing small-model training.

Prior art: `nanochat/optim.py` implements MuonAdamW with Polar Express orthogonalization,
NorMuon variance reduction, MuonEq row equilibration, cautious weight decay, and per-group
hyperparameters. `nanochat/gpt.py` assigns parameters to groups by shape and role.

### Current state

- `csrc/train_optim.c` has `adamw_all` and `sgd_all`.
- `adamw_pair_fn` applies one `weight_decay` value to every tensor; selective weight decay
  now skips norms, but there are no per-group LRs or betas.
- `blt_param_info` exposes `is_norm`, `is_enc`, `is_glob`, `is_dec`, `layer_idx`,
  `param_in_layer`, `component_group`, `numel`, and `backend`.
- There is no Muon implementation and no orthogonalization code.

### Implementation

1. Extend `blt_param_info` with an optimizer group and a decay flag. Suggested groups:

   - `matrix`: 2D transformer/FFN weights, Muon candidates.
   - `embedding`: encoder embedding, decoder d0, n-gram tables, AdamW.
   - `norm`: RMSNorm scales, AdamW, no weight decay.
   - `scalar`: any future biases or scalar gains, AdamW, no weight decay.

2. Add an optimizer config:

   ```c
   typedef struct {
       float lr;
       float beta1;
       float beta2;
       float eps;
       float weight_decay;
       int decay;
   } blt_optim_group_config;
   ```

   Store one per group in `blt_optim_config` and pass it through `adamw_all`/`sgd_all`.

3. Add flags:

   ```text
   --optimizer sgd|adamw|muon_adamw
   --matrix-lr F
   --embedding-lr F
   --norm-lr F
   --matrix-beta1 F
   --matrix-beta2 F
   --matrix-weight-decay F
   --muon-momentum F
   --muon-ns-steps N
   --muon-ns-eps F
   ```

   Keep the existing flags as defaults so old configs still work.

4. Implement Muon on CPU first:

   - Operate on 2D matrices only.
   - Maintain a momentum buffer.
   - Orthogonalize the momentum with Newton-Schulz or Polar Express iterations.
   - Apply NorMuon variance reduction and MuonEq row equilibration when enabled.
   - Apply cautious weight decay: decay only when the update and parameter agree in sign.
   - Scale the update by the matrix shape, following the reference implementation.

5. Keep AdamW for embeddings, norms, and scalars. Do not apply Muon to 1D tensors.

6. Add optional LR scaling:

   - AdamW LR scaled by `sqrt(batch_size / reference_batch_size)`.
   - Matrix LR scaled by `1 / sqrt(d_model / reference_d_model)`.
   - Weight decay scaled by the fraction of the training horizon.

   Make these opt-in and record the reference values in the config.

7. Add diagnostics: per-group gradient norm, update norm, and effective LR.

8. Add a CUDA path after the CPU implementation is correct. A fused kernel is optional; a
   correct multi-kernel version is enough for the first milestone.

### Acceptance

- `--optimizer adamw` reproduces the current AdamW behavior bit-for-bit when the new flags
  are at their defaults.
- Norm and scalar parameters never receive weight decay.
- Muon updates are orthogonalized and shape-scaled; a unit test checks the update against a
  small reference matrix.
- A tiny language-model run with `muon_adamw` converges and produces finite losses.
- CPU and CUDA Muon updates agree within tolerance.
- Per-group LR/WD flags are visible in `resolved_config.yaml` and the startup log.

### Tests and docs

- Unit test for group assignment: every parameter belongs to exactly one group.
- Unit test for orthogonalization: rows/columns are approximately orthonormal after the
  iteration.
- Unit test for cautious weight decay: no decay when the sign disagrees.
- Integration test comparing one step of AdamW and MuonAdamW on a fixed tiny model.
- Document the optimizer groups, defaults, and scaling formulas in `docs/OPTIMIZERS.md`.

### Risks

- Muon's benefit at fBLT's demonstrated scale is not established. Benchmark it against a
  tuned AdamW baseline before changing defaults.
- Muon assumes 2D matrices. Transposed or fused weights need a consistent orientation.
- Orthogonalization adds compute and memory; measure it against the baseline.
- Per-group hyperparameters multiply the search space. Keep defaults conservative.

---

## 6. Compute-optimal scaling dial

- **Status:** Open
- **Area:** Training
- **Depends on:** nothing; 5 and 4 make the generated runs more useful

### Why

fBLT configs specify `embed`, `hidden`, layer counts, learning rate, batch size, and step
count by hand. Scaling experiments therefore require a new config for every size, and it is
easy to change one axis without adjusting the others.

Prior art: nanochat derives width, heads, layers, LR, batch size, training horizon, and weight
decay from a `--depth` knob. It also supports `--target-param-data-ratio` and `--target-flops`
and computes a compute-optimal batch size from `Bopt ∝ D^0.383`
(`scripts/base_train.py`).

### Current state

- `configs/train/` contains hand-written YAML configs.
- `fblt/scripts/gen_configs.py` generates ablation configs from a baseline.
- `train_sweep.c` runs config-driven sweeps.
- Model dimensions and schedules are independent inputs.

### Implementation

1. Add `fblt/scale.py` or extend `fblt/scripts/gen_configs.py` with:

   ```text
   --depth N
   --target-param-data-ratio F
   --target-flops F
   --reference-config FILE
   --out FILE
   ```

2. Map a depth knob to the BLT architecture:

   - Global transformer depth is the primary axis.
   - Encoder and decoder depths remain explicit overrides.
   - Width, head count, head dimension, and FFN width follow a documented aspect-ratio rule.
   - Keep `embed % heads == 0` and the model builder's other divisibility constraints.

3. Derive training parameters:

   - Parameter count from the architecture.
   - Target tokens from the parameter:data ratio or target FLOPs.
   - Steps from target tokens and the effective batch size.
   - Batch size from the reference batch and `Bopt ∝ D^0.383`.
   - LR from the reference LR and batch-size scaling.
   - Weight decay from the training horizon.
   - Warmup as a fraction of steps.

4. Emit a YAML config plus a manifest containing the formulas, reference values, parameter
   count, target tokens, and estimated FLOPs.

5. Add a `--dry-run` mode that prints the derived values and validates them without training.

6. Validate the generator against the existing ablation configs. The generated config should
   reproduce a known parameter count and step count.

### Acceptance

- A single depth value produces a runnable config with consistent dimensions and schedules.
- `--target-param-data-ratio` and `--target-flops` produce documented step counts.
- The manifest records every formula and reference value used.
- Invalid divisibility or non-positive values fail before training.
- A generated config trains on CPU and CUDA without manual edits.

### Tests and docs

- Unit tests for width/head/dimension rounding and parameter-count estimation.
- A golden test that regenerates a checked-in reference config.
- Document the formulas, assumptions, and the fact that nanochat's constants are calibrated
  for a different architecture. Re-tune against fBLT data before trusting the defaults.

### Risks

- Scaling-law constants do not transfer automatically between architectures. Treat the
  generator as a starting point, not ground truth.
- BLT has three transformer stacks; a single depth knob can hide important trade-offs.
- Estimated FLOPs can diverge from measured throughput, especially with patching.

---

## 7. Document-aware data pipeline

- **Status:** Open
- **Area:** Data
- **Depends on:** nothing; 3 uses the sharding hooks

### Why

The current pipeline concatenates source files into one flat byte stream and slices
non-overlapping windows. A window can therefore cross a document boundary, even though the
BLT paper says local attention must not cross document boundaries. The corpus is also loaded
into memory as one buffer, and the sampler has no resumable state.

Prior art: nanochat uses BOS-aligned best-fit packing, document-boundary handling, shard
downloads, distributed row-group sharding, and resumable dataloader state
(`nanochat/dataloader.py`, `nanochat/dataset.py`).

### Current state

- `fblt/scripts/prep_stack_smol.py` writes files separated by a single newline.
- `fblt/scripts/prep_tinystories.py` writes a flat byte corpus.
- `csrc/train_blt_d.c` computes `num_windows` from the file size and calls
  `train_window_index(step, num_windows, seed)`.
- `data/manifest.json` records per-file offsets and lengths but the trainer does not use them.
- There is no attention-mask reset at document boundaries and no sampler checkpoint state.

### Implementation

1. Add a document-aware corpus format or index:

   - Reuse `manifest.json` offsets and lengths.
   - Optionally emit a boundary sentinel byte sequence that cannot appear in the corpus, or
     an explicit boundary mask array.
   - Store the boundary policy in the manifest.

2. Add `--doc-boundaries mask|sentinel|off` and `--boundary-token` if a sentinel is used.

3. Modify window construction:

   - Do not cross a document boundary when boundaries are enabled.
   - Pack multiple documents into a window with a boundary mask, or pad and mask the tail.
   - Reset local attention at every boundary.
   - Keep the patcher from merging patches across boundaries.

4. Add streaming and sharding:

   - Memory-map the corpus instead of loading it into one allocation.
   - Add `--corpus-shards N`, `--shard-index R`, and `--shard-count N`.
   - Add `--data-order sequential|shuffle|epoch`.
   - For a future DDP run, shard by document or by window using a deterministic global index.

5. Add resumable sampler state:

   - Persist epoch, document index, window index, and RNG state in the checkpoint.
   - On resume, reproduce the same next window as an uninterrupted run.
   - Keep the state in the optimizer metadata or a sidecar JSON, and document which wins.

6. Add packing diagnostics to the startup log: documents, windows, utilization, boundary
   count, and padding fraction.

### Acceptance

- No training window crosses a document boundary when boundary handling is enabled.
- Local attention does not attend across a boundary in a unit test.
- A packed window with multiple documents produces the same loss as the equivalent
  separately masked documents.
- Resume reproduces the next window exactly.
- A corpus larger than RAM can be trained with the streaming path.
- `--doc-boundaries off` reproduces the old behavior.

### Tests and docs

- Unit test for window construction around document boundaries.
- Unit test for packing utilization and tail padding.
- Resume regression test.
- Document the corpus format, manifest fields, boundary modes, and sharding semantics in
  `docs/DATA.md`.
- Update the README "Using your own dataset" section.

### Risks

- The patcher may merge bytes across boundaries even if attention is masked. Test at the
  patch level, not only at the token level.
- A sentinel byte can collide with real data. Prefer an explicit mask or a reserved sequence
  with validation.
- Variable-length packing complicates batch shapes. Start with fixed windows and masks, then
  add best-fit packing.
- Entropy-LM patching must use the same boundary policy at training and inference.

---

## 8. Batched inference and KV cache

- **Status:** Open
- **Area:** Inference
- **Depends on:** nothing; 12 uses this for rollouts

### Why

`--num-samples` currently loops over samples sequentially, and the greedy path recomputes the
prompt for each sample. The KV cache is used by the self-speculation path, not by plain greedy
decoding. That makes multi-sample generation and inference benchmarks slower than they need
to be.

Prior art: nanochat's `Engine` prefills once, expands the KV cache across samples, and decodes
the batch together (`nanochat/engine.py`).

### Current state

- `csrc/infer.c` owns the CLI and the generation loop.
- `csrc/infer/kv_cache.{h,c}` implements the cache used by self-speculation.
- `--num-samples` repeats the method dispatch and writes samples sequentially.
- `generate_greedy` recomputes the forward pass instead of reusing a cache.
- `infer_bench` measures the current paths.

### Implementation

1. Add a batch dimension to the KV cache:

   ```c
   blt_kv_cache *blt_kv_cache_create(size_t batch, size_t max_len, ...);
   ```

   Update the self-speculation path to use `batch=1` so its behavior is unchanged.

2. Add a batched prefill:

   - Encode the prompt once.
   - Expand the resulting cache to `num_samples` rows.
   - Give each row its own sampling RNG stream.

3. Add batched greedy decode:

   - Run one forward per byte across all live rows.
   - Select the next byte per row with the existing `blt_decode_select`.
   - Retire rows that hit an end condition.
   - Keep the single-sample path byte-identical.

4. Use the batched path for `--num-samples`, `infer_bench`, and future RL rollouts.

5. Add per-row statistics and aggregate them once. Print per-row lines only in verbose mode.

6. Add CUDA kernels for the batched cache layout. Keep a CPU reference for parity tests.

### Acceptance

- `--num-samples N` produces the same bytes as N sequential runs with the same per-sample
  seeds.
- Greedy output with `--num-samples 1` is byte-identical to the current path.
- Batched decode improves throughput on `infer_bench` at a fixed prompt length.
- KV-cache memory scales linearly with batch size and is reported in the log.
- Self-speculation and block-diffusion paths still pass their existing tests.

### Tests and docs

- Unit test for batched cache indexing and rollover.
- Integration test comparing batched and sequential samples.
- Benchmark entry with batch size, prompt length, and hardware.
- Document the batch layout, memory cost, and sampling semantics in `docs/CLI_REFERENCE.md`
  and `docs/BENCHMARKS.md`.

### Risks

- The cache layout change touches every attention path. Add parity tests before optimizing.
- Per-row RNG streams must be independent; reusing one stream changes sample diversity.
- Memory grows with batch size. Bound it and fail with a clear error.
- Non-greedy methods have their own verification logic; keep them out of the first batched
  milestone.

---

## 9. Checkpoint ergonomics

- **Status:** Open
- **Area:** Checkpoints
- **Depends on:** nothing

### Why

Checkpoint metadata v2 and optimizer-state v2 are implemented, but the workflow is still
manual: one path per save, no automatic discovery of the latest or best checkpoint, no
dataloader state, and no config compatibility shims for older checkpoints. A public release
benefits from resumable, self-describing runs.

Prior art: nanochat writes per-step model, metadata, and per-rank optimizer files, finds the
last/largest checkpoint, stores dataloader and loop state, and patches old configs on load
(`nanochat/checkpoint_manager.py`, `scripts/base_train.py`).

### Current state

- `csrc/models/checkpoint.c` writes and reads v2 metadata.
- `csrc/train_optim.c` writes and reads the FBOP optimizer state.
- `blt_format_save_path` substitutes the first `%d` in a path.
- `--save-weights`, `--save-optim`, `--load-weights`, and `--load-optim` are explicit.
- The trainer does not persist best metric/step, dataloader state, or loop state.
- There is no `model_info` command.

### Implementation

1. Add a checkpoint directory convention:

   ```text
   runs/<name>/checkpoints/
     model_000100.fblt
     meta_000100.json
     optim_000100.fbop
     best.fblt
     best.fbop
     latest -> model_000100.fblt
   ```

   Keep explicit file paths working for compatibility.

2. Add `--save-dir`, `--save-tag`, and `--resume auto|latest|best|PATH`.

3. Persist in metadata:

   - Best metric name, value, and step.
   - Optimizer and scheduler state identifiers.
   - Dataloader state (epoch, document/window index, RNG).
   - Loop state (step, tokens consumed, wall-clock start).
   - Git commit and package version.

4. Add config compatibility shims:

   - Missing fields get documented defaults.
   - Conflicting fields produce a warning.
   - Keep a `config_version` field and a migration table.

5. Add atomic writes: write to a temporary file, fsync, then rename. Never leave a partial
   checkpoint as the latest.

6. Add `tools/model_info.c` or extend `sanity_check` to print metadata, tensor summary, and
   parameter counts without loading the full model.

7. For multi-GPU, write per-rank optimizer shards and let rank 0 write the model/metadata.

### Acceptance

- `--resume auto` finds the latest valid checkpoint and continues with the stored step.
- `--resume best` loads the best-metric checkpoint when one exists.
- Metadata round-trips and is readable without the sidecar config.
- A killed save leaves the previous checkpoint intact.
- Old checkpoints load with defaults or a clear migration error.
- `model_info` prints architecture, step, best metric, and tensor counts.

### Tests and docs

- Round-trip test for every metadata field.
- Test for `auto`/`best` discovery with synthetic directories.
- Test that a truncated temporary file is ignored.
- Test config migration from the previous format.
- Document the directory layout, metadata schema, resume rules, and atomic-write behavior in
  `docs/CHECKPOINTS.md`.

### Risks

- Changing the default layout can break existing scripts. Keep explicit paths and add the new
  layout as opt-in.
- Metadata and sidecar config can disagree. Define precedence: CLI, metadata, sidecar, error.
- Symlinks on Windows need a fallback; use a pointer file if symlinks are unavailable.

---

## 10. Byte-level SFT and chat

- **Status:** Open
- **Area:** Post-training
- **Depends on:** 7 (boundary-aware data), 4 (evaluation)

### Why

fBLT can continue pretraining from a checkpoint, but it has no supervised fine-tuning path
for conversations. A byte-level model needs a chat template that is unambiguous in raw bytes
and a loss mask that trains only on assistant turns.

Prior art: nanochat's `chat_sft.py` renders conversations, masks non-assistant tokens, packs
sequences, and evaluates chat quality (`scripts/chat_sft.py`, `nanochat/tokenizer.py`).

### Current state

- `configs/train/finetune.yaml` and `--load-weights` support continued pretraining.
- There is no JSONL conversation reader, role template, or assistant-only loss mask.
- `--freeze encoder|global|decoder` is described in the old finetune notes but is not
  implemented.
- There is no chat evaluation.

### Implementation

1. Define a byte-level chat template:

   ```text
   <|system|>...<|end|>
   <|user|>...<|end|>
   <|assistant|>...<|end|>
   ```

   Use a reserved multi-byte sentinel sequence and validate that it does not appear in the
   training text. Document the exact bytes.

2. Add an SFT data reader:

   - JSONL records with `messages: [{role, content}, ...]`.
   - Render the template.
   - Mask loss on system/user bytes and train only on assistant bytes plus the end marker.
   - Optionally train on the end marker so the model learns to stop.

3. Add sequence packing:

   - Pack multiple conversations into a window with boundary masks.
   - Do not let attention cross conversations.
   - Track per-token loss weights through packing.

4. Add flags:

   ```text
   --sft-data FILE
   --sft-template FILE
   --mask-roles
   --init-lr-frac F
   --freeze encoder|global|decoder
   --task-mixture NAME=WEIGHT
   ```

5. Implement freezing in the optimizer first: skip frozen parameter groups. Skipping the
   backward pass is a later optimization.

6. Add a chat evaluation split:

   - Held-out SFT loss.
   - Prompt completion samples.
   - Optional task scores using item 4.

7. Add a minimal `fblt/chat_cli.py` after the training path works (see item 17).

### Acceptance

- A small JSONL conversation set trains without crossing conversation boundaries.
- Loss is computed only on assistant bytes; a unit test checks the mask.
- `--freeze encoder` leaves encoder weights bit-identical after a step.
- The model produces coherent multi-turn output on a held-out prompt after SFT.
- A checkpoint saved from SFT resumes correctly.

### Tests and docs

- Unit test for template rendering and mask construction.
- Unit test for packing with multiple conversations.
- Unit test for freezing.
- Integration test that trains a few steps on a synthetic conversation set.
- Document the template bytes, JSONL schema, masking rules, and freezing semantics in
  `docs/SFT.md`.

### Risks

- A sentinel sequence can collide with raw text. Reserve and validate it.
- Catastrophic forgetting is likely with a narrow SFT set. Mix in pretraining data and
  evaluate held-out BPB.
- Byte-level chat templates consume context quickly. Document the effective context budget.
- Do not train on user/system bytes; the model will learn to hallucinate both sides.

---

## 11. Sandboxed tool execution

- **Status:** Open
- **Area:** Post-training
- **Depends on:** 10 (chat template), 12 (optional)

### Why

A chat model becomes more useful when it can call a calculator or run a short program. This is
also needed for code benchmarks such as HumanEval. Execution must be isolated from the host.

Prior art: nanochat has a subprocess-based Python execution helper and a calculator tool
(`nanochat/execution.py`, `nanochat/engine.py`).

### Current state

- fBLT has no tool-call format, tool runtime, or sandbox.
- HumanEval evaluation (item 4) would need one.

### Implementation

1. Define a tool-call block in the byte-level chat template, for example:

   ```text
   <|tool|>{"name":"calculator","args":{"expression":"2+2"}}<|end|>
   ```

2. Implement a tool registry with a narrow interface:

   ```python
   def run(name: str, args: dict) -> str: ...
   ```

3. Start with a calculator and a Python subprocess tool.

4. Sandbox the subprocess:

   - Wall-clock timeout.
   - CPU and memory rlimits.
   - No network.
   - Temporary working directory.
   - Restricted environment variables.
   - Explicitly document that this is not a security boundary against a determined attacker.

5. Integrate with the chat loop:

   - Model emits a tool call.
   - Runtime executes it.
   - Result is appended to the context as a tool message.
   - Model continues.

6. Add tests for timeout, memory limit, filesystem isolation, and destructive calls.

### Acceptance

- The calculator returns correct results for basic expressions.
- A Python snippet runs and returns stdout.
- Timeout and memory limits terminate the subprocess.
- Network access is unavailable.
- The chat loop handles a tool result without losing the conversation.
- Documentation warns that the sandbox is a usability boundary, not a hardened security
  boundary.

### Tests and docs

- Unit tests for argument validation and error formatting.
- Integration test for a two-turn tool conversation.
- Document the tool protocol, supported tools, limits, and security model in `docs/TOOLS.md`.

### Risks

- Subprocess sandboxes are easy to escape. Do not run untrusted code on a host with secrets.
- Tool output can be huge; truncate and mark it.
- A model can loop on tool calls; add a maximum call count.
- Platform differences in rlimits and process isolation need documented fallbacks.

---

## 12. RL / GRPO

- **Status:** Open
- **Area:** Post-training
- **Depends on:** 8 (batched inference), 10 (chat format), 4 (task evaluation)

### Why

Supervised fine-tuning teaches format and style. Reinforcement learning can improve reasoning
and task success when a verifiable reward exists. GRPO is a practical starting point because
it avoids a separate value model.

Prior art: nanochat's `chat_rl.py` samples multiple rollouts per prompt, computes a
group-mean advantage, and optimizes a policy-gradient loss on GSM8K (`scripts/chat_rl.py`).

### Current state

- fBLT has no rollout engine, reward functions, or policy-gradient loss.
- Batched inference (item 8) is a prerequisite for efficient rollouts.

### Implementation

1. Build a rollout loop on top of batched inference:

   - Sample `G` completions per prompt.
   - Store prompt, completion, and reward.
   - Use the same chat template as SFT.

2. Add reward functions:

   - Exact-match/numeric reward for GSM8K.
   - Unit-test reward for code tasks.
   - Format reward for valid tool calls or answer tags.

3. Implement GRPO:

   - Compute the group mean reward.
   - Use `reward - group_mean` as the advantage.
   - Optimize a token/byte-level policy-gradient loss with the SFT model as reference.
   - Optional KL penalty to the reference model.
   - Mask loss to completion bytes only.

4. Add flags:

   ```text
   --rl-data FILE
   --num-rollouts G
   --max-new-bytes N
   --reward NAME
   --kl-coef F
   --init-lr-frac F
   --examples-per-step N
   ```

5. Add pass@k evaluation and reward curves.

6. Start with a small synthetic task and GSM8K before attempting code RL.

### Acceptance

- The rollout loop produces `G` independent completions per prompt.
- Group-mean advantages are zero-mean within each group.
- A tiny synthetic task shows reward improvement over a short run.
- Evaluation reports pass@1 and pass@k.
- Training is reproducible with a fixed seed and fixed rollout count.

### Tests and docs

- Unit test for advantage computation and loss masking.
- Unit test for reward functions.
- Integration test on a synthetic task with a known optimum.
- Document the algorithm, reward definitions, rollout format, and hyperparameters in
  `docs/RL.md`.

### Risks

- RL is unstable at small scale. Start with a verifiable, low-variance task.
- Reward hacking is likely; inspect completions, not just scores.
- Rollouts are expensive; batched inference and a cap on new bytes are required.
- The reference model doubles memory unless it shares weights or runs on CPU.

---

## 13. FP8 / FP16 training

- **Status:** Open
- **Area:** Training
- **Depends on:** 2 (BF16), 5 (optimizer groups)

### Why

BF16 is the first mixed-precision milestone. FP8 can reduce matmul cost further on supported
hardware, and FP16 is a fallback for older GPUs. Both need explicit scaling and a clear
fallback path.

Prior art: nanochat has an FP8 training module with rowwise/tensorwise scaling and a module
filter (`nanochat/fp8.py`), and selects compute dtype from the device capability
(`nanochat/common.py`).

### Current state

- BF16 dtype and cast kernels exist.
- FP32 and BF16 matmul paths exist.
- There is no FP8 dtype, scaling, or GEMM path.
- There is no FP16 loss scaling.

### Implementation

1. Add FP8 E4M3/E5M2 storage and a GEMM path for supported CUDA architectures.
2. Implement rowwise or tensorwise scaling:

   - Track amax history per tensor.
   - Compute scale factors before the GEMM.
   - Keep a master FP32 copy for the optimizer.

3. Apply FP8 only to eligible matmuls: dimensions divisible by the required alignment and
   above a minimum size. Keep norms, softmax, cross-entropy, and optimizer state in FP32.

4. Add `--dtype fp32|bf16|fp16|fp8` and a capability check with a clear error or fallback.

5. For FP16, add loss scaling and master FP32 weights.

6. Add numerical parity and long-run stability tests.

### Acceptance

- FP8 runs only on supported hardware and falls back cleanly otherwise.
- A short run has no NaN/Inf and tracks the BF16/FP32 loss within tolerance.
- Scale factors are finite and update without overflow.
- Checkpoints remain FP32 and load into any dtype.
- A benchmark records the speedup and hardware.

### Tests and docs

- Unit test for scale computation and amax updates.
- Unit test for the module filter.
- Integration test comparing FP32, BF16, FP16, and FP8 on a tiny model.
- Document supported GPUs, scaling modes, fallbacks, and numerical expectations in
  `docs/MIXED_PRECISION.md`.

### Risks

- FP8 support is architecture-specific; do not advertise it universally.
- Scaling bugs can silently produce zeros or infinities.
- FP16 overflow needs loss scaling; BF16 does not.
- More dtype paths multiply the test matrix.

---

## 14. Dataset download and cache

- **Status:** Open
- **Area:** Data
- **Depends on:** 7 (manifest and boundary policy)

### Why

Data preparation is currently a collection of scripts with manual downloads and no shared
cache. Contributors cannot reliably reproduce a corpus from a name and checksum.

Prior art: nanochat downloads dataset shards on demand with retries and file locks, keeps a
manifest, and exposes a cache directory (`nanochat/dataset.py`, `nanochat/common.py`).

### Current state

- `fblt/scripts/prep_stack_smol.py` and `prep_tinystories.py` prepare specific corpora.
- `data/manifest.json` exists for the sample corpus.
- There is no registry, cache directory, retry logic, or file lock.
- Large corpora are not downloaded automatically.

### Implementation

1. Add `fblt/dataset.py` with a registry:

   ```python
   @dataclass
   class DatasetSpec:
       name: str
       url: str
       sha256: str
       shards: int
       license: str
   ```

2. Use a cache directory:

   ```text
   $FBLT_CACHE (default ~/.cache/fblt)
     datasets/<name>/
       manifest.json
       shard-00000.bin
       ...
   ```

3. Download with:

   - HTTP range/resume support where available.
   - Retries with backoff.
   - A lock file per dataset to prevent concurrent downloads.
   - Checksum verification before publishing the manifest.
   - Manifest written last, so a partial download is never considered complete.

4. Add a CLI:

   ```bash
   python -m fblt.dataset --list
   python -m fblt.dataset --download tinystories
   python -m fblt.dataset --verify tinystories
   ```

5. Keep the output as raw bytes plus a manifest. Parquet/JSONL inputs are converted by the
   prep scripts, not by the C trainer.

### Acceptance

- A fresh cache can download and verify a dataset with one command.
- Re-running the command is a no-op.
- Two concurrent downloads do not corrupt the cache.
- A bad checksum fails before the manifest is written.
- The manifest records license, source, size, and SHA-256.

### Tests and docs

- Test cache hit/miss with a local HTTP fixture.
- Test lock behavior with two processes.
- Test checksum failure and partial-file cleanup.
- Document the cache layout, environment variable, and available datasets in
  `docs/DATASETS.md`.

### Risks

- Dataset licenses differ; record them and do not bundle restricted data.
- Network failures and redirects need clear errors.
- Large caches can fill a disk; report free space before download.

---

## 15. Reproducible speedrun and leaderboard

- **Status:** Open
- **Area:** Tooling
- **Depends on:** 4 (evaluation), 6 (scaling dial)

### Why

The README has a quickstart, but there is no single end-to-end recipe that builds, prepares
data, trains, evaluates, and generates from a checkpoint. There is also no shared place to
compare runs across hardware and configs.

Prior art: nanochat ships `runs/speedrun.sh`, `runs/miniseries.sh`, and
`runs/scaling_laws.sh`, plus CSV-based result aggregation.

### Current state

- `README.md` documents individual commands.
- `scripts/` contains environment and preparation helpers.
- `bench/` contains benchmark tools and plotting helpers.
- There is no end-to-end script or leaderboard.

### Implementation

1. Add `scripts/speedrun.sh`:

   - Build CPU or CUDA.
   - Prepare the sample corpus.
   - Train a fixed tiny config.
   - Evaluate BPB.
   - Generate a sample.
   - Write a JSON summary with hardware, commit, config, and metrics.

2. Add `scripts/miniseries.sh` for a small set of configs and seeds.

3. Add `scripts/scaling_laws.sh` that uses the scaling dial (item 6) to generate target-FLOP
   runs and aggregates the results.

4. Add `bench/leaderboard.py` or `docs/LEADERBOARD.md`:

   - Columns: model, params, data, steps, hardware, backend, BPB, task scores, acceptance,
     throughput, commit.
   - Accept JSONL from the eval harness.
   - Mark synthetic or non-reproducible entries.

5. Add a contribution template for leaderboard entries. Require a config and command.

6. Run the speedrun in CI on the sample corpus.

### Acceptance

- A fresh clone can run the speedrun end-to-end with documented prerequisites.
- The summary JSON is valid and includes the git commit and hardware.
- The leaderboard updates from JSONL without manual editing.
- CI runs the speedrun on a tiny config.
- The README quickstart links to the speedrun for a complete example.

### Tests and docs

- Shell syntax checks in CI.
- A dry-run mode that prints each command without training.
- Document prerequisites, expected outputs, and how to submit a leaderboard entry.

### Risks

- Hardware differences make wall-clock comparisons noisy. Separate throughput from quality.
- Dataset and tokenizer differences break comparability. Record both.
- A leaderboard can incentivize overfitting to the benchmark. Keep evaluation splits hidden
  or documented.

---

## 16. Peak FLOPS/BW table and MBU

- **Status:** Open
- **Area:** Tooling
- **Depends on:** nothing

### Why

MFU is currently computed from a manually supplied `--peak-tflops`. That makes cross-device
comparisons easy to get wrong, and inference has no memory-bandwidth utilization number at
all.

Prior art: nanochat has a device table for peak FLOPS and HBM bandwidth and uses it for MFU
and MBU (`nanochat/common.py`).

### Current state

- Training telemetry accepts `--peak-tflops`.
- `core/flops.h` estimates forward FLOPs.
- `infer_bench` reports wall-clock and acceptance, but not MBU.
- There is no device table.

### Implementation

1. Add `csrc/core/device_info.c` with a table keyed by CUDA device name:

   ```c
   typedef struct {
       const char *name;
       double fp32_tflops;
       double bf16_tflops;
       double fp8_tflops;
       double hbm_gbps;
   } blt_device_spec;
   ```

2. Match the device name reported by `cudaDeviceGetAttribute`/`cudaGetDeviceProperties`.
   Fall back to "unknown" and print a warning.

3. Use the table for:

   - Training MFU (`estimated_flops / time / peak`).
   - Inference MBU (`bytes_moved / time / bandwidth`).
   - Optional energy estimates later.

4. Keep `--peak-tflops` and add `--peak-bandwidth` as explicit overrides.

5. Print the matched device, peak values, MFU, and MBU in startup and `RESULT`.

6. Add a test that parses known device names and returns the expected entry.

### Acceptance

- A known GPU prints MFU without `--peak-tflops`.
- An unknown GPU prints a warning and omits MFU rather than dividing by zero.
- Inference reports MBU for the measured path.
- Overrides take precedence over the table.
- The table is documented with sources for each value.

### Tests and docs

- Unit test for name matching and fallback.
- Unit test for the MFU/MBU formulas with known inputs.
- Document the table, assumptions (clocks, MIG, power limits), and how to add a device.

### Risks

- Vendor peak numbers assume boost clocks and may not match a laptop or thermally limited
  device. Mark them as reference peaks.
- MIG instances and virtual GPUs report different names.
- MBU depends on an accurate bytes-moved model; document what is counted.

---

## 17. Interactive chat CLI

- **Status:** Open
- **Area:** Inference
- **Depends on:** 10 (byte-level chat template), 8 (batched inference) for multiple sessions

### Why

`fblt-infer` is one-shot. A chat CLI makes the 1.0 checkpoint easier to try and gives SFT/RL
work a manual evaluation surface.

Prior art: nanochat's `chat_cli.py` keeps a rolling conversation, streams output, and exposes
sampling controls.

### Current state

- `fblt-infer` accepts a prompt and writes generated bytes.
- There is no persistent context, role formatting, or interactive commands.
- The chat template does not exist yet (item 10).

### Implementation

1. Add `fblt/chat_cli.py`:

   - Load a checkpoint and optional entropy LM.
   - Render the chat template.
   - Keep a rolling byte context with a documented truncation policy.
   - Stream generated bytes.
   - Handle `/reset`, `/system`, `/temp`, `/top-k`, `/top-p`, `/seed`, `/quit`.

2. Add a non-interactive mode for scripted tests:

   ```bash
   fblt-chat --checkpoint ... --messages messages.jsonl --output out.jsonl
   ```

3. Use the same sampling options as `fblt-infer`.

4. Add tool calls when item 11 lands.

5. Enforce a maximum context and show a warning when truncating.

### Acceptance

- A multi-turn conversation keeps context across turns.
- `/reset` clears history.
- Sampling changes take effect on the next turn.
- A scripted conversation produces deterministic output with a fixed seed.
- The CLI works on CPU and CUDA.

### Tests and docs

- Test scripted stdin/stdout with a tiny checkpoint.
- Test context truncation at a boundary.
- Test command parsing.
- Document the commands, template, context limit, and sampling semantics in
  `docs/CHAT.md`.

### Risks

- Byte-level context is large; truncation can cut a template marker.
- Streaming output needs careful flushing.
- A chat UI can hide model failures; include a raw mode for debugging.

---

## 18. Experiment log and negative results

- **Status:** Done
- **Area:** Docs
- **Depends on:** nothing

### Why

fBLT has ablation tables, but no chronological record of what was tried, what failed, and
under which conditions. Negative results are especially valuable because they prevent
contributors from repeating dead ends.

Prior art: nanochat keeps `dev/LOG.md` with dated experiments and conclusions.

### Current state

- `docs/ABLATIONS.md` has structured ablation results.
- `docs/experiments/` has individual investigation notes.
- There is no dated, append-only log or template.

### Implementation

1. Add `docs/EXPERIMENT_LOG.md` with one entry per experiment:

   ```markdown
   ## YYYY-MM-DD - Short title

   - Commit:
   - Hardware:
   - Config:
   - Command:
   - Metric:
   - Result:
   - Conclusion:
   - Follow-up:
   ```

2. Record both positive and negative results.

3. Link raw configs and JSONL where possible. Do not paste large logs.

4. Add a short template to `CONTRIBUTING.md`.

5. Reference the log from the README community section.

### Acceptance

- The log has the current date, commit, hardware, command, and metric for each entry.
- Negative results are labeled as such.
- Entries link to reproducible configs.
- The template is documented.

### Risks

- Logs rot if they omit configs or commits.
- A chronological log can duplicate `ABLATIONS.md`; link rather than copy.
- Do not include secrets or private dataset paths.

---

## 19. Test coverage for new subsystems

- **Status:** Open
- **Area:** Testing
- **Depends on:** the corresponding feature items

### Why

The existing suite covers the 1.0 ops, models, and parity paths. The items in this file add
optimizers, data loaders, batched inference, evaluation, and post-training code. Each needs
targeted tests, including CPU/CUDA parity where applicable.

Prior art: nanochat has dedicated tests for optimizer steps, engine cache behavior, task
slicing, and sandbox limits.

### Current state

- The C suite is registered in `tests/test_main.c` with suite flags.
- Parity goldens live in `data/` and are generated by `tests/py/`.
- Python tests cover the wrapper and some utilities.
- New features in this file have no tests yet.

### Implementation

Add focused tests alongside each feature:

1. Gradient accumulation: zero/add/scale, averaged gradient, resume.
2. BF16: refresh, numerical parity, checkpoint interoperability.
3. Multi-GPU: all-reduce reference, single-rank no-op, data-index uniqueness.
4. Evaluation: prompt rendering, answer extraction, byte-NLL scoring, JSON schema.
5. Muon: group assignment, orthogonalization, cautious decay, AdamW parity.
6. Scaling dial: dimension rounding, parameter count, golden config.
7. Data pipeline: boundary masks, packing, streaming, resume.
8. Batched inference: cache indexing, batched vs sequential parity.
9. Checkpoints: metadata round-trip, auto-resume, atomic writes, migration.
10. SFT: template rendering, loss mask, freezing.
11. Tool execution: timeout, memory limit, network isolation.
12. RL: advantage computation, reward functions, deterministic rollout.
13. FP8/FP16: scaling, fallback, numerical parity.
14. Dataset cache: hit/miss, lock, checksum.
15. Speedrun: dry-run and summary schema.
16. Device table: name matching, MFU/MBU formulas.
17. Chat CLI: scripted conversation, truncation.
18. Experiment log: template validation (optional).

### Acceptance

- Every new public flag has at least one CLI or unit test.
- Numerical features have a CPU reference and, where applicable, a CUDA parity test.
- Tests run from the repository root.
- CI runs the CPU suite, Python tests, and a compile check for CUDA-only code.
- Flaky tests are fixed or marked with a documented reason.

### Tests and docs

- Follow `docs/TEST_API_REFERENCE.md`.
- Add test files under `tests/unit/<area>/` and register them in `tests/test_main.c`.
- Generate any new parity goldens with a script in `tests/py/`.
- Document how to run a single suite by editing `tests/test_main.c`.

### Risks

- GPU tests may not be available in CI. Provide a manual validation matrix.
- Golden files can hide implementation drift. Regenerate them only with the reference script.
- Tests that depend on random data need fixed seeds and tolerances.

---

## 20. Architecture experiments

- **Status:** Exploratory
- **Area:** Research
- **Depends on:** 6 (scaling dial), 18 (experiment log)

### Why

nanochat's architecture contains several ideas that are cheap to ablate and may interact with
BLT's patching and diffusion objectives. These are experiments, not committed changes. Each
one needs a baseline, multiple seeds, and a documented result.

### Candidate experiments

- Sliding-window attention patterns in the global transformer, for example a repeating
  `SSSL` pattern.
- Grouped-query attention (GQA) to reduce KV-cache size and decode cost.
- QK normalization for stability at higher learning rates.
- Logit softcap to bound the output distribution.
- Zero-initialized output projections for residual stability.
- Per-layer residual scaling or an `x0` blend for deep global stacks.

### Not recommended

These appeared in the nanochat experiment log but are not good first candidates for fBLT:

- Mixture-of-experts routing.
- SwiGLU changes (BLT already uses a gated FFN; confirm before duplicating).
- DyT normalization.
- Batch-size ramp.
- Skipping AdamW every other step.
- Partial RoPE.
- Orthogonal initialization.
- Bigram hash embeddings (fBLT already has n-gram hash embeddings).
- Value embeddings, smear, and backout (token-embedding tricks with low relevance to a
  byte-level model).

### Implementation

1. Add each experiment behind a config flag, default off.
2. Run a baseline and the variant with at least three seeds at a fixed token budget.
3. Report BPB, masked accuracy, throughput, memory, and any stability issues.
4. Record the result in `docs/EXPERIMENT_LOG.md` and, if it is a full ablation, in
   `docs/ABLATIONS.md`.
5. Do not change a default without a clear, replicated result.

### Acceptance

- Each experiment has a config, command, seeds, and metrics.
- The baseline is run on the same hardware and data.
- Results include negative outcomes.
- Any default change is justified in the experiment log.

### Risks

- Small-scale results may not transfer to larger models.
- Architecture changes can interact with patching and diffusion in non-obvious ways.
- Multiple simultaneous changes make attribution impossible. Test one axis at a time.
---

## 21. Hugging Face checkpoint and model integration

**Status:** Open
**Area:** Integration

### Why

The official BLT repository distributes 1B and 7B checkpoints through Hugging Face and provides
loading helpers for `LMTransformer` and `ByteLatentTransformer`. fBLT currently uses its own
`.fblt`/`.fbop` formats and metadata.

### Current state

- Checkpoint v2 stores architecture config, training step, and best metric.
- `fblt-infer` auto-detects dimensions from metadata or `resolved_config.yaml`.
- There is no Hugging Face loader, model card, or `transformers` integration.

### Implementation

1. Define a stable export/import mapping between fBLT tensors and a Hugging Face-compatible
   layout.
2. Add `fblt export-hf` and `fblt import-hf` commands or Python helpers.
3. Document required metadata: architecture, patching, entropy LM, block size, and training
   config.
4. Add tests that round-trip a tiny checkpoint through the new format.

### Acceptance

- A tiny fBLT checkpoint can be exported and reloaded with identical outputs.
- The exported metadata is sufficient to reconstruct inference settings.
- The workflow is documented in `docs/CHECKPOINTS.md` or `docs/STARTER_GUIDE.md`.

### Risks

- Hugging Face APIs and model formats change over time.
- Exporting large artifacts should not require committing them to the repository.

---

## 22. FineWeb/DCLM corpus pipeline

**Status:** Open
**Area:** Data

### Why

The official BLT stack trains on FineWeb and DCLM-style corpora. fBLT currently supports flat
byte corpora and the TinyStories/source-code sample pipeline.

### Current state

- `fblt/scripts/prep_stack_smol.py` and `prep_tinystories.py` prepare specific corpora.
- `data/manifest.json` records per-file offsets and lengths.
- There is no FineWeb/DCLM downloader, shuffler, or document-aware preprocessor.

### Implementation

1. Add a documented pipeline for downloading and preparing FineWeb/DCLM bytes.
2. Reuse the manifest and boundary-aware window logic from item 7.
3. Record dataset version, license, preprocessing, and split in the manifest.
4. Add a small smoke corpus for tests.

### Acceptance

- A documented command prepares a held-out byte corpus from a named source.
- The manifest records provenance and checksums.
- Training and evaluation can use the resulting corpus without manual edits.

---

## 23. Tokenizer/BPE hybrid patching modes

**Status:** Open
**Area:** Data

### Why

The official BLT patcher supports multiple patching modes, including entropy, BPE, BPE
patcher, space, static, and byte modes. fBLT currently supports byte, fixed-stride, and
entropy patching.

### Current state

- `fblt/scripts/build_sample_corpus.py` builds a byte corpus.
- Training supports `--entropy-patches`.
- Inference selects fixed-stride-4 or entropy patching from `resolved_config.yaml`.

### Implementation

1. Survey which hybrid modes are meaningful for fBLT's byte-level objective.
2. Implement one well-documented mode at a time, starting with space or static patching.
3. Keep patcher and inference segmentation consistent.
4. Add unit tests for patch boundaries and attention masks.

### Acceptance

- At least one additional patching mode trains and infers correctly.
- The mode is selectable through config and documented.
- Boundary behavior is tested.

---

## 24. Official-scale configuration and scaling matrix

**Status:** Open
**Area:** Training

### Why

The official BLT stack includes configurations and scaling experiments across model sizes and
training budgets. fBLT has a scaling dial proposal but no matrix of larger configurations.

### Current state

- `fblt/scripts/gen_configs.py` generates ablation configs.
- `train_sweep.c` runs config-driven sweeps.
- Item 6 proposes a compute-optimal scaling dial.

### Implementation

1. Add documented configs for several larger model sizes.
2. Record parameter counts, token budgets, and hardware assumptions.
3. Reuse the scaling dial rather than hand-writing each config.
4. Add a small CPU smoke config for CI.

### Acceptance

- Generated configs are runnable without manual edits.
- Parameter counts and budgets are recorded in manifests.
- The matrix is linked from `docs/BENCHMARKS.md` or `docs/ABLATIONS.md`.

---

## 25. Tensor parallelism and multi-node training

**Status:** Open
**Area:** Training

### Why

The official BLT stack uses distributed training with FSDP and tensor parallelism. fBLT's
multi-GPU item currently covers data parallelism only.

### Current state

- `MULTI_GPU.md` documents a data-parallel design.
- Item 3 covers NCCL data parallelism.
- There is no tensor-parallel or multi-node implementation.

### Implementation

1. Complete basic data parallelism first.
2. Profile communication bottlenecks at larger model sizes.
3. Add tensor parallelism only if profiling shows it is necessary.
4. Document launcher and shared-filesystem requirements.

### Acceptance

- A single-node multi-GPU run is validated before multi-node work.
- Tensor parallelism is justified by measurements.
- Launch instructions are tested on at least one cluster.

---

## 26. Backward-pass and launch-overhead optimization

- **Status:** In progress — clip-norm bug fixed, clip scale folded into the optimizer, accumulator only allocated for `grad_accum > 1`; elementwise fusion and CUDA graph capture remain.
- **Area:** Training (CUDA)
- **Depends on:** 1 (gradient accumulation)

### Why

Profiling (E009) shows the backward pass is ~50% of step time on CUDA (FP32
and BF16), and the GPU is busy only ~15% of each step. The dominant costs are
host launch overhead, synchronous host-device copies, and redundant per-tensor
work, not compute.

### Measured (research config, W=64, embed=64, hidden=128, layers=2)

- CPU: 50.9 ms/step (fwd 15.1, bwd 33.7, opt 1.6)
- CUDA FP32: 26.2 ms/step (fwd ~4.8-7.8, bwd ~12.9, opt ~4.9)
- CUDA BF16: 25.1 ms/step (fwd ~4.7-6.6, bwd ~12.4, opt ~4.8)
- GPU kernel time: ~3.8 ms/step (38.3 ms / 10 steps); ~1340 kernel launches/step.
- `cudaMemset`: 573/step (5.5 ms/step GPU, ~44 ms/step host API time).
- `cudaMemcpy` (D2H): 105/step, 69 of which are 4-byte scalar downloads in
  `clip_all` (per-tensor `blt_vec_dot`).
- `cudaLaunchKernel`: ~1190/step, ~9.5 us/launch host time.

### Current state

- `blt_tensor_create` zeroes every allocation (`cudaMemset`), including
  scratch tensors that are fully overwritten.
- `blt_model_grad_zero` zeroes all gradient tensors separately; called 3 times
  per step even when `grad_accum == 1`.
- `clip_all` calls `blt_vec_dot` (a synchronous D2H) per parameter tensor.
- `blt_cuda_memset` uses the synchronous `cudaMemset`.
- `adamw_all`/`sgd_all` scale gradients per tensor (69 extra launches).

### Implementation

1. [x] Async memset: change `blt_cuda_memset` to `cudaMemsetAsync` on the
   default stream. Order is preserved because all kernels use the default
   stream.
2. [x] Skip redundant zero passes when `grad_accum == 1`: backward directly into
   `grad`, drop the accumulator zero/add/scale. The accumulator is only
   allocated when `grad_accum > 1`.
3. [x] Device-side grad norm: add `blt_vec_dot_into` that writes to a device
   pointer without D2H; `blt_grad_clip_norm` counts the parameter tensors
   first, allocates the exact-sized device array, launches all dots into it,
   and does one D2H, preserving deterministic summation order. The previous
   grow-on-demand buffer discarded the first 64 sums when the model had more
   than 64 parameter tensors (the default has 67).
4. [x] Fold clip scale into the optimizer: `adamw_all`/`sgd_all` take a
   `grad_scale` and apply it inside the update kernel, dropping the separate
   per-tensor scale pass. `adamw_update_norm` uses the same scale.
5. [x] Add `blt_tensor_create_uninit` for tensors fully overwritten by their
   producing kernel. Used at the upload sites in `train_blt_d.c`; the
   forward/backward matmul scratch tensors were left zero-initialized because
   their overwrite guarantees need checking per op.
6. [ ] Reduce elementwise launch count in backward: fuse consecutive add/scale/
   copy chains where the producing kernel can write the final result.
7. [ ] CUDA graph capture for the whole train step. Attempted; blocked by sync
   cudaMemcpy/cudaMemset during capture, host-side scratch arena state, and
   per-step inputs (t, positions, tokens) being read on host. Requires
   device-resident inputs and async copies throughout. Deferred to a dedicated
   branch (see E009).

### Acceptance

- `--grad-accum 1` step time improves measurably with no loss change.
- Gradient clipping remains bit-identical (same per-tensor values and order).
- CPU behavior unchanged (CPU path keeps synchronous `memset`/`vec_dot`).
- Parity tests still pass; no change to op numerics.

### Tests and docs

- Regression test: `grad_accum == 1` matches current trainer bit-for-bit.
- CUDA parity test for clip norm (host-summed array vs per-tensor).
- Document the measured breakdown and the new flags in `docs/BENCHMARKS.md`.

### Risks

- Async memset could hide use-after-free if a tensor is read before being
  written; only allocate uninitialized where the producer fully writes.
- CUDA graph capture is fragile with dynamic shapes (patch count varies per
  window); capture may need a fixed-shape path.

---

## Reference material

- [README.md](README.md) - build, training, inference, benchmarks.
- [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) - every training and inference flag.
- [docs/ABLATIONS.md](docs/ABLATIONS.md) - existing ablation results.
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) - performance results.
- [MULTI_GPU.md](MULTI_GPU.md) - multi-GPU design notes.
- [CONTRIBUTING.md](CONTRIBUTING.md) - branch policy and PR guidelines.
- [nanochat](https://github.com/karpathy/nanochat) - prior art for several training,
  evaluation, and post-training ideas.
