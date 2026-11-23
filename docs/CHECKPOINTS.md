# Checkpoints

`train_blt_d` writes weights, optional AdamW state, and two pointer files that name the
newest and best checkpoints. `--save-dir DIR` names the whole set; `--resume` reads it back.

## Save directory layout

```
DIR/checkpoints/
  step_000120.fblt      weights at step 120
  step_000120.fbop      AdamW moments + step counter for that save
  best_000120.fblt      weights at the step that produced the best metric
  best_000120.fbop      AdamW moments for the best step
  latest.txt            path of the most recent weights save
  best.txt              path of the best-metric weights save
```

The `06` in `%06d` is only the zero-pad width; step numbers grow past 999999 without
breaking the name. `--save-dir` supplies defaults, it does not override: an explicit
`--save-weights`, `--save-optim` or `--save-best` wins for that file.

Both pointer files contain one path, newline-terminated, exactly as it was passed to the
writer. They are written after the checkpoint they name, so a pointer never resolves to a
file that does not exist yet.

## Saving

```bash
# periodic weights + AdamW state, plus latest/best pointers
fblt-train --config configs/train/debug.yaml --backend cpu \
  --save-dir runs/debug --save-every 500 --steps 2000

# evaluation every 200 steps; improves over the best metric are saved separately
fblt-train ... --save-dir runs/debug --eval-corpus data/heldout.bin --eval-every 200 \
  --save-every 200
```

Without `--save-every` only the end-of-run checkpoint is written. `--steps 0` writes nothing:
it is the eval-only path and leaves the pointers where they are. With `--eval-corpus` and
`--eval-every > 0`, a run that improves on the tracked metric (`--best-metric bpb` by
default, or `masked_acc`) also writes `best_<step>.fblt` and updates `best.txt`. A run with
`--eval-corpus` but `--eval-every 0` evaluates once at the end and writes no best file,
because there is nothing to compare.

Each write goes to `<path>.tmp`, is flushed and `fsync`ed (POSIX) or `_commit`ed (Windows),
and only then renamed over the target. A process killed mid-save leaves the previous
checkpoint intact rather than truncating it. On Windows the target is removed before the
rename because MSVC `rename()` refuses an existing destination, so a crash in that window
leaves neither file; POSIX has no such window.

## Resuming

```bash
fblt-train --config configs/train/debug.yaml --backend cpu \
  --save-dir runs/debug --resume latest --steps 500
```

| Mode | Checkpoint | Missing pointer |
| --- | --- | --- |
| `latest` | whatever `latest.txt` names | fatal, with the expected path |
| `best` | whatever `best.txt` names | fatal |
| `auto` | `best.txt` if it resolves, else `latest.txt` | fatal when neither resolves |

`--resume` requires `--save-dir` and cannot be combined with `--load-weights` or
`--load-optim`; the trainer exits with a message rather than guessing which source wins.

On resume the trainer:

1. loads the weights from the pointer target;
2. adopts `<stem>.fbop` beside it when that file exists, restoring the AdamW moments and
   the stored step counter (otherwise it prints one line that the optimizer restarts cold);
3. sets the starting step from `--start-step` if given, else from the stored AdamW step,
   else from the `step_<digits>` file name, else from `training.step` in the metadata, and
   fails with `pass --start-step` when none of those apply.

The window order is a deterministic function of the step number and `--seed`
(`train_window_index`), so a run resumed at the right step sees a different order than the
one it stopped in, and a resumed run is bit-identical to an uninterrupted run with the same
window order and seed. That identity does not survive changing `--window`, `--grad-accum` or
the corpus between runs.

### Resuming without `--resume`

The older path still works and is what `--resume` resolves to:

```bash
fblt-train ... --load-weights runs/x/step_000500.fblt \
  --load-optim runs/x/step_000500.fbop --steps 500
```

`--load-weights` alone restarts the optimizer cold. `--start-step N` overrides the stored
counter. `--load-optim` requires `--load-weights` and requires AdamW.

## Metadata

Version 2 containers carry a JSON block. `train_blt_d` writes:

```json
{
  "format": "fblt-checkpoint",
  "format_version": 2,
  "config": {
    "embed": 32, "hidden": 64, "enc_layers": 1, "glob_layers": 1, "dec_layers": 1,
    "cross_attn": "all", "d0_mode": "learned", "diffusion": 1,
    "block_size": 8, "window": 128
  },
  "training": {"step": 1, "optimizer": "adamw", "seed": 7}
}
```

Best saves add what made the file win, inside `training`:

```json
"training": {
  "step": 1,
  "optimizer": "adamw",
  "seed": 7,
  "best_metric": {"name": "causal_bpb", "value": 8.001449, "step": 1}
}
```

Version 1 containers (written by older builds) have no metadata block. Loaders accept them;
readers report `meta: null`.

Not yet recorded: dataloader/RNG state, tokens consumed, wall-clock start, git commit,
package version, and a `config_version` for migration. A checkpoint from a future container
version is rejected with a message naming the version rather than being misread.

## Inspecting

```bash
python -m fblt.model_info runs/debug/checkpoints/best_000120.fblt
```

```json
{
  "path": "runs/debug/checkpoints/best_000120.fblt",
  "size_bytes": 123456,
  "meta": {"format": "fblt-checkpoint", "config": {...}, "training": {...}},
  "tensor_count": 36,
  "param_count": 23760,
  "step": 120,
  "best_metric": {"name": "causal_bpb", "value": 8.001449, "step": 120}
}
```

The tensor section is walked by seeking over payloads, so the command does not read a
multi-GB checkpoint into memory. A truncated payload is an error, not a partial count.

The same information is available programmatically in `fblt/_checkpoint.py`:
`read_meta`, `checkpoint_metadata`, `checkpoint_step`, `latest_checkpoint`,
`best_checkpoint`, `resume_checkpoint`, `model_info`. `latest_checkpoint` and
`best_checkpoint` honour the pointer files, validate the container (header and tensor
walk), and skip a truncated file instead of returning it. Without pointer files they fall
back to `step_<digits>` ordering and to `training.best_metric` metadata respectively.

## Hugging Face export

`fblt-export-hf` turns a v2 checkpoint into a Hugging Face directory: a real
`model.safetensors` with renamed tensors, a `config.json` that carries both HF-style keys and
the fBLT architecture, a model card, and the original metadata. `fblt-import-hf` converts it
back to a `.fblt` the C loader accepts.

```bash
fblt-export-hf --checkpoint runs/debug/checkpoints/step_000120.fblt --out runs/debug/hf
fblt-import-hf --hf-dir runs/debug/hf --out runs/debug/restored.fblt
python -m fblt.hf_export verify --checkpoint runs/debug/checkpoints/step_000120.fblt
```

This is an interchange format, not a `transformers` architecture. There is no
`PreTrainedModel` subclass for BLT, so `AutoModel.from_pretrained` will not load the
directory; the weights and metadata are there for a converter or a custom model class. What
the export does guarantee is that the tensor names, shapes and dtype are self-describing and
that nothing is lost: importing the directory reproduces the source `.fblt` byte-for-byte.

### Directory layout

```
hf/
  model.safetensors     F32 weights, names in the blt.* namespace
  config.json           HF keys plus an "fblt" block with the architecture and metadata
  fblt_metadata.json    the exact metadata bytes from the source checkpoint
  README.md             model card: dimensions, patching settings, limits
```

`safetensors` stores shapes as JSON integers and the payload as little-endian F32, so the
exported weights are portable even though the `.fblt` container writes dimensions as
`sizeof(size_t)`.

### Tensor names

The mapping is fixed and bijective. HF-style leaf names sit under a `blt.` prefix because
the architecture is not a standard one.

| `.fblt` | safetensors |
| --- | --- |
| `enc.embed` | `blt.encoder.byte_embedding.weight` |
| `enc.ngram<k>` | `blt.encoder.ngram_embeddings.<k>.weight` |
| `enc.L<i>.norm1` | `blt.encoder.layers.<i>.input_layernorm.weight` |
| `enc.L<i>.attn_qkv` | `blt.encoder.layers.<i>.self_attn.qkv_proj.weight` |
| `enc.L<i>.attn_proj` | `blt.encoder.layers.<i>.self_attn.o_proj.weight` |
| `enc.L<i>.norm2` | `blt.encoder.layers.<i>.post_attention_layernorm.weight` |
| `enc.L<i>.ffn_up` | `blt.encoder.layers.<i>.mlp.up_proj.weight` |
| `enc.L<i>.ffn_gate` | `blt.encoder.layers.<i>.mlp.gate_proj.weight` |
| `enc.L<i>.ffn_down` | `blt.encoder.layers.<i>.mlp.down_proj.weight` |
| `enc.L<i>.cross_norm` | `blt.encoder.layers.<i>.cross_attn.layer_norm.weight` |
| `enc.L<i>.cross_q/k/v` | `blt.encoder.layers.<i>.cross_attn.{q,k,v}_proj.weight` |
| `enc.L<i>.cross_proj` | `blt.encoder.layers.<i>.cross_attn.o_proj.weight` |
| `glob.L<i>.<leaf>` | `blt.global_transformer.layers.<i>.<leaf>` |
| `dec.L<i>.<leaf>` | `blt.decoder.layers.<i>.<leaf>` |
| `dec.lm_head` | `blt.lm_head.weight` |
| `dec.d0_embed` | `blt.decoder.d0_embedding.weight` |

`<leaf>` uses the same table as the encoder layer rows. The global transformer has no
cross-attention tensors. Norms are RMSNorm and carry a weight only; there are no biases.

`qkv_proj` is fused: the output dimension is `[q | k | v]`, each block `embed` wide, and
heads are interleaved inside each block. Split it on the last axis to get the three
projections.

### `config.json`

HF tools read the top-level keys; everything fBLT-specific is under `"fblt"`.

```json
{
  "architectures": ["BLTForCausalLM"],
  "model_type": "fblt",
  "format": "fblt-hf/v1",
  "vocab_size": 256,
  "hidden_size": 32,
  "intermediate_size": 64,
  "num_hidden_layers": 1,
  "num_attention_heads": 4,
  "num_key_value_heads": 4,
  "max_position_embeddings": 32,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-06,
  "rope_theta": 500000.0,
  "torch_dtype": "float32",
  "tie_word_embeddings": false,
  "fblt": {
    "format_version": 2,
    "tensor_schema": "fblt-hf-tensors/v1",
    "checkpoint_metadata": {"config": {}, "training": {}},
    "architecture": {
      "embed": 32, "hidden": 64,
      "enc_layers": 1, "glob_layers": 1, "dec_layers": 1,
      "cross_attn": "all", "d0_mode": "learned",
      "vocab_size": 256, "d0_vocab": 257,
      "ngram_tables": 2, "per_ngram_vocab": 50, "num_attention_heads": 4
    },
    "inference": {
      "block_size": 4, "window": 32, "diffusion": 1,
      "d0_mode": "learned", "entropy_patches": null, "entropy_lm": null
    },
    "training": {"step": 120, "optimizer": "adamw", "seed": 7},
    "implementation_defaults": {"num_attention_heads": 4, "rope_theta": 500000.0,
      "rms_norm_eps": 1e-06, "norm": "rmsnorm", "activation": "swiglu",
      "ngram_sizes": [3, 4]},
    "source_checkpoint": {"path": "...", "size_bytes": 123456, "sha256": "..."}
  }
}
```

A BLT model has three stacks, so `num_hidden_layers` is the decoder depth only and the
per-stack counts live in `fblt.architecture`. `hidden_size` is the model width (`embed`) and
`intermediate_size` is the FFN width (`hidden`). `architectures` names the class a future
`transformers` port would register; fBLT ships no such class, which is why the card says the
directory is not loadable with `AutoModel`.

The checkpoint does not record the attention head count, RoPE theta, RMSNorm epsilon or the
ngram hash sizes; the fBLT C implementation hardcodes them. They are written under
`implementation_defaults` and labelled as such instead of being presented as recorded
metadata.

### What the metadata can and cannot reconstruct

The export records the architecture dimensions, `cross_attn` placement, `d0_mode`,
`block_size`, `window`, `diffusion` and the training step/optimizer/seed. That is enough to
instantiate the model and score it.

Entropy patching is a training choice and the entropy LM is a separate file (`.fblt`, written
by entropy-LM training); neither is stored in the checkpoint. `entropy_patches` and
`entropy_lm` are therefore `null`, not guessed. To reproduce a run that trained with entropy
patching, supply the entropy LM with `--entropy-lm`; without it, inference must use
fixed-stride patches (`--fixed-patches`) at the recorded block size.

### Validation

Export fails instead of writing a misleading directory when the source is missing, is a v1
container (no metadata), is truncated, or when the metadata's layer counts do not match the
tensor set and shapes. Import fails when a tensor is missing, extra, or the wrong shape, and
when `config.json` and `fblt_metadata.json` disagree. Export also refuses to overwrite an
existing export unless `--overwrite` is passed.

`python -m fblt.hf_export verify --checkpoint FILE` exports to a temporary directory,
imports it back, and compares tensor payloads, metadata bytes and file hashes. It prints JSON
and exits non-zero if anything differs.

## Limits

- A deleted checkpoint leaves a dangling pointer. `--resume` fails with a message naming
  the pointer; the Python `latest_checkpoint`/`best_checkpoint` helpers fall back to
  scanning `*.fblt`, which the trainer does not do.
- Pointers store the path as written. Use an absolute `--save-dir`, or resume from the same
  working directory.
- `--save-tag` does not exist; use separate `--save-dir` directories for separate runs.
- Weight and optimizer files are written by separate atomic renames, so a crash between
  them can leave a `.fblt` with no matching `.fbop` (resume warns and restarts the optimizer
  cold). There is no cross-file transaction.
