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
