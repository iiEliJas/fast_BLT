# Evaluation

## Tasks and metrics

Document each task, prompt format, split, and metric here.

## Byte-level vs token-level

Token-level benchmark numbers are not directly comparable to byte-level scores without the same protocol.

## Usage

```bash
python -m fblt.eval --task bpb --checkpoint model.fblt --corpus data/heldout.bin --out eval.json
```

Few-shot/zero-shot selectable via `--few-shot N`. Stores entropy LM, threshold, block size, seed in record.
EOF