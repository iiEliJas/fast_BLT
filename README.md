# fBLT

[![CI](https://github.com/iiEliJas/fBLT/actions/workflows/ci.yml/badge.svg)](https://github.com/iiEliJas/fBLT/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Fast Byte Latent Transformer: a byte-level language model implemented in C and CUDA, with a Python package for configuration, training, and inference. It has no tokenizer and no vocabulary. Text is modeled directly as bytes.

The project implements ideas from two Meta FAIR papers:

- [Byte Latent Transformer](https://arxiv.org/abs/2412.09871) - byte modeling with entropy-based dynamic patching.
- [Fast Byte Latent Transformer](https://arxiv.org/abs/2605.08044) - faster inference with diffusion decoding and self-speculation.

## Status

Version 1.0 is a complete implementation of the training and inference pipeline. The largest demonstrated model is **7.8M parameters** (TinyStories, 600k steps). CPU and CUDA builds are supported. BF16 mixed-precision training is available on CUDA (`--dtype bf16`); multi-GPU training and post-training (SFT/RL) are not implemented in 1.0; see [todo.md](todo.md).

## Example output

A 7.8M-parameter BLT-D model trained on TinyStories for 600k steps, with entropy patching at block size 4 and a 512-byte window. Held-out causal bits per byte: **0.9978**. Greedy decoding, no sampling:

```text
Lily found a big box of cookies and started to play with it. She was happy to have
a new friend to play with. She was happy to have her friends and they went back to
playing together.
```

![Held-out causal BPB vs training steps](graphs/quality_vs_steps.png)

## Quickstart

The fastest path from a fresh checkout to generated text on CPU:

```bash
pip install -e .                    # Python package + fblt-train / fblt-infer
cmake -S . -B build                 # CPU build
cmake --build build -j$(nproc)      # build all targets

python fblt/scripts/build_sample_corpus.py  # writes data/train.bin and data/heldout.bin
fblt-train --config configs/train/debug.yaml --backend cpu

# Use the checkpoint path reported by the [CKPT] line, or let the shell expand the glob
fblt-infer --checkpoint runs/debug-*/checkpoint.fblt --backend cpu --prompt "int main"
```

The sample corpus is built from the repository's own C/H source files (~1.1 MB) under the Apache 2.0 license. The debug config trains a tiny model in seconds. For the complete workflow, CUDA setup, custom datasets, resumption, and troubleshooting, see the [Starter Guide](docs/STARTER_GUIDE.md).

## How it works

Five-stage pipeline:

1. **Entropy Model**: a small byte-level LM that predicts per-byte entropy.
2. **Patcher**: splits the byte stream into patches using entropy thresholds.
3. **Local Encoder**: a small transformer that compresses byte patches into embeddings, using hash n-gram embeddings to recognize byte patterns without a vocabulary.
4. **Patch Transformer**: the main model; block-causal attention over patches.
5. **Local Decoder**: a small transformer that expands patches back to bytes, with optional self-speculation.

Fast-BLT adds three inference modes on top of this pipeline:

- **BLT-S (self-speculation)**: draft bytes decoder-only, then verify them with one full forward. Output matches greedy decoding.
- **BLT-D (block diffusion)**: generate a block of future bytes in parallel from masked states. Cheapest per byte, but output can diverge from greedy.
- **BLT-DV (diffusion + verification)**: BLT-D drafts a block and then verifies it. Output matches greedy decoding when verification succeeds.

## Build

```bash
# CPU
cmake -S . -B build
cmake --build build -j$(nproc)
cmake --build build --target test   # C test suite

# CUDA
cmake -S . -B build-cuda -DUSE_CUDA=ON
cmake --build build-cuda -j$(nproc)
cmake --build build-cuda --target test
```

Python setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install ruff pytest
```

CUDA builds default to architecture `89`; use `-DCMAKE_CUDA_ARCHITECTURES=...` for another GPU. On WSL2, if an apt NVIDIA library shadows the WSL driver, put the WSL driver directory first on `LD_LIBRARY_PATH` and set `CUDA_VISIBLE_DEVICES=0` if it is empty. Full build, test, and CUDA instructions are in the [Starter Guide](docs/STARTER_GUIDE.md).

## Supported platforms

| OS | CPU | CUDA |
|----|-----|------|
| Linux | ✅ | ✅ |
| Windows (MSYS2/MinGW) | ✅ | compile-checked |
| macOS | ✅ | ❌ (no NVIDIA) |

## Documentation

- [Starter Guide](docs/STARTER_GUIDE.md) - installation, first run, training, inference, checkpoints, evaluation, and troubleshooting.
- [CLI Reference](docs/CLI_REFERENCE.md) - every training, inference, benchmark, and diagnostic flag.
- [Benchmarks](docs/BENCHMARKS.md) - inference and CUDA performance results.
- [Ablations](docs/ABLATIONS.md) - architecture and schedule ablation results.
- [Research Guide](docs/RESEARCH_GUIDE.md) - how to formulate, run, and report experiments.
- [Experiment Log](docs/EXPERIMENT_LOG.md) - index of significant research activity.
- [Experiments](docs/experiments/) - investigation notes and report template.
- [Profiling](docs/PROFILING.md) - CPU and CUDA profiling recipes.
- [todo.md](todo.md) - public roadmap.
- [CONTRIBUTING.md](CONTRIBUTING.md) - branch policy, style, and PR guidelines.

## Research results

At the 7.8M-parameter scale, BLT-DV beats greedy decoding only at the trained block size (`B=4`): 65% draft acceptance, output byte-identical to greedy, 0.85x memory bandwidth and 0.89x wall-clock. At `B=8` and above it loses badly, because block cells past the trained size were never supervised. Entropy patching more than doubles acceptance over fixed-stride patching. See [BENCHMARKS.md](docs/BENCHMARKS.md) and [ABLATIONS.md](docs/ABLATIONS.md).

## Known limitations

- Patch count is capped at 128 per window (`BLT_MAX_PATCHES`).
- Demonstrated scale is 7.8M parameters; larger configs are untested.
- CUDA runs are non-deterministic unless `--deterministic` is passed.
- Sampling (`--temperature`, `--top-p`, `--top-k`, etc.) works only with `--method greedy`.
- Single-GPU / single-process only.
- BF16 mixed precision is CUDA-only. The trainer keeps FP32 master weights, optimizer state, normalization, softmax, cross-entropy, and checkpoints; only forward matmul operands and the backward input-gradient GEMM read the BF16 shadows. The local encoder/decoder and the global transformer all use BF16 shadow weights, refreshed after every optimizer step. On CPU the trainer rejects `--dtype bf16` because the CPU matmul path only emulates the cast with FP32 temporaries. Raw BF16 GEMMs are 2.2-3.2x faster than FP32 on an RTX 4060, but end-to-end step time improves only a few percent at the model sizes in this repository; see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Community experiments

fBLT is small and there are more ideas than time to test them. If you change something and see what happens, open an issue with what you changed, your hardware, and the command. Attach the `resolved_config.yaml` from your run directory. Negative results are welcome. See [todo.md](todo.md) for scoped work items.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branch policy, code style, and PR guidelines. Run `ruff check .`, `ruff format --check .`, `pytest tests/`, and `cmake --build build --target test` before submitting.

## References

If you build on this work, please consider citing the foundational papers:

```bibtex
@misc{pagnoni2024bytelatenttransformerpatches,
      title={Byte Latent Transformer: Patches Scale Better Than Tokens},
      author={Artidoro Pagnoni and Ramakanth Pasunuru and Pedro Rodriguez and John Nguyen and
              Benjamin Muller and Margaret Li and Chunting Zhou and Lili Yu and Jason Weston and
              Luke Zettlemoyer and Gargi Ghosh and Mike Lewis and Ari Holtzman and Srinivasan Iyer},
      year={2024},
      eprint={2412.09871},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      doi={10.48550/arXiv.2412.09871}
}

@misc{kallini2026fastbytelatenttransformer,
      title={Fast Byte Latent Transformer},
      author={Julie Kallini and Artidoro Pagnoni and Tomasz Limisiewicz and Gargi Ghosh and
              Luke Zettlemoyer and Christopher Potts and Xiaochuang Han and Srinivasan Iyer},
      year={2026},
      eprint={2605.08044},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

[Apache 2.0](LICENSE)
