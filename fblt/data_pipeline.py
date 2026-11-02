"""Document-aware data pipeline (item 7)."""

import json
import mmap


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def windows_for_docs(docs, window, mode="mask"):
    # docs: list of byte strings
    out = []
    for d in docs:
        b = d
        i = 0
        n = len(b)
        while i < n:
            chunk = b[i : i + window]
            if len(chunk) < window and mode in ("mask", "pad"):
                # pad
                pad = window - len(chunk)
                chunk = chunk + b"\x00" * pad
            out.append((chunk, i + window >= n))
            i += window
    return out


def pack_documents(docs, window):
    # best-fit packing: concatenate docs with boundary markers, split into windows
    packed = b"".join(d + b"\n" for d in docs)  # sentinel newline
    windows = []
    i = 0
    L = len(packed)
    while i < L:
        windows.append(packed[i : i + window])
        i += window
    return windows


def stream_corpus(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        yield mm
        mm.close()


def sampler_state(epoch, doc_idx, window_idx, rng_seed):
    return {"epoch": epoch, "doc_idx": doc_idx, "window_idx": window_idx, "rng_seed": rng_seed}


def save_sampler_state(path, state):
    import json

    with open(path, "w") as f:
        json.dump(state, f)


def load_sampler_state(path):
    import json

    with open(path) as f:
        return json.load(f)


def resume_index(state, total_windows):
    # deterministic next window
    return (state["window_idx"] + 1) % total_windows
