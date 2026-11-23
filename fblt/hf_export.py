"""Convert fBLT checkpoints to and from a Hugging Face directory.

The export is a real tensor-level conversion: weights land in ``model.safetensors``
under HF-style names, the container metadata is preserved byte-for-byte, and the
inverse reconstructs an fBLT v2 container without reading whole tensors into memory.
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
import tempfile
import typing

from fblt import _safetensors
from fblt._checkpoint import TensorLayout, read_meta_raw, tensor_layout
from fblt._provenance import sha256_file
from fblt._safetensors import SafetensorsError

HF_FORMAT = "fblt-hf/v1"
TENSOR_SCHEMA = "fblt-hf-tensors/v1"
SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
METADATA_NAME = "fblt_metadata.json"
CARD_NAME = "README.md"
OUTPUT_NAMES = (SAFETENSORS_NAME, CONFIG_NAME, METADATA_NAME, CARD_NAME)

_NUM_HEADS = 4
_ROPE_THETA = 500000.0
_RMS_NORM_EPS = 1e-06
_DIM_FMT = "<Q" if struct.calcsize("P") == 8 else "<I"
_CHUNK = 1 << 20

_SELF_LEAVES = {
    "norm1": "input_layernorm.weight",
    "attn_qkv": "self_attn.qkv_proj.weight",
    "attn_proj": "self_attn.o_proj.weight",
    "norm2": "post_attention_layernorm.weight",
    "ffn_up": "mlp.up_proj.weight",
    "ffn_gate": "mlp.gate_proj.weight",
    "ffn_down": "mlp.down_proj.weight",
}
_CROSS_LEAVES = {
    "cross_norm": "cross_attn.layer_norm.weight",
    "cross_q": "cross_attn.q_proj.weight",
    "cross_k": "cross_attn.k_proj.weight",
    "cross_v": "cross_attn.v_proj.weight",
    "cross_proj": "cross_attn.o_proj.weight",
}
_LEAF_TO_FBLT = {value: key for key, value in {**_SELF_LEAVES, **_CROSS_LEAVES}.items()}
_STACK_TO_HF = {"enc": "blt.encoder", "glob": "blt.global_transformer", "dec": "blt.decoder"}
_STACK_FROM_HF = {"encoder": "enc", "global_transformer": "glob", "decoder": "dec"}
_NGRAM_FBLT_RE = re.compile(r"^enc\.ngram(\d+)$")
_NGRAM_HF_RE = re.compile(r"^blt\.encoder\.ngram_embeddings\.(\d+)\.weight$")
_FBLT_STACK_RE = re.compile(r"^(enc|glob|dec)\.L(\d+)\.([a-z0-9_]+)$")
_HF_STACK_RE = re.compile(r"^blt\.(encoder|global_transformer|decoder)\.layers\.(\d+)\.(.+)$")


def fblt_tensor_to_hf(name: str) -> str:
    """Map a canonical fBLT tensor name to its Hugging Face name."""
    if name == "enc.embed":
        return "blt.encoder.byte_embedding.weight"
    if name == "dec.lm_head":
        return "blt.lm_head.weight"
    if name == "dec.d0_embed":
        return "blt.decoder.d0_embedding.weight"
    match = _NGRAM_FBLT_RE.match(name)
    if match:
        return f"blt.encoder.ngram_embeddings.{match.group(1)}.weight"
    match = _FBLT_STACK_RE.match(name)
    if match:
        stack, layer, leaf = match.groups()
        leaf_hf = _SELF_LEAVES.get(leaf) or _CROSS_LEAVES.get(leaf)
        if leaf_hf is None:
            raise ValueError(f"no Hugging Face mapping for tensor '{name}'")
        return f"{_STACK_TO_HF[stack]}.layers.{layer}.{leaf_hf}"
    raise ValueError(f"no Hugging Face mapping for tensor '{name}'")


def hf_tensor_to_fblt(name: str) -> str:
    """Map a Hugging Face tensor name back to the canonical fBLT name."""
    if name == "blt.encoder.byte_embedding.weight":
        return "enc.embed"
    if name == "blt.lm_head.weight":
        return "dec.lm_head"
    if name == "blt.decoder.d0_embedding.weight":
        return "dec.d0_embed"
    match = _NGRAM_HF_RE.match(name)
    if match:
        return f"enc.ngram{match.group(1)}"
    match = _HF_STACK_RE.match(name)
    if match:
        stack_hf, layer, leaf_hf = match.groups()
        leaf = _LEAF_TO_FBLT.get(leaf_hf)
        if leaf is None:
            raise ValueError(f"no fBLT mapping for tensor '{name}'")
        return f"{_STACK_FROM_HF[stack_hf]}.L{layer}.{leaf}"
    raise ValueError(f"no fBLT mapping for tensor '{name}'")


def _numel(shape) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _require_int(mapping, key, what, *, minimum=0):
    value = mapping.get(key) if isinstance(mapping, dict) else None
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{what} must be an integer >= {minimum}")
    return value


def _metadata_config(metadata) -> typing.Dict[str, typing.Any]:
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("checkpoint metadata is empty or not a JSON object")
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint metadata has no 'config' object")
    for key in ("embed", "hidden", "enc_layers", "glob_layers", "dec_layers"):
        if key not in config:
            raise ValueError(f"checkpoint metadata config is missing '{key}'")
    _require_int(config, "embed", "metadata config.embed", minimum=1)
    _require_int(config, "hidden", "metadata config.hidden", minimum=1)
    _require_int(config, "enc_layers", "metadata config.enc_layers")
    _require_int(config, "glob_layers", "metadata config.glob_layers")
    _require_int(config, "dec_layers", "metadata config.dec_layers")
    return config


def _canonical_tensors(
    *,
    embed: int,
    hidden: int,
    enc_layers: int,
    glob_layers: int,
    dec_layers: int,
    ngram_tables: int,
    per_ngram_vocab: int,
    vocab_size: int,
    d0_vocab: int,
) -> typing.List[typing.Tuple[str, typing.Tuple[int, ...]]]:
    entries = [("enc.embed", (vocab_size, embed))]
    for index in range(ngram_tables):
        entries.append((f"enc.ngram{index}", (per_ngram_vocab, embed)))
    stacks = (
        ("enc", enc_layers, True),
        ("glob", glob_layers, False),
        ("dec", dec_layers, True),
    )
    for stack, layers, has_cross in stacks:
        for layer in range(layers):
            base = f"{stack}.L{layer}"
            entries.extend(
                [
                    (f"{base}.norm1", (embed,)),
                    (f"{base}.attn_qkv", (embed, 3 * embed)),
                    (f"{base}.attn_proj", (embed, embed)),
                    (f"{base}.norm2", (embed,)),
                    (f"{base}.ffn_up", (embed, hidden)),
                    (f"{base}.ffn_gate", (embed, hidden)),
                    (f"{base}.ffn_down", (hidden, embed)),
                ]
            )
            if has_cross:
                entries.extend(
                    [
                        (f"{base}.cross_norm", (embed,)),
                        (f"{base}.cross_q", (embed, embed)),
                        (f"{base}.cross_k", (embed, embed)),
                        (f"{base}.cross_v", (embed, embed)),
                        (f"{base}.cross_proj", (embed, embed)),
                    ]
                )
    entries.append(("dec.lm_head", (embed, vocab_size)))
    entries.append(("dec.d0_embed", (d0_vocab, embed)))
    return entries


def _by_name(layouts) -> typing.Dict[str, TensorLayout]:
    mapping = {}
    for layout in layouts:
        if layout.name in mapping:
            raise ValueError(f"checkpoint has duplicate tensor '{layout.name}'")
        mapping[layout.name] = layout
    return mapping


def _ngram_indices(names) -> typing.List[int]:
    indices = []
    for name in names:
        match = _NGRAM_FBLT_RE.match(name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def _ngram_count(names) -> int:
    indices = _ngram_indices(names)
    if not indices:
        raise ValueError("checkpoint has no enc.ngram<k> tensors")
    if indices != list(range(len(indices))):
        raise ValueError(f"ngram tensors are not contiguous from 0: {indices}")
    return len(indices)


def _derive_architecture(metadata) -> typing.Dict[str, typing.Any]:
    config = _metadata_config(metadata)
    embed = config["embed"]
    hidden = config["hidden"]
    return {
        "embed": embed,
        "hidden": hidden,
        "enc_layers": config["enc_layers"],
        "glob_layers": config["glob_layers"],
        "dec_layers": config["dec_layers"],
        "cross_attn": config.get("cross_attn"),
        "d0_mode": config.get("d0_mode"),
    }


def _validate_tensor_set(by_name, canonical) -> None:
    expected = {name: tuple(shape) for name, shape in canonical}
    actual = set(by_name)
    unknown = sorted(actual - set(expected))
    if unknown:
        raise ValueError(f"unexpected tensor(s): {', '.join(unknown)}")
    missing = [name for name, _ in canonical if name not in actual]
    if missing:
        raise ValueError(f"missing tensor(s): {', '.join(missing)}")
    for name, shape in canonical:
        got = tuple(by_name[name].shape)
        if got != tuple(shape):
            raise ValueError(f"tensor '{name}' shape {got} != expected {tuple(shape)}")


def _scan_source(fblt_path) -> typing.Tuple[dict, typing.Dict[str, TensorLayout], list]:
    if not os.path.isfile(fblt_path):
        raise ValueError(f"checkpoint not found: {fblt_path}")
    version, raw, layouts = tensor_layout(fblt_path)
    if version == 1:
        raise ValueError(
            "checkpoint has no metadata (v1); this exporter needs a v2 checkpoint "
            "with metadata; retrain or re-save"
        )
    if version != 2:
        raise ValueError(f"unsupported checkpoint version {version}")
    if raw is None:
        raise ValueError("checkpoint has no metadata block")
    try:
        metadata = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"checkpoint metadata is not valid JSON: {exc}") from exc
    _metadata_config(metadata)
    return metadata, _by_name(layouts), layouts


def _architecture_from_tensors(metadata, by_name) -> typing.Dict[str, typing.Any]:
    arch = _derive_architecture(metadata)
    embed = arch["embed"]
    for required in ("enc.embed", "dec.lm_head", "dec.d0_embed"):
        if required not in by_name:
            raise ValueError(f"checkpoint is missing tensor '{required}'")
    embed_shape = tuple(by_name["enc.embed"].shape)
    lm_shape = tuple(by_name["dec.lm_head"].shape)
    d0_shape = tuple(by_name["dec.d0_embed"].shape)
    if len(embed_shape) != 2 or len(lm_shape) != 2 or len(d0_shape) != 2:
        raise ValueError("enc.embed, dec.lm_head and dec.d0_embed must be 2-D")
    vocab_size = embed_shape[0]
    if lm_shape[1] != vocab_size:
        raise ValueError(
            f"vocab_size disagrees: enc.embed shape[0]={vocab_size}, "
            f"dec.lm_head shape[1]={lm_shape[1]}"
        )
    if embed_shape[1] != embed:
        raise ValueError(f"enc.embed shape {embed_shape} does not match metadata embed={embed}")
    if lm_shape[0] != embed:
        raise ValueError(f"dec.lm_head shape {lm_shape} does not match metadata embed={embed}")
    if d0_shape[1] != embed:
        raise ValueError(f"dec.d0_embed shape {d0_shape} does not match metadata embed={embed}")
    per_ngram_vocab, ngram_tables = _ngram_vocab(by_name, embed)
    arch.update(
        {
            "vocab_size": vocab_size,
            "d0_vocab": d0_shape[0],
            "ngram_tables": ngram_tables,
            "per_ngram_vocab": per_ngram_vocab,
            "num_attention_heads": _NUM_HEADS,
        }
    )
    return arch


def _ngram_vocab(by_name, embed) -> typing.Tuple[int, int]:
    indices = _ngram_indices(by_name)
    if not indices:
        raise ValueError("checkpoint has no enc.ngram<k> tensors")
    if indices != list(range(len(indices))):
        raise ValueError(f"ngram tensors are not contiguous from 0: {indices}")
    vocab = None
    for index in indices:
        shape = tuple(by_name[f"enc.ngram{index}"].shape)
        if len(shape) != 2 or shape[1] != embed:
            raise ValueError(
                f"tensor 'enc.ngram{index}' shape {shape} does not match embed={embed}"
            )
        if vocab is None:
            vocab = shape[0]
        elif shape[0] != vocab:
            raise ValueError("ngram tensors disagree on the per-ngram vocabulary size")
    return vocab, len(indices)


def build_config(
    metadata: dict,
    architecture: dict,
    *,
    format_version: int = 2,
    source_path: str = "",
    source_size_bytes: int = 0,
    source_sha256: str = "",
) -> dict:
    """Build the deterministic config.json object from parsed metadata and shapes."""
    config = metadata.get("config") if isinstance(metadata, dict) else {}
    config = config if isinstance(config, dict) else {}
    window = config.get("window")
    inference = {
        "block_size": config.get("block_size"),
        "window": window,
        "diffusion": config.get("diffusion"),
        "d0_mode": config.get("d0_mode"),
        "entropy_patches": None,
        "entropy_lm": None,
    }
    ngram_tables = architecture.get("ngram_tables")
    out = {
        "architectures": ["BLTForCausalLM"],
        "model_type": "fblt",
        "format": HF_FORMAT,
        "vocab_size": architecture["vocab_size"],
        "hidden_size": architecture["embed"],
        "intermediate_size": architecture["hidden"],
        "num_hidden_layers": architecture["dec_layers"],
        "num_attention_heads": architecture["num_attention_heads"],
        "num_key_value_heads": architecture["num_attention_heads"],
    }
    if isinstance(window, int) and not isinstance(window, bool):
        out["max_position_embeddings"] = window
    out.update(
        {
            "hidden_act": "silu",
            "rms_norm_eps": _RMS_NORM_EPS,
            "rope_theta": _ROPE_THETA,
            "torch_dtype": "float32",
            "tie_word_embeddings": False,
            "fblt": {
                "format_version": format_version,
                "tensor_schema": TENSOR_SCHEMA,
                "checkpoint_metadata": metadata,
                "architecture": {
                    "embed": architecture["embed"],
                    "hidden": architecture["hidden"],
                    "enc_layers": architecture["enc_layers"],
                    "glob_layers": architecture["glob_layers"],
                    "dec_layers": architecture["dec_layers"],
                    "cross_attn": architecture["cross_attn"],
                    "d0_mode": architecture["d0_mode"],
                    "vocab_size": architecture["vocab_size"],
                    "d0_vocab": architecture["d0_vocab"],
                    "ngram_tables": architecture["ngram_tables"],
                    "per_ngram_vocab": architecture["per_ngram_vocab"],
                    "num_attention_heads": architecture["num_attention_heads"],
                },
                "inference": inference,
                "training": metadata.get("training")
                if isinstance(metadata.get("training"), dict)
                else None,
                "implementation_defaults": {
                    "num_attention_heads": _NUM_HEADS,
                    "rope_theta": _ROPE_THETA,
                    "rms_norm_eps": _RMS_NORM_EPS,
                    "norm": "rmsnorm",
                    "activation": "swiglu",
                    "ngram_sizes": [3, 4] if ngram_tables == 2 else None,
                    "note": (
                        "compiled into the fBLT C implementation; not recorded in the "
                        "checkpoint metadata"
                    ),
                },
                "source_checkpoint": {
                    "path": source_path,
                    "size_bytes": source_size_bytes,
                    "sha256": source_sha256,
                },
            },
        }
    )
    return out


def _model_card(config: dict, source_path: str) -> str:
    arch = config["fblt"]["architecture"]
    inference = config["fblt"]["inference"]
    training = config["fblt"]["training"] or {}

    def show(value):
        return "n/a" if value is None else str(value)

    lines = [
        "---",
        "library_name: fblt",
        "tags:",
        "  - fblt",
        "  - byte-latent-transformer",
        "  - byte-level",
        "---",
        "",
        f"# fBLT checkpoint: {os.path.basename(source_path)}",
        "",
        "This is a byte-level fBLT/BLT checkpoint converted to the `fblt-hf/v1` layout. "
        "The weights live in the `blt.*` namespace and describe a custom architecture, "
        "so they are not loadable with `AutoModel` without a custom implementation.",
        "",
        "## Architecture",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Encoder layers | {arch['enc_layers']} |",
        f"| Global transformer layers | {arch['glob_layers']} |",
        f"| Decoder layers | {arch['dec_layers']} |",
        f"| Embed dim (`hidden_size`) | {arch['embed']} |",
        f"| FFN dim (`intermediate_size`) | {arch['hidden']} |",
        f"| Attention heads | {arch['num_attention_heads']} |",
        f"| Vocabulary | {arch['vocab_size']} byte symbols |",
        f"| Diffusion vocabulary (d0) | {arch['d0_vocab']} |",
        f"| N-gram tables | {arch['ngram_tables']} (vocab {arch['per_ngram_vocab']} each) |",
        f"| Block size | {show(inference['block_size'])} |",
        f"| Window | {show(inference['window'])} |",
        f"| Diffusion | {show(inference['diffusion'])} |",
        f"| Training step | {show(training.get('step'))} |",
        f"| Optimizer | {show(training.get('optimizer'))} |",
        f"| Seed | {show(training.get('seed'))} |",
        "",
        "`num_hidden_layers` in `config.json` is the decoder stack, not the sum of all "
        "three stacks.",
        "",
        "## Files",
        "",
        "| File | Contents |",
        "|---|---|",
        f"| `{SAFETENSORS_NAME}` | F32 weights in the `blt.*` namespace |",
        f"| `{CONFIG_NAME}` | HF-compatible config plus the full fBLT metadata |",
        f"| `{METADATA_NAME}` | Exact `.fblt` v2 metadata bytes for a byte-identical import |",
        f"| `{CARD_NAME}` | This card |",
        "",
        "## Tensor names",
        "",
        "See `docs/CHECKPOINTS.md` for the canonical fBLT tensor order and shapes. "
        "The fused `self_attn.qkv_proj.weight` stores columns as `[q | k | v]`, with "
        "heads interleaved inside each of the three blocks.",
        "",
        "## Limits",
        "",
        "- The entropy-LM weights are not part of the checkpoint and are not stored here.",
        "- The vocabulary is byte-level (256 symbols), not a tokenizer vocabulary.",
        "- `.fblt` dimensions are platform-native `sizeof(size_t)`; the safetensors "
        "shapes are portable.",
        "- Checkpoints are large; do not commit them to version control.",
    ]
    return "\n".join(lines) + "\n"


def _write_bytes_atomic(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_json_atomic(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _write_safetensors(path, items, source_path) -> None:
    header = _safetensors.encode_header([(name, layout.shape) for name, layout in items])
    tmp = path + ".tmp"
    try:
        with open(source_path, "rb") as src, open(tmp, "wb") as dst:
            dst.write(header)
            for _, layout in items:
                entry = {"offset": layout.offset, "nbytes": layout.nbytes}
                _safetensors.copy_tensor(src, entry, dst)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def export_to_hf(fblt_path, out_dir, *, overwrite: bool = False) -> dict:
    """Export a v2 fBLT checkpoint into a Hugging Face directory."""
    fblt_path = os.fspath(fblt_path)
    out_dir = os.fspath(out_dir)
    metadata, by_name, layouts = _scan_source(fblt_path)
    architecture = _architecture_from_tensors(metadata, by_name)
    canonical = _canonical_tensors(
        embed=architecture["embed"],
        hidden=architecture["hidden"],
        enc_layers=architecture["enc_layers"],
        glob_layers=architecture["glob_layers"],
        dec_layers=architecture["dec_layers"],
        ngram_tables=architecture["ngram_tables"],
        per_ngram_vocab=architecture["per_ngram_vocab"],
        vocab_size=architecture["vocab_size"],
        d0_vocab=architecture["d0_vocab"],
    )
    _validate_tensor_set(by_name, canonical)

    if not overwrite:
        existing = [name for name in OUTPUT_NAMES if os.path.exists(os.path.join(out_dir, name))]
        if existing:
            raise ValueError(
                f"output directory '{out_dir}' already contains "
                f"{', '.join(existing)}; pass overwrite=True to replace"
            )
    os.makedirs(out_dir, exist_ok=True)

    items = sorted(
        ((fblt_tensor_to_hf(name), by_name[name]) for name, _ in canonical),
        key=lambda item: item[0],
    )
    config = build_config(
        metadata,
        architecture,
        format_version=2,
        source_path=fblt_path,
        source_size_bytes=os.path.getsize(fblt_path),
        source_sha256=sha256_file(fblt_path),
    )

    _write_safetensors(os.path.join(out_dir, SAFETENSORS_NAME), items, fblt_path)
    _write_json_atomic(os.path.join(out_dir, CONFIG_NAME), config)
    raw_meta = read_meta_raw(fblt_path)
    _write_bytes_atomic(os.path.join(out_dir, METADATA_NAME), raw_meta or b"")
    card = _model_card(config, fblt_path)
    with open(os.path.join(out_dir, CARD_NAME) + ".tmp", "w", encoding="utf-8") as fh:
        fh.write(card)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(os.path.join(out_dir, CARD_NAME) + ".tmp", os.path.join(out_dir, CARD_NAME))
    return config


def _read_config(hf_dir: str) -> dict:
    path = os.path.join(hf_dir, CONFIG_NAME)
    with open(path, "r", encoding="utf-8") as fh:
        config = json.load(fh)
    if not isinstance(config, dict):
        raise ValueError(f"{path} is not a JSON object")
    if config.get("format") != HF_FORMAT:
        raise ValueError(f"{path}: format {config.get('format')!r} != {HF_FORMAT!r}")
    if config.get("model_type") != "fblt":
        raise ValueError(f"{path}: model_type {config.get('model_type')!r} != 'fblt'")
    fblt_block = config.get("fblt")
    if not isinstance(fblt_block, dict):
        raise ValueError(f"{path}: missing 'fblt' object")
    if fblt_block.get("tensor_schema") != TENSOR_SCHEMA:
        raise ValueError(
            f"{path}: tensor_schema {fblt_block.get('tensor_schema')!r} != {TENSOR_SCHEMA!r}"
        )
    metadata = fblt_block.get("checkpoint_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: fblt.checkpoint_metadata is not an object")
    _metadata_config(metadata)
    return config


def _metadata_bytes(hf_dir: str, metadata: dict) -> typing.Tuple[bytes, bool]:
    path = os.path.join(hf_dir, METADATA_NAME)
    if not os.path.isfile(path):
        return json.dumps(metadata, separators=(",", ":")).encode("utf-8"), True
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if parsed != metadata:
        raise ValueError(f"{path} does not match config.json fblt.checkpoint_metadata")
    return raw, False


def _cross_check_architecture(config: dict, values: dict) -> None:
    fblt_block = config["fblt"]
    declared = fblt_block.get("architecture")
    if declared is None:
        return
    if not isinstance(declared, dict):
        raise ValueError("config.json fblt.architecture is not an object")
    names = {
        "embed": "enc.embed",
        "vocab_size": "enc.embed/dec.lm_head",
        "d0_vocab": "dec.d0_embed",
        "per_ngram_vocab": "enc.ngram<k>",
        "ngram_tables": "enc.ngram<k>",
        "enc_layers": "metadata config.enc_layers",
        "glob_layers": "metadata config.glob_layers",
        "dec_layers": "metadata config.dec_layers",
    }
    for key, source in names.items():
        if key not in declared:
            continue
        if declared[key] != values[key]:
            raise ValueError(
                f"fblt.architecture.{key} {declared[key]!r} != {values[key]!r} from {source}"
            )


def _read_safetensors(hf_dir: str) -> typing.Dict[str, dict]:
    path = os.path.join(hf_dir, SAFETENSORS_NAME)
    _, tensors, _ = _safetensors.parse_header(path)
    by_name = {}
    for hf_name, entry in tensors.items():
        fblt_name = hf_tensor_to_fblt(hf_name)
        if fblt_name in by_name:
            raise ValueError(f"two HF tensors map to '{fblt_name}'")
        by_name[fblt_name] = entry
    return by_name


def import_from_hf(hf_dir, out_path) -> dict:
    """Rebuild an fBLT v2 checkpoint from a Hugging Face directory."""
    hf_dir = os.fspath(hf_dir)
    out_path = os.fspath(out_path)
    config = _read_config(hf_dir)
    metadata = config["fblt"]["checkpoint_metadata"]
    meta_bytes, reserialized = _metadata_bytes(hf_dir, metadata)

    by_name = _read_safetensors(hf_dir)
    for required in ("enc.embed", "dec.lm_head", "dec.d0_embed"):
        if required not in by_name:
            raise ValueError(f"missing tensor '{fblt_tensor_to_hf(required)}'")
    ngram_tables = _ngram_count(by_name)
    embed_shape = tuple(by_name["enc.embed"]["shape"])
    lm_shape = tuple(by_name["dec.lm_head"]["shape"])
    d0_shape = tuple(by_name["dec.d0_embed"]["shape"])
    if len(embed_shape) != 2 or len(lm_shape) != 2 or len(d0_shape) != 2:
        raise ValueError("enc.embed, dec.lm_head and dec.d0_embed must be 2-D")
    vocab_size = embed_shape[0]
    if lm_shape[1] != vocab_size:
        raise ValueError(
            f"vocab_size disagrees: enc.embed shape[0]={vocab_size}, "
            f"dec.lm_head shape[1]={lm_shape[1]}"
        )
    per_ngram_vocab = None
    for index in range(ngram_tables):
        shape = tuple(by_name[f"enc.ngram{index}"]["shape"])
        if per_ngram_vocab is None:
            per_ngram_vocab = shape[0] if len(shape) == 2 else None
        if len(shape) != 2 or shape[0] != per_ngram_vocab:
            raise ValueError("ngram tensors disagree on the per-ngram vocabulary size")
    if not isinstance(per_ngram_vocab, int):
        raise ValueError("ngram tensors have an invalid shape")

    cfg = metadata["config"]
    architecture = {
        "embed": cfg["embed"],
        "hidden": cfg["hidden"],
        "enc_layers": cfg["enc_layers"],
        "glob_layers": cfg["glob_layers"],
        "dec_layers": cfg["dec_layers"],
        "vocab_size": vocab_size,
        "d0_vocab": d0_shape[0],
        "ngram_tables": ngram_tables,
        "per_ngram_vocab": per_ngram_vocab,
    }
    _cross_check_architecture(config, architecture)

    canonical = _canonical_tensors(
        embed=cfg["embed"],
        hidden=cfg["hidden"],
        enc_layers=cfg["enc_layers"],
        glob_layers=cfg["glob_layers"],
        dec_layers=cfg["dec_layers"],
        ngram_tables=ngram_tables,
        per_ngram_vocab=per_ngram_vocab,
        vocab_size=vocab_size,
        d0_vocab=d0_shape[0],
    )
    _validate_safetensors_set(by_name, canonical)

    tmp = out_path + ".tmp"
    try:
        with open(tmp, "wb") as dst, open(os.path.join(hf_dir, SAFETENSORS_NAME), "rb") as src:
            dst.write(b"FBLT")
            dst.write(struct.pack("<I", 2))
            dst.write(struct.pack("<I", len(meta_bytes)))
            dst.write(meta_bytes)
            dst.write(struct.pack("<I", len(canonical)))
            for name, shape in canonical:
                entry = by_name[name]
                raw_name = name.encode("utf-8")
                dst.write(struct.pack("<H", len(raw_name)))
                dst.write(raw_name)
                dst.write(struct.pack("<I", len(shape)))
                for dim in shape:
                    dst.write(struct.pack(_DIM_FMT, dim))
                _safetensors.copy_tensor(src, entry, dst)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if reserialized:
        config["_fblt_metadata_reserialized"] = True
    return config


def _validate_safetensors_set(by_name, canonical) -> None:
    expected = {name: tuple(shape) for name, shape in canonical}
    actual = set(by_name)
    unknown = sorted(actual - set(expected))
    if unknown:
        names = ", ".join(fblt_tensor_to_hf(name) for name in unknown)
        raise ValueError(f"unexpected tensor(s): {names}")
    missing = [name for name, _ in canonical if name not in actual]
    if missing:
        names = ", ".join(fblt_tensor_to_hf(name) for name in missing)
        raise ValueError(f"missing tensor(s): {names}")
    for name, shape in canonical:
        got = tuple(by_name[name]["shape"])
        if got != tuple(shape):
            raise ValueError(
                f"tensor '{fblt_tensor_to_hf(name)}' shape {got} != expected {tuple(shape)}"
            )


def _payloads_equal(path_a: str, path_b: str, layout_a, layout_b) -> typing.List[str]:
    problems = []
    with open(path_a, "rb") as fh_a, open(path_b, "rb") as fh_b:
        fh_a.seek(layout_a.offset)
        fh_b.seek(layout_b.offset)
        remaining = layout_a.nbytes
        while remaining:
            size = min(remaining, _CHUNK)
            chunk_a = fh_a.read(size)
            chunk_b = fh_b.read(size)
            if chunk_a != chunk_b or len(chunk_a) != size:
                problems.append(f"tensor '{layout_a.name}' payload differs")
                break
            remaining -= size
    return problems


def _compare_checkpoints(source_path: str, restored_path: str) -> dict:
    problems = []
    _, source_meta, source_layouts = tensor_layout(source_path)
    _, restored_meta, restored_layouts = tensor_layout(restored_path)
    metadata_identical = source_meta == restored_meta
    if not metadata_identical:
        problems.append("metadata bytes differ")
    if len(source_layouts) != len(restored_layouts):
        problems.append(f"tensor count {len(restored_layouts)} != {len(source_layouts)}")
    else:
        for source_layout, restored_layout in zip(source_layouts, restored_layouts):
            if source_layout.name != restored_layout.name:
                problems.append(
                    f"tensor order differs: '{restored_layout.name}' != '{source_layout.name}'"
                )
                break
            if tuple(source_layout.shape) != tuple(restored_layout.shape):
                problems.append(f"tensor '{source_layout.name}' shape differs")
                continue
            if source_layout.nbytes != restored_layout.nbytes:
                problems.append(f"tensor '{source_layout.name}' size differs")
                continue
            problems.extend(
                _payloads_equal(source_path, restored_path, source_layout, restored_layout)
            )
    source_sha = sha256_file(source_path)
    restored_sha = sha256_file(restored_path)
    return {
        "identical": not problems,
        "tensor_count": len(source_layouts),
        "metadata_identical": metadata_identical,
        "source_sha256": source_sha,
        "restored_sha256": restored_sha,
        "problems": problems,
    }


def verify_roundtrip(fblt_path, out_dir=None, reload_path=None, *, overwrite: bool = False) -> dict:
    """Export, import and compare payloads/metadata; temporary paths are removed."""
    created_dir = None
    created_file = None
    if out_dir is None:
        created_dir = tempfile.mkdtemp(prefix="fblt-hf-")
        out_dir = created_dir
    if reload_path is None:
        fd, created_file = tempfile.mkstemp(prefix="fblt-reload-", suffix=".fblt")
        os.close(fd)
        reload_path = created_file
    try:
        export_to_hf(fblt_path, out_dir, overwrite=overwrite)
        import_from_hf(out_dir, reload_path)
        return _compare_checkpoints(os.fspath(fblt_path), os.fspath(reload_path))
    finally:
        if created_file is not None:
            try:
                os.unlink(created_file)
            except OSError:
                pass
        if created_dir is not None:
            shutil.rmtree(created_dir, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fblt-hf-export",
        description="Export fBLT checkpoints to a Hugging Face directory and back.",
    )
    sub = parser.add_subparsers(dest="command")
    export = sub.add_parser("export", help="export a .fblt checkpoint")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--overwrite", action="store_true")
    imp = sub.add_parser("import", help="import a Hugging Face directory")
    imp.add_argument("--hf-dir", required=True)
    imp.add_argument("--out", required=True)
    verify = sub.add_parser("verify", help="export, import and compare")
    verify.add_argument("--checkpoint", required=True)
    verify.add_argument("--out")
    verify.add_argument("--reload")
    return parser


def main(argv=None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not args_list:
        parser.print_help()
        return 2
    args = parser.parse_args(args_list)
    try:
        if args.command == "export":
            config = export_to_hf(args.checkpoint, args.out, overwrite=args.overwrite)
            del config
            count = len(tensor_layout(args.checkpoint)[2])
            print(f"hf-export: {count} tensors -> {os.path.join(args.out, CONFIG_NAME)}")
            return 0
        if args.command == "import":
            import_from_hf(args.hf_dir, args.out)
            count = len(tensor_layout(args.out)[2])
            print(f"hf-export: {count} tensors -> {args.out}")
            return 0
        if args.command == "verify":
            result = verify_roundtrip(
                args.checkpoint,
                out_dir=args.out,
                reload_path=args.reload,
                overwrite=True,
            )
            print(json.dumps(result))
            return 0 if result["identical"] else 1
    except (ValueError, OSError, SafetensorsError) as exc:
        print(f"hf-export: error: {exc}", file=sys.stderr)
        return 1
    return 2


def main_export() -> None:
    sys.exit(main(["export", *sys.argv[1:]]))


def main_import() -> None:
    sys.exit(main(["import", *sys.argv[1:]]))


if __name__ == "__main__":
    sys.exit(main())
