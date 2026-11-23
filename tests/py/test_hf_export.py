import json
import os
import random
import struct
import subprocess
import zlib
from pathlib import Path

import pytest

from fblt import _checkpoint, _safetensors
from fblt._provenance import sha256_file
from fblt.hf_export import (
    CARD_NAME,
    CONFIG_NAME,
    METADATA_NAME,
    SAFETENSORS_NAME,
    TENSOR_SCHEMA,
    export_to_hf,
    fblt_tensor_to_hf,
    hf_tensor_to_fblt,
    import_from_hf,
    main,
    verify_roundtrip,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

EMBED = 8
HIDDEN = 16
LAYERS = 1
NGRAM_TABLES = 2
PER_NGRAM_VOCAB = 5
VOCAB = 256
D0 = 257

_SELF = (
    "norm1",
    "attn_qkv",
    "attn_proj",
    "norm2",
    "ffn_up",
    "ffn_gate",
    "ffn_down",
)
_CROSS = ("cross_norm", "cross_q", "cross_k", "cross_v", "cross_proj")
_DIM = "<Q" if struct.calcsize("P") == 8 else "<I"


def canonical_spec(enc_layers=LAYERS):
    """Test-local copy of the C tensor order; deliberately not imported."""
    entries = [("enc.embed", (VOCAB, EMBED))]
    for index in range(NGRAM_TABLES):
        entries.append((f"enc.ngram{index}", (PER_NGRAM_VOCAB, EMBED)))
    for stack, layers, has_cross in (
        ("enc", enc_layers, True),
        ("glob", LAYERS, False),
        ("dec", LAYERS, True),
    ):
        for layer in range(layers):
            base = f"{stack}.L{layer}"
            entries.extend(
                [
                    (f"{base}.norm1", (EMBED,)),
                    (f"{base}.attn_qkv", (EMBED, 3 * EMBED)),
                    (f"{base}.attn_proj", (EMBED, EMBED)),
                    (f"{base}.norm2", (EMBED,)),
                    (f"{base}.ffn_up", (EMBED, HIDDEN)),
                    (f"{base}.ffn_gate", (EMBED, HIDDEN)),
                    (f"{base}.ffn_down", (HIDDEN, EMBED)),
                ]
            )
            if has_cross:
                entries.extend(
                    [
                        (f"{base}.cross_norm", (EMBED,)),
                        (f"{base}.cross_q", (EMBED, EMBED)),
                        (f"{base}.cross_k", (EMBED, EMBED)),
                        (f"{base}.cross_v", (EMBED, EMBED)),
                        (f"{base}.cross_proj", (EMBED, EMBED)),
                    ]
                )
    entries.append(("dec.lm_head", (EMBED, VOCAB)))
    entries.append(("dec.d0_embed", (D0, EMBED)))
    return entries


def default_metadata(enc_layers=LAYERS):
    return {
        "format": "fblt-checkpoint",
        "format_version": 2,
        "config": {
            "embed": EMBED,
            "hidden": HIDDEN,
            "enc_layers": enc_layers,
            "glob_layers": LAYERS,
            "dec_layers": LAYERS,
            "cross_attn": "all",
            "d0_mode": "learned",
            "diffusion": 1,
            "block_size": 4,
            "window": 32,
        },
        "training": {"step": 2, "optimizer": "adamw", "seed": 7},
    }


def _payload(name, shape):
    numel = 1
    for dim in shape:
        numel *= dim
    rng = random.Random(zlib.crc32(name.encode("utf-8")))
    return struct.pack(f"<{numel}f", *(rng.uniform(-1.0, 1.0) for _ in range(numel)))


def write_checkpoint(path, *, metadata=None, tensors=None, version=2, truncate=None):
    if metadata is None:
        metadata = default_metadata()
    if tensors is None:
        tensors = canonical_spec()
    data = b"FBLT" + struct.pack("<I", version)
    if version == 2:
        blob = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        data += struct.pack("<I", len(blob)) + blob
    data += struct.pack("<I", len(tensors))
    for name, shape in tensors:
        raw = name.encode("utf-8")
        data += struct.pack("<H", len(raw)) + raw + struct.pack("<I", len(shape))
        for dim in shape:
            data += struct.pack(_DIM, dim)
        data += _payload(name, shape)
    if truncate is not None:
        data = data[:truncate]
    path = os.fspath(path)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def rewrite_safetensors(src, dst, *, drop=(), shape_override=None, extra=()):
    """Rebuild a safetensors file, optionally dropping/reshaping/adding tensors."""
    metadata, tensors, _ = _safetensors.parse_header(src)
    shape_override = shape_override or {}
    items = []
    for name in sorted(tensors):
        if name in drop:
            continue
        shape = tuple(shape_override.get(name, tensors[name]["shape"]))
        items.append((name, shape, name))
    for name, shape in extra:
        items.append((name, tuple(shape), None))
    items.sort(key=lambda item: item[0])
    header = _safetensors.encode_header(
        [(name, shape) for name, shape, _ in items], metadata=metadata or None
    )
    tmp = os.fspath(dst) + ".tmp"
    with open(src, "rb") as source, open(tmp, "wb") as target:
        target.write(header)
        for name, shape, source_name in items:
            if source_name is None:
                numel = 1
                for dim in shape:
                    numel *= dim
                target.write(b"\0" * (4 * numel))
            else:
                _safetensors.copy_tensor(source, tensors[source_name], target)
    os.replace(tmp, dst)


def test_tensor_name_map_is_bijective():
    names = [name for name, _ in canonical_spec()]
    assert len(names) == 36
    hf_names = [fblt_tensor_to_hf(name) for name in names]
    assert len(set(hf_names)) == len(hf_names)
    assert all(name.startswith("blt.") for name in hf_names)
    for name, hf_name in zip(names, hf_names):
        assert hf_tensor_to_fblt(hf_name) == name
    with pytest.raises(ValueError):
        fblt_tensor_to_hf("enc.not_a_tensor")
    with pytest.raises(ValueError):
        hf_tensor_to_fblt("blt.not_a_tensor")


def test_export_writes_expected_layout(tmp_path):
    src = write_checkpoint(tmp_path / "model.fblt")
    out_dir = tmp_path / "hf"
    export_to_hf(src, out_dir)

    for name in (SAFETENSORS_NAME, CONFIG_NAME, METADATA_NAME, CARD_NAME):
        assert (out_dir / name).is_file()

    config = json.loads((out_dir / CONFIG_NAME).read_text())
    assert set(config) == {
        "architectures",
        "model_type",
        "format",
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "max_position_embeddings",
        "hidden_act",
        "rms_norm_eps",
        "rope_theta",
        "torch_dtype",
        "tie_word_embeddings",
        "fblt",
    }
    assert config["architectures"] == ["BLTForCausalLM"]
    assert config["model_type"] == "fblt"
    assert config["format"] == "fblt-hf/v1"
    assert config["vocab_size"] == VOCAB
    assert config["hidden_size"] == EMBED
    assert config["intermediate_size"] == HIDDEN
    assert config["num_hidden_layers"] == LAYERS
    assert config["num_attention_heads"] == 4
    assert config["num_key_value_heads"] == 4
    assert config["max_position_embeddings"] == 32
    assert config["hidden_act"] == "silu"
    assert config["rms_norm_eps"] == 1e-06
    assert config["rope_theta"] == 500000.0
    assert config["torch_dtype"] == "float32"
    assert config["tie_word_embeddings"] is False

    fblt_block = config["fblt"]
    assert fblt_block["format_version"] == 2
    assert fblt_block["tensor_schema"] == TENSOR_SCHEMA
    assert fblt_block["checkpoint_metadata"] == default_metadata()
    assert fblt_block["architecture"] == {
        "embed": EMBED,
        "hidden": HIDDEN,
        "enc_layers": LAYERS,
        "glob_layers": LAYERS,
        "dec_layers": LAYERS,
        "cross_attn": "all",
        "d0_mode": "learned",
        "vocab_size": VOCAB,
        "d0_vocab": D0,
        "ngram_tables": NGRAM_TABLES,
        "per_ngram_vocab": PER_NGRAM_VOCAB,
        "num_attention_heads": 4,
    }
    assert fblt_block["inference"] == {
        "block_size": 4,
        "window": 32,
        "diffusion": 1,
        "d0_mode": "learned",
        "entropy_patches": None,
        "entropy_lm": None,
    }
    assert fblt_block["training"] == {"step": 2, "optimizer": "adamw", "seed": 7}
    assert fblt_block["implementation_defaults"]["ngram_sizes"] == [3, 4]
    assert fblt_block["source_checkpoint"] == {
        "path": os.fspath(src),
        "size_bytes": os.path.getsize(src),
        "sha256": sha256_file(src),
    }
    assert (out_dir / METADATA_NAME).read_bytes() == _checkpoint.read_meta_raw(src)

    metadata, tensors, _ = _safetensors.parse_header(out_dir / SAFETENSORS_NAME)
    assert metadata == {}
    spec = dict(canonical_spec())
    assert len(tensors) == len(spec)
    source_layouts = {item.name: item for item in _checkpoint.tensor_layout(src)[2]}
    for fblt_name, shape in spec.items():
        hf_name = fblt_tensor_to_hf(fblt_name)
        assert hf_name.startswith("blt.")
        entry = tensors[hf_name]
        assert entry["dtype"] == "F32"
        assert tuple(entry["shape"]) == shape
        source = source_layouts[fblt_name]
        with open(src, "rb") as fh:
            fh.seek(source.offset)
            source_bytes = fh.read(source.nbytes)
        with open(out_dir / SAFETENSORS_NAME, "rb") as fh:
            assert _safetensors.read_tensor(fh, entry) == source_bytes


def test_export_is_deterministic(tmp_path):
    src = write_checkpoint(tmp_path / "model.fblt")
    export_to_hf(src, tmp_path / "a")
    export_to_hf(src, tmp_path / "b")
    assert (tmp_path / "a" / SAFETENSORS_NAME).read_bytes() == (
        tmp_path / "b" / SAFETENSORS_NAME
    ).read_bytes()
    assert (tmp_path / "a" / CONFIG_NAME).read_bytes() == (
        tmp_path / "b" / CONFIG_NAME
    ).read_bytes()


def test_import_is_byte_identical(tmp_path):
    src = write_checkpoint(tmp_path / "model.fblt")
    result = verify_roundtrip(src)
    assert result["identical"] is True
    assert result["metadata_identical"] is True
    assert result["tensor_count"] == 36
    assert result["problems"] == []
    assert result["source_sha256"] == result["restored_sha256"]
    assert result["source_sha256"] == sha256_file(src)

    reload_path = tmp_path / "restored.fblt"
    export_to_hf(src, tmp_path / "hf")
    import_from_hf(tmp_path / "hf", reload_path)
    assert sha256_file(reload_path) == sha256_file(src)


def test_export_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        export_to_hf(tmp_path / "absent.fblt", tmp_path / "hf")


def test_export_rejects_v1_checkpoint(tmp_path):
    src = write_checkpoint(tmp_path / "v1.fblt", version=1)
    with pytest.raises(ValueError, match="v2"):
        export_to_hf(src, tmp_path / "hf")


def test_export_rejects_truncated_payload(tmp_path):
    src = write_checkpoint(tmp_path / "model.fblt")
    with open(src, "rb") as fh:
        data = fh.read()
    truncated = tmp_path / "truncated.fblt"
    truncated.write_bytes(data[:-4])
    with pytest.raises(ValueError, match="truncated"):
        export_to_hf(truncated, tmp_path / "hf")


def test_export_rejects_metadata_layer_mismatch(tmp_path):
    src = write_checkpoint(
        tmp_path / "two_layers.fblt",
        metadata=default_metadata(enc_layers=2),
        tensors=canonical_spec(enc_layers=1),
    )
    with pytest.raises(ValueError, match="enc.L1"):
        export_to_hf(src, tmp_path / "hf")


def _exported_dir(tmp_path):
    src = write_checkpoint(tmp_path / "model.fblt")
    out_dir = tmp_path / "hf"
    export_to_hf(src, out_dir)
    return src, out_dir


def test_import_rejects_missing_tensor(tmp_path):
    _, out_dir = _exported_dir(tmp_path)
    rewrite_safetensors(
        out_dir / SAFETENSORS_NAME,
        out_dir / SAFETENSORS_NAME,
        drop=("blt.lm_head.weight",),
    )
    with pytest.raises(ValueError, match="missing tensor"):
        import_from_hf(out_dir, tmp_path / "restored.fblt")


def test_import_rejects_wrong_shape(tmp_path):
    _, out_dir = _exported_dir(tmp_path)
    qkv = "blt.encoder.layers.0.self_attn.qkv_proj.weight"
    rewrite_safetensors(
        out_dir / SAFETENSORS_NAME,
        out_dir / SAFETENSORS_NAME,
        shape_override={qkv: (24, 8)},
    )
    with pytest.raises(ValueError, match="shape"):
        import_from_hf(out_dir, tmp_path / "restored.fblt")


def test_import_rejects_bad_or_missing_format(tmp_path):
    _, out_dir = _exported_dir(tmp_path)
    config_path = out_dir / CONFIG_NAME
    config = json.loads(config_path.read_text())
    config["format"] = "something-else"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="format"):
        import_from_hf(out_dir, tmp_path / "restored.fblt")

    config.pop("format")
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="format"):
        import_from_hf(out_dir, tmp_path / "restored.fblt")


def test_import_rejects_unknown_tensor(tmp_path):
    _, out_dir = _exported_dir(tmp_path)
    rewrite_safetensors(
        out_dir / SAFETENSORS_NAME,
        out_dir / SAFETENSORS_NAME,
        extra=(("blt.mystery.weight", (2,)),),
    )
    with pytest.raises(ValueError, match="no fBLT mapping"):
        import_from_hf(out_dir, tmp_path / "restored.fblt")


def test_cli_verify_and_no_args(tmp_path, capsys):
    src = write_checkpoint(tmp_path / "model.fblt")
    assert main(["verify", "--checkpoint", str(src)]) == 0
    out = capsys.readouterr().out
    assert '"identical": true' in out

    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_refuses_non_empty_dir_without_overwrite(tmp_path, capsys):
    src = write_checkpoint(tmp_path / "model.fblt")
    out_dir = tmp_path / "hf"
    export_to_hf(src, out_dir)
    assert main(["export", "--checkpoint", str(src), "--out", str(out_dir)]) == 1
    assert "overwrite" in capsys.readouterr().err
    assert main(["export", "--checkpoint", str(src), "--out", str(out_dir), "--overwrite"]) == 0


def test_safetensors_library_interop(tmp_path):
    safetensors = pytest.importorskip("safetensors")
    numpy = pytest.importorskip("numpy")
    src = write_checkpoint(tmp_path / "model.fblt")
    out_dir = tmp_path / "hf"
    export_to_hf(src, out_dir)
    path = str(out_dir / SAFETENSORS_NAME)

    with safetensors.safe_open(path, framework="numpy") as opened:
        loaded = {name: opened.get_tensor(name) for name in opened.keys()}
    _, tensors, _ = _safetensors.parse_header(path)
    assert set(loaded) == set(tensors)
    for name, entry in tensors.items():
        assert tuple(loaded[name].shape) == tuple(entry["shape"])

    name = "blt.encoder.byte_embedding.weight"
    layouts = {item.name: item for item in _checkpoint.tensor_layout(src)[2]}
    source = layouts["enc.embed"]
    with open(src, "rb") as fh:
        fh.seek(source.offset)
        raw = fh.read(source.nbytes)
    expected = numpy.frombuffer(raw, dtype="<f4").reshape(tensors[name]["shape"])
    numpy.testing.assert_array_equal(loaded[name], expected)


_TRAIN = REPO_ROOT / "build" / "train_blt_d"
_INFER = REPO_ROOT / "build" / "infer"
_CORPUS = REPO_ROOT / "data" / "train.bin"
_INTEGRATION_READY = _TRAIN.is_file() and _INFER.is_file() and _CORPUS.is_file()


@pytest.mark.skipif(
    not _INTEGRATION_READY,
    reason="requires build/train_blt_d, build/infer and data/train.bin",
)
def test_trainer_checkpoint_roundtrip_and_inference(tmp_path):
    model = tmp_path / "model.fblt"
    subprocess.run(
        [
            str(_TRAIN),
            "--backend",
            "cpu",
            "--corpus",
            str(_CORPUS),
            "--embed",
            str(EMBED * 4),
            "--hidden",
            str(HIDDEN * 4),
            "--layers",
            str(LAYERS),
            "--window",
            "32",
            "--block-size",
            "4",
            "--steps",
            "2",
            "--save-weights",
            str(model),
            "--report-every",
            "1000",
        ],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        timeout=300,
    )
    assert model.is_file()

    result = verify_roundtrip(model)
    assert result["identical"] is True
    assert result["problems"] == []

    out_dir = tmp_path / "hf"
    restored = tmp_path / "restored.fblt"
    export_to_hf(model, out_dir)
    import_from_hf(out_dir, restored)
    assert sha256_file(restored) == sha256_file(model)

    first = tmp_path / "gen1.bin"
    second = tmp_path / "gen2.bin"
    for checkpoint, output in ((model, first), (restored, second)):
        subprocess.run(
            [
                str(_INFER),
                "--checkpoint",
                str(checkpoint),
                "--backend",
                "cpu",
                "--prompt",
                "int main",
                "--new-bytes",
                "16",
                "--block-size",
                "4",
                "--fixed-patches",
                "--output",
                str(output),
            ],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            timeout=120,
        )
    assert first.read_bytes() == second.read_bytes()
