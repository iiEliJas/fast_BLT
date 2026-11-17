import hashlib
import json
import re

import pytest

import fblt.dataset as ds
from fblt.dataset import download_dataset, download_one, verify_dataset


def _local_spec(tmp_path, name, data, sha256):
    src = tmp_path / f"{name}.bin"
    src.write_bytes(data)
    spec = ds.DatasetSpec(name=name, url=str(src), sha256=sha256, size=len(data))
    ds.REGISTRY[name] = spec
    return src


def test_registry_entries_are_verified():
    # A registry entry without a URL or checksum used to download nothing and exit 0.
    for name, spec in ds.REGISTRY.items():
        assert spec.url.startswith(("http://", "https://")), name
        assert re.fullmatch(r"[0-9a-f]{64}", spec.sha256), name
        assert spec.size > 0, name
        assert spec.license and spec.license != "unknown", name
        assert spec.version, name


def test_download_one_checksum(tmp_path):
    data = b"hello world"
    h = hashlib.sha256(data).hexdigest()
    src = tmp_path / "src.bin"
    src.write_bytes(data)
    dest = tmp_path / "dst.bin"
    download_one(str(src), str(dest), sha256=h)
    assert dest.read_bytes() == data


def test_verify_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("FBLT_CACHE", str(tmp_path))
    data = b"abc"
    h = hashlib.sha256(data).hexdigest()
    src = _local_spec(tmp_path, "local", data, h)
    manifest_path = download_dataset("local")
    assert verify_dataset("local")
    manifest = json.loads(open(manifest_path).read())
    assert manifest["url"] == str(src)
    assert manifest["sha256"] == h
    assert manifest["size"] == len(data)


def test_rerun_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("FBLT_CACHE", str(tmp_path))
    data = b"abc"
    _local_spec(tmp_path, "local", data, hashlib.sha256(data).hexdigest())
    first = download_dataset("local")
    shard = tmp_path / "datasets" / "local" / "shard-00000.bin"
    mtime = shard.stat().st_mtime_ns
    assert download_dataset("local") == first
    assert shard.stat().st_mtime_ns == mtime


def test_unknown_dataset_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("FBLT_CACHE", str(tmp_path))
    with pytest.raises(ValueError, match="unknown dataset"):
        download_dataset("no-such-dataset")
    assert not (tmp_path / "datasets" / "no-such-dataset").exists()


def test_missing_url_raises_before_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("FBLT_CACHE", str(tmp_path))
    ds.REGISTRY["nourl"] = ds.DatasetSpec(name="nourl", url="", sha256="0" * 64)
    with pytest.raises(ValueError, match="no download URL"):
        download_dataset("nourl")
    assert not (tmp_path / "datasets" / "nourl").exists()


def test_missing_checksum_raises_before_writing(tmp_path, monkeypatch):
    monkeypatch.setenv("FBLT_CACHE", str(tmp_path))
    ds.REGISTRY["nosum"] = ds.DatasetSpec(name="nosum", url="https://example.com/x", sha256="")
    with pytest.raises(ValueError, match="SHA-256"):
        download_dataset("nosum")
    assert not (tmp_path / "datasets" / "nosum").exists()


def test_bad_checksum_writes_no_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("FBLT_CACHE", str(tmp_path))
    monkeypatch.setattr(ds, "RETRIES", 1)
    _local_spec(tmp_path, "bad", b"abc", "0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        download_dataset("bad")
    d = tmp_path / "datasets" / "bad"
    assert not (d / "manifest.json").exists()
    assert not (d / "shard-00000.bin").exists()
    assert not (d / "shard-00000.bin.tmp").exists()


def test_verify_rejects_tampered_shard(tmp_path, monkeypatch):
    monkeypatch.setenv("FBLT_CACHE", str(tmp_path))
    data = b"abc"
    _local_spec(tmp_path, "local", data, hashlib.sha256(data).hexdigest())
    download_dataset("local")
    assert verify_dataset("local")
    (tmp_path / "datasets" / "local" / "shard-00000.bin").write_bytes(b"abd")
    assert not verify_dataset("local")


def test_cli_exit_codes(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FBLT_CACHE", str(tmp_path))
    assert ds.main(["--list"]) == 0
    assert "tinyshakespeare" in capsys.readouterr().out
    assert ds.main(["--download", "no-such-dataset"]) == 1
    assert "unknown dataset" in capsys.readouterr().err
    assert ds.main(["--verify", "no-such-dataset"]) == 1
