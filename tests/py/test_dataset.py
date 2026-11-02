import hashlib

import fblt.dataset as ds
from fblt.dataset import download_dataset, download_one, verify_dataset


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
    # register a local dataset
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    h = hashlib.sha256(b"abc").hexdigest()
    ds.REGISTRY["local"] = ds.DatasetSpec(
        name="local", url=str(src), sha256=h, shards=1, license="MIT", source="local"
    )
    download_dataset("local")
    assert verify_dataset("local")
