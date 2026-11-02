import hashlib

from fblt.dataset import cache_path, download


def test_cache_path():
    p = cache_path("foo.bin")
    assert p.endswith("foo.bin")


def test_download_checksum(tmp_path):
    data = b"hello world"
    h = hashlib.sha256(data).hexdigest()
    src = tmp_path / "src.bin"
    src.write_bytes(data)
    dest = tmp_path / "dst.bin"
    download(str(src), str(dest), sha256=h)
    assert dest.read_bytes() == data
