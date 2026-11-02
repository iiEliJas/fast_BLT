import json
import os
import tempfile

from fblt._checkpoint import (
    atomic_write,
    best_checkpoint,
    latest_checkpoint,
    model_info,
    resume_checkpoint,
    save_dir_layout,
)


def test_atomic_write_roundtrip():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "x.bin")
    atomic_write(p, b"hello")
    with open(p, "rb") as f:
        assert f.read() == b"hello"
    assert not os.path.exists(p + ".tmp")


def test_resume_latest_best():
    d = tempfile.mkdtemp()
    save_dir_layout(d)
    p1 = os.path.join(d, "checkpoints", "a.fblt")
    p2 = os.path.join(d, "checkpoints", "b.fblt")
    for p, loss in [(p1, 2.0), (p2, 1.0)]:
        meta = json.dumps({"best_loss": loss}).encode()
        header = b"FBLT" + (2).to_bytes(4, "little") + len(meta).to_bytes(4, "little")
        atomic_write(p, header + meta)
    assert latest_checkpoint(d) == p2
    assert best_checkpoint(d) == p2
    assert resume_checkpoint(d, "auto") == p2
    mi = model_info(p2)
    assert mi["size_bytes"] > 0
