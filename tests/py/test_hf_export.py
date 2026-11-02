import os
import tempfile

from fblt.hf_export import export_to_hf, verify_roundtrip


def test_roundtrip():
    d = tempfile.mkdtemp()
    src = os.path.join(d, "m.fblt")
    with open(src, "wb") as f:
        f.write(b"FBLT test")
    out_dir = os.path.join(d, "hf")
    export_to_hf(src, out_dir)
    assert os.path.exists(os.path.join(out_dir, "config.json"))
    reload = os.path.join(d, "reload.fblt")
    res = verify_roundtrip(src, out_dir, reload)
    assert res["identical"] is True
    assert res["meta"]["model_type"] == "fblt"
