import json
import os
import subprocess
import sys
import tempfile


def test_eval_smoke():
    d = tempfile.mkdtemp()
    out = os.path.join(d, "eval.json")
    cmd = [
        sys.executable,
        "-m",
        "fblt.eval",
        "--task",
        "bpb",
        "--checkpoint",
        "m.fblt",
        "--corpus",
        "data/train.bin",
        "--out",
        out,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0
    with open(out) as f:
        rec = json.load(f)
    assert rec["task"] == "bpb"
