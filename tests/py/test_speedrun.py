import json
import os
import tempfile

from fblt.speedrun import aggregate_leaderboard, run_speedrun


def test_speedrun_runner():
    d = tempfile.mkdtemp()
    r = run_speedrun("configs/x.yaml", d)
    assert r["status"] == "ok"


def test_aggregate_leaderboard():
    d = tempfile.mkdtemp()
    os.makedirs(d + "/res", exist_ok=True)
    with open(d + "/res/a.json", "w") as f:
        json.dump({"x": 1}, f)
    out = d + "/lb.json"
    rows = aggregate_leaderboard(d + "/res", out)
    assert len(rows) == 1
    assert os.path.exists(out)
