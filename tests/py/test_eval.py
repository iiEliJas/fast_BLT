import os
import tempfile

from fblt.eval import extract_answer, render_prompt, run_task


def test_render_prompt():
    ex = {"prompt": "Q: 2+2? A:"}
    assert "2+2" in render_prompt("math", ex, 0)


def test_extract_answer():
    assert extract_answer("the answer is 4") == "4"


def test_run_task_schema():
    d = tempfile.mkdtemp()
    out = os.path.join(d, "r.json")
    rec = run_task("bpb", "m.fblt", "c.bin", 0, "cpu", out)
    assert rec["task"] == "bpb"
    assert "metrics" in rec
    assert os.path.exists(out)
