import os
import tempfile

from fblt.data_pipeline import (
    load_sampler_state,
    resume_index,
    sampler_state,
    save_sampler_state,
    windows_for_docs,
)


def test_no_cross_boundary():
    docs = [b"abcd", b"efgh"]
    ws = windows_for_docs(docs, 4)
    assert len(ws) == 2


def test_sampler_resume():
    st = sampler_state(0, 0, 5, 7)
    p = os.path.join(tempfile.mkdtemp(), "s.json")
    save_sampler_state(p, st)
    st2 = load_sampler_state(p)
    assert resume_index(st2, 10) == 6
