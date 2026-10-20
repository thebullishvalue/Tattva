"""The study runs the normal pipeline: run_target on the fixture IS the app's run.

``pipeline.run.run_target`` chains the same step functions app.py's phases
call (fetch, prepare, Mūla, Swayam, the closed loop) without Streamlit; the
offline pooled-prior study drives it. On the Gold fixture its call_ts equals
the call_ts the app stored (AppTest, offline) byte for byte — index, columns,
dtypes, every value and the provenance stamp — and so does its verdict.
"""
import numpy as np

from pipeline.run import run_target


def _column_bytes(col) -> bytes:
    a = col.to_numpy()
    if a.dtype == object:
        return "\x1f".join(map(repr, a)).encode("utf-8")
    return np.ascontiguousarray(a).tobytes()


def test_run_target_is_the_app_pipeline_byte_for_byte(gold_app, offline_fetch):
    ss = gold_app.session_state
    app_ct = ss["call_ts"]
    bundle = run_target("Gold")
    ct = bundle["call_ts"]
    assert ct.index.equals(app_ct.index) and ct.index.dtype == app_ct.index.dtype
    assert list(ct.columns) == list(app_ct.columns)
    assert list(ct.dtypes) == list(app_ct.dtypes)
    for c in ct.columns:
        assert _column_bytes(ct[c]) == _column_bytes(app_ct[c]), c
    assert ct.attrs == app_ct.attrs
    assert bundle["verdict"] == ss["verdict"]
    for key in ("asof", "horizon", "exec_lag", "epoch", "fingerprint", "priors_meta"):
        assert bundle[key] == ss[key], key
