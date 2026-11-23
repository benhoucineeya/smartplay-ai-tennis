import tempfile, pathlib, numpy as np, pandas as pd, pytest
from core import events, reco, identity, context, evaluation as ev
from scripts import prepare_datasets as pd_

def test_floor_bounce_candidate():
    rows = [dict(frame=f, x=5.0 * f, y=100 + 12 * f if f < 10 else 220 - 12 * (f - 10), observed=True) for f in range(20)]
    e = events.detect_events(rows, 25); assert e and e[0]["kind"] == "floor_bounce_candidate" and abs(e[0]["frame"] - 10) <= 1

def test_no_event_on_straight_line_or_gap():
    rows = [dict(frame=f, x=5.0 * f, y=50.0, observed=True) for f in range(20)]; assert events.detect_events(rows, 25) == []

def test_reco_and_style():
    assert reco.recommend(reco.STYLE_TEXT["Aggressive Volleyer"], "equipment")[0][1]["id"] == "E1"
    assert reco.recommend(reco.STYLE_TEXT["Baseline Defender"], "equipment")[0][1]["id"] == "E2"
    assert reco.play_style(dict(pct_net=50, pct_baseline=10, mean_speed=2, distance=50)) == "Aggressive Volleyer"
    assert reco.play_style(dict(pct_net=None)) is None
    assert reco.match_partners(dict(style="Baseline Defender", level=5), [dict(name="a", style="Aggressive Volleyer", level=5), dict(name="b", style="Baseline Defender", level=5)])[0][1] == "a"

def test_identity_consent_match_delete_vote():
    p = pathlib.Path(tempfile.mkdtemp()) / "i.db"; rng = np.random.default_rng(0); a, b = rng.normal(size=512), rng.normal(size=512)
    with pytest.raises(PermissionError): identity.enroll("x", a, False, p)
    identity.enroll("alice", a, True, p)
    assert identity.match(a + rng.normal(scale=.1, size=512), path=p)[0] == "alice" and identity.match(b, path=p)[0] is None
    assert identity.delete_player("alice", p) == 1 and identity.match(a, path=p) == (None, 0.0)
    v = identity.TrackVote(3); assert [v.update(1, "a"), v.update(1, "a"), v.update(1, "a")] == [None, None, "a"]
    v.override(1, "b"); assert v.update(1, "a") == "b"

def test_context():
    with pytest.raises(ValueError): context.session_context(fatigue=11)
    w = context.fetch_weather(36.8, 10.18, timeout=2); assert w is None or w["source"] == "open-meteo.com"
    assert context.notes(dict(fatigue=9))

def test_evaluation():
    pred = pd.DataFrame(dict(track_id=[1, 1, 1], t=[0.0, 1.0, 2.0], stroke=["Smash", "Lob", "Uncertain"]))
    gt = pd.DataFrame(dict(track_id=[1, 1], t=[0.5, 1.5], stroke=["Smash", "Volley"]))
    r = ev.stroke_eval(pred, gt); assert r["accuracy"] == 0.5 and abs(r["coverage"] - 2 / 3) < 1e-9
    b = ev.ball_eval(pd.DataFrame(dict(frame=[0, 1], x=[0., 3.], y=[0., 4.], observed=[True, True])), pd.DataFrame(dict(frame=[1, 2], x=[0., 0.], y=[0., 0.])))
    assert b["matched"] == 1 and b["mean_px_error"] == 5.0

def test_split_no_leakage_and_label_validation():
    s = pd_.split_by_video([f"v{i}" for i in range(10)]); allv = s["train"] + s["val"] + s["test"]
    assert len(allv) == len(set(allv)) == 10 and s["val"] and s["test"]
    d = pathlib.Path(tempfile.mkdtemp()); (d / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n3 0.5 0.5 0.1 0.1\n0 1.5 0.5 0.1 0.1\n")
    pr, ct = pd_.validate_yolo_labels(d, 2); assert len(pr) == 2 and ct[0] == 1
