import tempfile, pathlib, numpy as np, pandas as pd
from core import court, ball, db, pose_utils as pu

def test_homography_corners():
    src = np.array([[100, 500], [700, 500], [600, 100], [200, 100]], float)
    out = court.to_court(court.court_homography(src), src)
    assert np.allclose(out, [[0, 0], [10, 0], [10, 20], [0, 20]], atol=1e-6)

def test_ball_gap_predicted_then_reset():
    obs = [(i * 5.0, 100.0) for i in range(10)] + [None] * 3 + [(65.0, 100.0)] + [None] * 20
    rows = ball.track_ball(obs, 25)
    assert any(not r["observed"] for r in rows) and len(rows) < len(obs)

def test_implausible_speed_excluded():
    df = pd.DataFrame(dict(track_id=1, t=np.arange(5) * 0.04, x=np.arange(5) * 1.0, y=5.0))
    m = court.metrics(df, court.court_homography([(0, 0), (10, 0), (10, 20), (0, 20)]))
    assert m.implausible_steps[0] == 4 and m.mean_speed[0] is None

def test_db_roundtrip():
    p = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    res = dict(fps=25.0, frames=2, tracks=pd.DataFrame(dict(frame=[0], t=[0.0], track_id=[1], x=[1.0], y=[2.0], conf=[.9])),
               ball=pd.DataFrame(dict(frame=[0], x=[1.0], y=[1.0], observed=[True])), strokes=pd.DataFrame(columns=db.COLS["strokes"][1:]))
    sid = db.save_session(res, "v.mp4", False, p)
    assert len(db.list_sessions(p)) == 1 and sid == 1

def test_stroke_rules():
    for s in pu.STROKES:
        assert pu.classify(pu.extract_features(pu.synth_stroke(s, seed=1)))[0] == s
