import pytest
pytest.importorskip("streamlit")
import pathlib, numpy as np, pandas as pd
from streamlit.testing.v1 import AppTest

def _res():
    t = pd.DataFrame(dict(frame=range(10), t=np.arange(10) * .08, track_id=1, x=np.linspace(100, 300, 10), y=np.linspace(400, 380, 10), conf=.9))
    b = pd.DataFrame(dict(frame=range(10), x=np.linspace(10, 90, 10), y=np.linspace(50, 80, 10), observed=[True] * 8 + [False] * 2))
    s = pd.DataFrame(dict(track_id=[1], t=[0.0], stroke=["Smash"], reason=["demo"], elbow=[170], knee_min=[165], lean=[10]))
    return dict(tracks=t, ball=b, strokes=s, previews=[np.zeros((48, 64, 3), np.uint8)], fps=25.0, fps_eff=12.5, frames=10, n_ball_obs=8)

def test_dashboard_renders_empty_and_with_results():
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60).run(); assert not at.exception, at.exception
    at.session_state["res"] = _res(); at.run(); assert not at.exception, at.exception
    at.sidebar.text_area[0].set_value("100,500 700,500 600,100 200,100").run(); assert not at.exception, at.exception
