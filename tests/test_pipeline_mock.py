"""Runs core.pipeline.run_video end-to-end with a FAKE YOLO (scripted pose + ball). Verifies the plumbing, NOT detection quality."""
import sys, types, numpy as np, pytest, cv2
from core import pose_utils as pu

class T:
    def __init__(s, a): s.a = np.asarray(a)
    def cpu(s): return s
    def numpy(s): return s.a
    def int(s): return T(s.a.astype(int))
    def tolist(s): return s.a.tolist()
    def argmax(s): return int(s.a.argmax())
    def __getitem__(s, k): return T(s.a[k])
    def __len__(s): return len(s.a)
    def __float__(s): return float(s.a)
    def __lt__(s, o): return s.a < o
class Boxes:
    def __init__(s, xyxy, conf, ids=None): s.xyxy, s.conf, s.id = T(xyxy), T(conf), (T(ids) if ids is not None else None)
    def __len__(s): return len(s.xyxy)
class Kps:
    def __init__(s, xy, conf): s.xy, s.conf = T(xy), T(conf)
class Res:
    def __init__(s, boxes, kps=None, frame=None): s.boxes, s.keypoints, s._f = boxes, kps, frame
    def plot(s): return s._f.copy()

SEQ = pu.synth_stroke("Smash", noise=0, variation=0, seed=0)    # (30,17,2), y up
class FakeYOLO:
    n = 0
    def __init__(s, w): s.pose = "pose" in w
    def track(s, fr, **kw):
        kp = np.stack([300 + 100 * SEQ[FakeYOLO.n % len(SEQ)][:, 0], 400 - 100 * SEQ[FakeYOLO.n % len(SEQ)][:, 1]], 1); FakeYOLO.n += 1
        box = [[kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()]]
        return [Res(Boxes(box, [.9], [7]), Kps(kp[None], np.ones((1, 17))), fr)]
    def __call__(s, fr, **kw):
        k = FakeYOLO.n
        if 20 <= k <= 23: return [Res(Boxes(np.zeros((0, 4)), np.zeros(0)))]          # simulated misses
        return [Res(Boxes([[50 + 5 * k, 100, 56 + 5 * k, 106]], [.5]))]

def test_run_video_with_fake_models(tmp_path, monkeypatch):
    path = str(tmp_path / "v.avi"); w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 480))
    if not w.isOpened(): pytest.skip("OpenCV cannot write test video")
    for _ in range(70): w.write(np.zeros((480, 640, 3), np.uint8))
    w.release(); FakeYOLO.n = 0
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    from core.pipeline import run_video
    r = run_video(path, "fake-pose.pt", "fake-ball.pt", 32, 0.25, 1, 70, "right")
    assert r["frames"] == 70 and r["tracks"].track_id.nunique() == 1 and len(r["tracks"]) == 70
    assert (~r["ball"].observed.astype(bool)).any() and r["n_ball_obs"] < 70               # Kalman bridged the simulated misses
    assert "Smash" in set(r["strokes"].stroke) and len(r["previews"]) >= 1
    with pytest.raises(ValueError): run_video(str(tmp_path / "missing.avi"))
