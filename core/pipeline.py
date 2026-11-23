"""Video pipeline: YOLOv8-pose + ByteTrack (players), ball detector + Kalman, stroke rules."""
import cv2, numpy as np, pandas as pd
from . import pose_utils as pu
from .ball import track_ball

def stroke_rows(tid, kps, K, fps, hand, win_s=1.2, step_s=0.6):
    arr = np.full((K, 17, 2), np.nan)
    for k, v in kps.items(): arr[k] = v
    n, st = max(3, int(win_s * fps)), max(1, int(step_s * fps))
    sh, el, wr = pu.ARM[hand]; hp, kn, an = pu.LEG[hand]
    used, rows = [0, 5, 6, 11, 12, sh, el, wr, hp, kn, an], []
    for s in range(0, K - n + 1, st):
        w = arr[s:s + n]; nan = np.isnan(w[:, used])
        if nan.all(): continue
        if nan.any():
            rows.append(dict(track_id=tid, t=round(s / fps, 2), stroke="Uncertain", reason="missing keypoints", elbow=None, knee_min=None, lean=None)); continue
        f = pu.extract_features(w * np.array([1, -1]), hand, fps); label, why = pu.classify(f)
        rows.append(dict(track_id=tid, t=round(s / fps, 2), stroke=label, reason=why[-1], elbow=round(f["elbow_at_peak"]), knee_min=round(f["knee_min"]), lean=round(f["lean_max"])))
    return rows

def run_video(path, pose_w="yolov8n-pose.pt", ball_w="yolov8n.pt", ball_class=32, conf=0.25,
              stride=2, max_frames=600, hand="right", progress=None):
    from ultralytics import YOLO
    pm, bm = YOLO(pose_w), YOLO(ball_w)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError("Cannot open this video file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    tracks, kps, obs, previews, k, i = [], {}, [], [], 0, 0
    while k < max_frames:
        ok, fr = cap.read()
        if not ok: break
        if i % stride == 0:
            r = pm.track(fr, persist=True, tracker="bytetrack.yaml", classes=[0], conf=conf, verbose=False)[0]
            if r.boxes is not None and r.boxes.id is not None:
                xy = r.boxes.xyxy.cpu().numpy()
                for j, tid in enumerate(r.boxes.id.int().cpu().tolist()):
                    tracks.append(dict(frame=k, t=i / fps, track_id=tid, x=float((xy[j, 0] + xy[j, 2]) / 2), y=float(xy[j, 3]), conf=float(r.boxes.conf[j])))
                    if r.keypoints is not None:
                        kp = r.keypoints.xy[j].cpu().numpy().copy()
                        if r.keypoints.conf is not None: kp[r.keypoints.conf[j].cpu().numpy() < 0.3] = np.nan
                        kps.setdefault(tid, {})[k] = kp
            b = bm(fr, classes=[ball_class] if ball_class is not None else None, conf=0.05, verbose=False)[0]
            o = None
            if len(b.boxes):
                x1, y1, x2, y2 = b.boxes.xyxy[int(b.boxes.conf.argmax())].cpu().numpy(); o = ((x1 + x2) / 2, (y1 + y2) / 2)
            obs.append(o)
            if k % 10 == 0 and len(previews) < 40:
                img = r.plot()
                if o: cv2.circle(img, (int(o[0]), int(o[1])), 10, (0, 165, 255), 2)
                previews.append(cv2.cvtColor(cv2.resize(img, (640, int(img.shape[0] * 640 / img.shape[1]))), cv2.COLOR_BGR2RGB))
            k += 1
            if progress: progress(min(1.0, (i + 1) / total) if total else 0.0)
        i += 1
    cap.release()
    fe = fps / stride; strokes = []
    for tid, d in kps.items(): strokes += stroke_rows(tid, d, k, fe, hand)
    cols = dict(tracks=["frame", "t", "track_id", "x", "y", "conf"], ball=["frame", "x", "y", "observed"],
                strokes=["track_id", "t", "stroke", "reason", "elbow", "knee_min", "lean"])
    return dict(tracks=pd.DataFrame(tracks, columns=cols["tracks"]), ball=pd.DataFrame(track_ball(obs, fe), columns=cols["ball"]),
                strokes=pd.DataFrame(strokes, columns=cols["strokes"]), previews=previews, fps=fps, fps_eff=fe, frames=k, n_ball_obs=sum(o is not None for o in obs))
