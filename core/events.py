"""Ball event CANDIDATES from an image-space trajectory (single camera => always uncertain)."""
import numpy as np

def detect_events(rows, fps, min_speed=40.0, min_angle=50.0, w=3):
    """rows: dicts(frame,x,y,observed); fps = processed-frame rate. Image coords (y points down).
    min_speed is in px/s and depends on the camera: tune it. Confidence = angle strength x share of observed points."""
    if len(rows) < 2 * w + 1: return []
    f = np.array([r["frame"] for r in rows]); P = np.array([[r["x"], r["y"]] for r in rows], float)
    ob = np.array([bool(r["observed"]) for r in rows], float); out = []
    for i in range(w, len(rows) - w):
        if f[i + w] - f[i - w] != 2 * w: continue          # hole in the trajectory: no evidence
        v1, v2 = (P[i] - P[i - w]) * fps / w, (P[i + w] - P[i]) * fps / w
        s1, s2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if min(s1, s2) < min_speed: continue
        ang = float(np.degrees(np.arccos(np.clip(v1 @ v2 / (s1 * s2), -1, 1))))
        if ang < min_angle: continue
        if v1[1] > 0 and v2[1] < 0 and abs(v1[1]) > 0.5 * abs(v1[0]): kind = "floor_bounce_candidate"
        elif np.sign(v1[0]) != np.sign(v2[0]) and abs(v1[0]) > abs(v1[1]): kind = "wall_or_racket_candidate"
        else: kind = "direction_change"
        out.append(dict(frame=int(f[i]), t=round(float(f[i]) / fps, 2), kind=kind, angle=round(ang, 1),
                        confidence=round(min(1.0, ang / 120) * float(ob[i - w:i + w + 1].mean()), 2)))
    keep = []
    for e in sorted(out, key=lambda e: -e["confidence"]):   # non-maximum suppression
        if all(abs(e["frame"] - k["frame"]) > 2 * w for k in keep): keep.append(e)
    return sorted(keep, key=lambda e: e["frame"])
