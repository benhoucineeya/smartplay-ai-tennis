"""
pose_utils.py - Pose-based stroke detection for padel (proposal / prototype).

Everything works on COCO-17 keypoints, shape (T, 17, 2), with y pointing UP.
YOLOv8-pose outputs this format, and so does RTMPose (e.g. through `rtmlib`),
so the backend can be swapped without touching the rules below.
"""
from __future__ import annotations
import numpy as np

# COCO-17 order: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows,
# 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles (left, right pairs)
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
ARM = {"right": (6, 8, 10), "left": (5, 7, 9)}   # shoulder, elbow, wrist
LEG = {"right": (12, 14, 16), "left": (11, 13, 15)}  # hip, knee, ankle
STROKES = ["Smash", "Bandeja", "Volley", "Lob"]

# Decision thresholds (placeholders: calibrate on labelled padel clips)
THRESH = dict(min_amp=0.4, above_head=0.1, smash_elbow=150.0,
              smash_vdown=5.0, lob_vup=2.5, lob_knee=140.0)

# ----------------------------------------------------------------------------
# 1) Synthetic skeletons (simulation mode)
# keyframe = (time 0-1, upper-arm angle, forearm angle, knee angle, trunk lean)
# arm angles are absolute directions in degrees (0 = right, 90 = up)
# ----------------------------------------------------------------------------
READY = (0, -70, 30, 175, 0)
KEYFRAMES = {
    "Smash":   [READY, (.30, 100, 170, 165, -10), (.40, 80, 85, 170, 0),
                (.55, -40, -60, 170, 10), (1, -70, 30, 175, 0)],
    "Bandeja": [READY, (.35, 100, 160, 165, -5), (.50, 70, 130, 165, 0),
                (.85, -20, -10, 170, 5), (1, -70, 30, 175, 0)],
    "Volley":  [READY, (.30, -10, 50, 170, 5), (.45, -40, -5, 170, 8),
                (1, -70, 30, 175, 0)],
    "Lob":     [READY, (.25, -80, -95, 110, 15), (.45, 10, 20, 165, -5),
                (.70, -30, -20, 170, 0), (1, -70, 30, 175, 0)],
}
DURATION = {"Smash": 1.2, "Bandeja": 1.6, "Volley": 1.0, "Lob": 1.4}


def _pose(a1, a2, knee, lean, L1=0.55, L2=0.5, thigh=0.9):
    """One skeleton, torso length = 1 unit, right arm animated, facing +x."""
    phi = np.radians((180 - knee) / 2)
    hip = np.array([0.0, -1.8 * (1 - np.cos(phi))])
    r = np.radians(-lean)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    up = lambda x, y: hip + R @ np.array([x, y])
    ls, rs = up(-.28, 1.0), up(.28, 1.0)

    def arm(s, d1, d2):
        el = s + L1 * np.array([np.cos(np.radians(d1)), np.sin(np.radians(d1))])
        return el, el + L2 * np.array([np.cos(np.radians(d2)), np.sin(np.radians(d2))])

    re, rw = arm(rs, a1, a2)
    le, lw = arm(ls, -100, -95)
    kx, ky, ay = thigh * np.sin(phi), hip[1] - thigh * np.cos(phi), hip[1] - 1.8 * np.cos(phi)
    return np.array([
        up(0, 1.4), up(-.05, 1.45), up(.05, 1.45), up(-.12, 1.4), up(.12, 1.4),
        ls, rs, le, re, lw, rw,
        [-.2, hip[1]], [.2, hip[1]], [-.2 + kx, ky], [.2 + kx, ky], [-.2, ay], [.2, ay]])


def _smooth(u):
    return u * u * (3 - 2 * u)


def synth_stroke(stroke, fps=25, noise=0.01, variation=4.0, seed=None):
    """Generate a (T,17,2) keypoint sequence for one stroke."""
    rng = np.random.default_rng(seed)
    keys = np.array(KEYFRAMES[stroke], float)
    keys[1:-1, 1:3] += rng.normal(0, variation, (len(keys) - 2, 2))
    T = int(DURATION[stroke] * fps)
    seq = []
    for t in np.linspace(0, 1, T):
        j = min(np.searchsorted(keys[:, 0], t, side="right") - 1, len(keys) - 2)
        u = _smooth(np.clip((t - keys[j, 0]) / (keys[j + 1, 0] - keys[j, 0]), 0, 1))
        seq.append(_pose(*(keys[j, 1:] * (1 - u) + keys[j + 1, 1:] * u)))
    return np.array(seq) + rng.normal(0, noise, (T, 17, 2))


# ----------------------------------------------------------------------------
# 2) Features, rule-based classifier, posture feedback
# ----------------------------------------------------------------------------
def _angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def extract_features(seq, hand="right", fps=25):
    s, e, w = ARM[hand]
    h, k, a = LEG[hand]
    sh, hip = (seq[:, 5] + seq[:, 6]) / 2, (seq[:, 11] + seq[:, 12]) / 2
    torso = np.median(np.linalg.norm(sh - hip, axis=1)) + 1e-9   # scale-free units
    wrist_h = (seq[:, w, 1] - seq[:, s, 1]) / torso               # wrist vs shoulder
    head_h = (seq[:, 0, 1] - seq[:, s, 1]) / torso
    smooth = np.convolve(np.pad(wrist_h, 1, mode="edge"), np.ones(3) / 3, "valid")
    speed = np.gradient(smooth) * fps                             # torso units / s
    elbow, knee = _angle(seq[:, s], seq[:, e], seq[:, w]), _angle(seq[:, h], seq[:, k], seq[:, a])
    vec = sh - hip
    lean = np.degrees(np.arctan2(vec[:, 0], vec[:, 1]))
    sh_line = seq[:, 6] - seq[:, 5]
    tilt = np.degrees(np.arctan2(np.abs(sh_line[:, 1]), np.abs(sh_line[:, 0])))
    pk = int(np.argmax(wrist_h))
    return dict(amp=float(np.ptp(wrist_h)), above_head=float(np.max(wrist_h - head_h)),
                elbow_at_peak=float(elbow[pk]), vup=float(speed.max()),
                vdown=float(-speed[pk:].min()), knee_min=float(knee.min()),
                lean_max=float(np.abs(lean).max()), tilt_max=float(tilt.max()),
                series=dict(wrist_h=wrist_h, speed=speed, elbow=elbow, knee=knee), peak=pk)


def classify(f, th=THRESH):
    """Return (label, list of human-readable reasons)."""
    if f["amp"] < th["min_amp"]:
        return "No stroke", [f"Wrist barely moves (amplitude {f['amp']:.2f} < {th['min_amp']})"]
    if f["above_head"] > th["above_head"]:
        why = [f"Wrist goes above the head (+{f['above_head']:.2f} torso units)"]
        if f["elbow_at_peak"] >= th["smash_elbow"] and f["vdown"] >= th["smash_vdown"]:
            return "Smash", why + [f"Arm extended at contact ({f['elbow_at_peak']:.0f}°) and fast downward swing ({f['vdown']:.1f}/s)"]
        return "Bandeja", why + [f"Elbow bent ({f['elbow_at_peak']:.0f}°) or controlled swing ({f['vdown']:.1f}/s)"]
    if f["knee_min"] < th["lob_knee"] and f["vup"] >= th["lob_vup"]:
        return "Lob", [f"Low base (knee {f['knee_min']:.0f}°) and fast upward swing ({f['vup']:.1f}/s)"]
    return "Volley", [f"Compact motion below head height (amplitude {f['amp']:.2f})"]


def coaching_tips(label, f):
    """Illustrative tips only: thresholds must be validated with coaches."""
    tips = []
    if label == "Smash" and f["elbow_at_peak"] < 165:
        tips.append(f"Reach higher at contact: elbow only {f['elbow_at_peak']:.0f}°, aim for a nearly straight arm.")
    if label == "Bandeja" and f["lean_max"] > 20:
        tips.append("Trunk leans too much; stay balanced and keep the racket above the head.")
    if label == "Lob" and f["knee_min"] > 130:
        tips.append(f"Bend the knees more (min {f['knee_min']:.0f}°): the lift should come from the legs.")
    if label == "Volley" and f["amp"] > 1.0:
        tips.append("Volley swing is too long; keep it short and compact in front of you.")
    if f["tilt_max"] > 25:
        tips.append(f"Shoulders tilt up to {f['tilt_max']:.0f}°: check balance and rotation.")
    return tips or ["No major posture issue detected by the current rules."]


# ----------------------------------------------------------------------------
# 3) Real video with YOLOv8-pose (optional, needs ultralytics + opencv)
# ----------------------------------------------------------------------------
def extract_keypoints_yolo(path, model_name="yolov8n-pose.pt", stride=2, max_frames=600, width=640):
    """Run YOLOv8-pose, keep the largest person per frame.
    Returns keypoints (T,17,2) in IMAGE coords (y down), small RGB frames, effective fps.
    To use RTMPose instead, replace the `model(...)` call by an rtmlib wrapper
    that returns the same (17,2) array."""
    import cv2
    from ultralytics import YOLO
    model = YOLO(model_name)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    kps, frames, i = [], [], 0
    while len(kps) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
            res = model(frame, verbose=False)[0]
            if res.keypoints is not None and len(res.boxes) > 0:
                b = res.boxes.xyxy
                big = int(((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax())
                kps.append(res.keypoints.xy[big].cpu().numpy())
            else:
                kps.append(np.full((17, 2), np.nan))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    arr = np.array(kps, float)
    ok = ~np.isnan(arr[:, 0, 0])
    if ok.sum() < 5:
        raise ValueError("No person detected in the video.")
    idx = np.arange(len(arr))
    for j in range(17):                      # fill missing frames by interpolation
        for c in range(2):
            arr[:, j, c] = np.interp(idx, idx[ok], arr[ok, j, c])
    return arr, frames, fps / stride


def analyze_windows(kp_img, fps, hand="right", win_s=1.2, step_s=0.6):
    """Slide a window over the clip and classify each one."""
    kp = kp_img * np.array([1, -1])          # image coords -> y up
    n, step = int(win_s * fps), max(1, int(step_s * fps))
    rows = []
    for s in range(0, len(kp) - n + 1, step):
        f = extract_features(kp[s:s + n], hand, fps)
        label, why = classify(f)
        rows.append(dict(start_s=round(s / fps, 2), stroke=label, reason=why[-1],
                         elbow=round(f["elbow_at_peak"]), knee_min=round(f["knee_min"]),
                         lean=round(f["lean_max"]), tips=" ".join(coaching_tips(label, f))))
    return rows
