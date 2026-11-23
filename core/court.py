"""Court calibration (4 clicked corners -> metres) and per-player metrics."""
import numpy as np, pandas as pd
COURT_W, COURT_L = 10.0, 20.0   # metres; net at y = 10

def homography(src, dst):
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    H = np.linalg.svd(np.array(A, float))[2][-1].reshape(3, 3)
    return H / H[2, 2]

def court_homography(corners_px):
    """corners: near-left, near-right, far-right, far-left (pixels)."""
    return homography(np.asarray(corners_px, float), [(0, 0), (COURT_W, 0), (COURT_W, COURT_L), (0, COURT_L)])

def to_court(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]

def zone(y):
    d = abs(y - COURT_L / 2)
    return "net" if d < 3.5 else "mid" if d < 6.5 else "baseline"

def metrics(df, H=None, vmax=12.0):
    """Per-track distance/speed/zones. Pixels only (no speed) if H is None.
    Steps faster than vmax m/s are treated as tracking errors and excluded."""
    rows = []
    for tid, g in df.groupby("track_id"):
        g = g.sort_values("t"); P = g[["x", "y"]].to_numpy(float)
        if H is not None: P = to_court(H, P)
        d = np.linalg.norm(np.diff(P, axis=0), axis=1); dt = np.diff(g["t"].to_numpy())
        r = dict(track_id=tid, frames=len(g), unit="m" if H is not None else "px")
        if H is None:
            r["distance"] = round(float(d.sum()), 1)
        else:
            v = d / np.maximum(dt, 1e-6); ok = v <= vmax
            z = [zone(y) for y in P[:, 1]]
            r.update(distance=round(float(d[ok].sum()), 1),
                     mean_speed=round(float(v[ok].mean()), 2) if ok.any() else None,
                     max_speed=round(float(v[ok].max()), 2) if ok.any() else None,
                     implausible_steps=int((~ok).sum()),
                     **{f"pct_{n}": round(100 * z.count(n) / len(z)) for n in ("net", "mid", "baseline")})
        rows.append(r)
    return pd.DataFrame(rows)
