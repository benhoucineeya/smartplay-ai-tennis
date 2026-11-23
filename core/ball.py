"""Constant-velocity Kalman tracker for the ball (observed vs predicted kept apart)."""
import numpy as np

class BallKalman:
    def __init__(s, dt=1 / 25, max_gap=8, gate=120.0, q=50.0, r=9.0):
        s.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        s.H, s.Q, s.R = np.eye(2, 4), np.eye(4) * q, np.eye(2) * r
        s.x = s.P = None; s.gap = 0; s.gate, s.max_gap = gate, max_gap

    def step(s, obs):
        """obs = (x, y) or None -> (x, y, observed) or None when the track is lost."""
        if s.x is None:
            if obs is None: return None
            s.x, s.P, s.gap = np.array([obs[0], obs[1], 0, 0], float), np.eye(4) * 100, 0
            return (float(obs[0]), float(obs[1]), True)
        s.x, s.P = s.F @ s.x, s.F @ s.P @ s.F.T + s.Q
        if obs is not None and np.linalg.norm(np.array(obs) - s.x[:2]) <= s.gate * (1 + s.gap):
            K = s.P @ s.H.T @ np.linalg.inv(s.H @ s.P @ s.H.T + s.R)
            s.x = s.x + K @ (np.array(obs) - s.H @ s.x); s.P = (np.eye(4) - K @ s.H) @ s.P; s.gap = 0
            return (float(s.x[0]), float(s.x[1]), True)
        s.gap += 1
        if s.gap > s.max_gap: s.x = None; return None   # prolonged loss: reset, draw nothing
        return (float(s.x[0]), float(s.x[1]), False)

def track_ball(observations, fps, **kw):
    kf, rows = BallKalman(dt=1 / fps, **kw), []
    for k, o in enumerate(observations):
        r = kf.step(o)
        if r: rows.append(dict(frame=k, x=r[0], y=r[1], observed=r[2]))
    return rows
