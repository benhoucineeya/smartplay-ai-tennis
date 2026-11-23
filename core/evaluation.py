"""Metrics against HUMAN-labelled data. Without labels no accuracy can be claimed."""
import numpy as np, pandas as pd

def stroke_eval(pred, gt, win=1.2):
    """pred: track_id,t,stroke (window start). gt: track_id,t,stroke (stroke time). Uncertain windows are excluded and reported as coverage."""
    p = pred[pred.stroke != "Uncertain"].reset_index(drop=True); truth = []
    for _, r in p.iterrows():
        m = gt[(gt.track_id == r.track_id) & (gt.t >= r.t) & (gt.t <= r.t + win)]
        truth.append(m.stroke.iloc[0] if len(m) else "No stroke")
    cm = pd.crosstab(pd.Series(truth, name="true"), p.stroke.rename("pred")); L = sorted(set(cm.index) | set(cm.columns))
    cm = cm.reindex(index=L, columns=L, fill_value=0); rows = {}
    for c in L:
        tp, pr, rc = cm.at[c, c], cm[c].sum(), cm.loc[c].sum(); P, R = tp / pr if pr else 0, tp / rc if rc else 0
        rows[c] = dict(precision=round(P, 3), recall=round(R, 3), f1=round(2 * P * R / (P + R), 3) if P + R else 0.0, support=int(rc))
    return dict(confusion=cm, per_class=pd.DataFrame(rows).T, accuracy=float(np.mean(np.array(truth) == p.stroke.to_numpy())) if len(p) else None,
                coverage=len(p) / max(len(pred), 1))

def ball_eval(traj, gt):
    """gt: frame,x,y (pixels) for frames where the ball is visible; compares with OBSERVED trajectory points only."""
    m = gt.merge(traj[traj.observed.astype(bool)], on="frame", suffixes=("_gt", "_pr"))
    e = np.hypot(m.x_gt - m.x_pr, m.y_gt - m.y_pr)
    return dict(gt_frames=len(gt), matched=len(m), recall=len(m) / max(len(gt), 1),
                mean_px_error=float(e.mean()) if len(m) else None, median_px_error=float(e.median()) if len(m) else None)
