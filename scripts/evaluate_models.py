"""Usage: python -m scripts.evaluate_models strokes pred.csv gt.csv | ball traj.csv gt.csv"""
import sys, pandas as pd
from core import evaluation as ev
if __name__ == "__main__":
    kind, a, b = sys.argv[1:4]; A, B = pd.read_csv(a), pd.read_csv(b)
    if kind == "strokes":
        r = ev.stroke_eval(A, B); print(r["confusion"], "\n", r["per_class"], f"\naccuracy={r['accuracy']} coverage={r['coverage']:.2f}")
    else: print(ev.ball_eval(A, B))
