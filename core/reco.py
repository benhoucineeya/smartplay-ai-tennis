"""Transparent baselines: TF-IDF content filtering, rule-based exercises, heuristic partner matching.
The CATALOG is SYNTHETIC demo data (no brands, no real ratings). No interaction history => no CF/NeuMF."""
import re, numpy as np
CATALOG = [
 dict(id="E1", kind="equipment", name="Diamond racket, high balance (demo)", text="diamond racket high balance hard core maximum power for attacking players who finish points at the net with smash and volley"),
 dict(id="E2", kind="equipment", name="Round racket, low balance (demo)", text="round racket low balance large sweet spot control comfort for defensive players at the back of the court who lob"),
 dict(id="E3", kind="equipment", name="Teardrop racket, medium balance (demo)", text="teardrop racket medium balance balanced power and control for versatile all-court players"),
 dict(id="E4", kind="equipment", name="Light racket + cushioned shoes (demo)", text="lightweight racket and cushioned shoes for fast movement endurance and court coverage"),
 dict(id="X1", kind="exercise", name="Net volley drill (demo)", text="net volley drill backhand and forehand volleys to strengthen offensive net play and reflexes"),
 dict(id="X2", kind="exercise", name="Back-wall defence drill (demo)", text="back wall defence drill lobs and rebounds off the glass to build patient defensive play"),
 dict(id="X3", kind="exercise", name="Overhead drill (demo)", text="overhead smash and bandeja drill with a feeder to improve overhead technique and power"),
 dict(id="X4", kind="exercise", name="Footwork circuit (demo)", text="footwork and endurance circuit lateral shuffles and sprints to improve court coverage"),
 dict(id="X5", kind="exercise", name="Lob technique drill (demo)", text="lob technique drill bend the knees and lift the ball high over the opponents at the net")]
STYLE_TEXT = {"Aggressive Volleyer": "attacking offensive net volley smash power", "Baseline Defender": "defensive back court lob glass control patient",
              "All-Court Runner": "endurance movement court coverage lightweight fast", "Balanced": "versatile balanced all-court control power"}
_tok = lambda s: re.findall(r"[a-z]+", s.lower())

class TfIdf:
    def __init__(s, docs):
        toks = [_tok(d) for d in docs]; s.idx = {t: i for i, t in enumerate(sorted({t for d in toks for t in d}))}
        df = np.zeros(len(s.idx))
        for d in toks:
            for t in set(d): df[s.idx[t]] += 1
        s.idf = np.log((1 + len(docs)) / (1 + df)) + 1          # smoothed idf
        s.M = np.array([s.vec(d) for d in docs])
    def vec(s, text):
        tk = _tok(text) if isinstance(text, str) else text; v = np.zeros(len(s.idx))
        for t in tk:
            if t in s.idx: v[s.idx[t]] += 1
        return v / max(len(tk), 1) * s.idf
    def sims(s, text):
        q = s.vec(text); return s.M @ q / (np.linalg.norm(s.M, axis=1) * np.linalg.norm(q) + 1e-9)   # cosine

def recommend(profile_text, kind, k=2, catalog=CATALOG):
    items = [c for c in catalog if c["kind"] == kind]; sc = TfIdf([c["text"] for c in items]).sims(profile_text)
    return [(round(float(sc[i]), 3), items[i]) for i in np.argsort(-sc)[:k]]

def play_style(m):
    """Thesis rules (Tab. 5) on calibrated metrics. Returns None if a needed metric is unavailable."""
    ok = lambda k: m.get(k) is not None and m.get(k) == m.get(k)
    if not all(ok(k) for k in ("pct_net", "pct_baseline", "mean_speed", "distance")): return None
    if m["pct_net"] > 40 and m["mean_speed"] > 1.5: return "Aggressive Volleyer"
    if m["pct_baseline"] > 70: return "Baseline Defender"
    if m["distance"] > 100: return "All-Court Runner"
    return "Balanced"

def exercise_tips(strokes):
    """Rule-based, from pose-derived stroke rows (DataFrame). Measured values + generic advice."""
    tips = []
    if len(strokes) == 0: return tips
    if (strokes.stroke == "Uncertain").mean() > 0.5: tips.append("More than half of the windows have missing keypoints: film closer/steadier before trusting any technique feedback.")
    lob = strokes[(strokes.stroke == "Lob") & strokes.knee_min.notna()]
    if len(lob) and lob.knee_min.mean() > 130: tips.append(f"Lobs: mean minimum knee angle {lob.knee_min.mean():.0f} deg, little knee bend. Try the lob technique drill.")
    sm = strokes[(strokes.stroke == "Smash") & strokes.elbow.notna()]
    if len(sm) and sm.elbow.mean() < 165: tips.append(f"Smashes: mean elbow angle {sm.elbow.mean():.0f} deg at the highest point. Try the overhead drill (reach higher).")
    return tips

_COMP = {("Baseline Defender", "Aggressive Volleyer"): 1.0, ("Aggressive Volleyer", "Baseline Defender"): 1.0,
         ("All-Court Runner", "Balanced"): .7, ("Balanced", "All-Court Runner"): .7}
def match_partners(target, cands, k=3):
    """Heuristic: complementary styles (60%) + similar self-declared level 1-10 (40%)."""
    sc = [(round(.6 * _COMP.get((target["style"], c["style"]), .5) + .4 * (1 - abs(target["level"] - c["level"]) / 9), 3), c["name"]) for c in cands]
    return sorted(sc, reverse=True)[:k]
