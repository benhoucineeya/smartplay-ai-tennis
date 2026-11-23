"""Opt-in player identity: embeddings + cosine matching. Consent required, unknown allowed, deletable."""
import datetime, collections, numpy as np
from . import db
_norm = lambda v: (lambda a: a / (np.linalg.norm(a) + 1e-9))(np.asarray(v, np.float32))

def enroll(name, emb, consent, path=db.DB):
    if not consent: raise PermissionError("Explicit consent is required to store a face embedding.")
    c = db.connect(path)
    with c:
        row = c.execute("SELECT id FROM players WHERE name=?", (name,)).fetchone()
        pid = row[0] if row else c.execute("INSERT INTO players(name,consent,created) VALUES(?,?,?)", (name, 1, datetime.datetime.now().isoformat(timespec="seconds"))).lastrowid
        c.execute("INSERT INTO embeddings VALUES(?,?)", (pid, _norm(emb).tobytes()))
    c.close(); return pid

def match(emb, threshold=0.45, path=db.DB):
    """Return (name or None, best cosine similarity). Threshold is a placeholder: calibrate per model."""
    c = db.connect(path); rows = c.execute("SELECT p.name, e.vec FROM embeddings e JOIN players p ON p.id=e.player_id").fetchall(); c.close()
    if not rows: return None, 0.0
    q = _norm(emb); n, s = max(((n, float(q @ np.frombuffer(b, np.float32))) for n, b in rows), key=lambda r: r[1])
    return (n, s) if s >= threshold else (None, s)

def delete_player(name, path=db.DB):
    c = db.connect(path)
    with c: n = c.execute("DELETE FROM players WHERE name=?", (name,)).rowcount   # embeddings cascade
    c.close(); return n

def list_players(path=db.DB):
    c = db.connect(path); r = [x[0] for x in c.execute("SELECT name FROM players ORDER BY name")]; c.close(); return r

class TrackVote:
    """Assign a name to a track only after n consecutive identical matches; manual override possible."""
    def __init__(s, n=3): s.n, s.h, s.fixed = n, collections.defaultdict(lambda: collections.deque(maxlen=n)), {}
    def update(s, tid, name):
        if tid in s.fixed: return s.fixed[tid]
        s.h[tid].append(name); d = s.h[tid]
        return d[0] if len(d) == s.n and len(set(d)) == 1 else None
    def override(s, tid, name): s.fixed[tid] = name

def embed_face(img_bgr):
    """Optional backend (pip install insightface onnxruntime). Check the licence of the pretrained weights before any commercial use."""
    try: from insightface.app import FaceAnalysis
    except ImportError: return None
    global _app
    if "_app" not in globals(): _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]); _app.prepare(ctx_id=-1)
    faces = _app.get(img_bgr); return faces[0].normed_embedding if faces else None
