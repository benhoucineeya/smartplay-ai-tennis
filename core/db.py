"""SQLite storage (sessions, tracks, ball, strokes). Videos are never stored in the DB."""
import sqlite3, datetime, pathlib, pandas as pd
DB = pathlib.Path(__file__).resolve().parent.parent / "data" / "smartplay.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, video TEXT, created TEXT, fps REAL, frames INTEGER, calibrated INTEGER);
CREATE TABLE IF NOT EXISTS tracks(session_id INTEGER REFERENCES sessions(id), frame INTEGER, t REAL, track_id INTEGER, x REAL, y REAL, conf REAL);
CREATE TABLE IF NOT EXISTS ball(session_id INTEGER REFERENCES sessions(id), frame INTEGER, x REAL, y REAL, observed INTEGER);
CREATE TABLE IF NOT EXISTS strokes(session_id INTEGER REFERENCES sessions(id), track_id INTEGER, t REAL, stroke TEXT, reason TEXT, elbow REAL, knee_min REAL, lean REAL);
CREATE TABLE IF NOT EXISTS players(id INTEGER PRIMARY KEY, name TEXT UNIQUE, consent INTEGER NOT NULL, created TEXT);
CREATE TABLE IF NOT EXISTS embeddings(player_id INTEGER REFERENCES players(id) ON DELETE CASCADE, vec BLOB);
CREATE INDEX IF NOT EXISTS idx_tracks ON tracks(session_id, track_id);"""
COLS = dict(tracks=["session_id", "frame", "t", "track_id", "x", "y", "conf"],
            ball=["session_id", "frame", "x", "y", "observed"],
            strokes=["session_id", "track_id", "t", "stroke", "reason", "elbow", "knee_min", "lean"])

def connect(path=DB):
    path = pathlib.Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path); c.execute("PRAGMA foreign_keys=ON"); c.executescript(SCHEMA); return c

def save_session(res, video, calibrated, path=DB):
    c = connect(path)
    with c:
        sid = c.execute("INSERT INTO sessions(video,created,fps,frames,calibrated) VALUES(?,?,?,?,?)",
                        (video, datetime.datetime.now().isoformat(timespec="seconds"), res["fps"], res["frames"], int(calibrated))).lastrowid
        for name, cols in COLS.items():
            df = res[name]
            if len(df): df.assign(session_id=sid)[cols].to_sql(name, c, if_exists="append", index=False)
    c.close(); return sid

def list_sessions(path=DB):
    c = connect(path); df = pd.read_sql("SELECT * FROM sessions ORDER BY id DESC", c); c.close(); return df
