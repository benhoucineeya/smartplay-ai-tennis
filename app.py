"""SmartPlay AI dashboard.  Run: py -m streamlit run app.py"""
import re, tempfile, pathlib
import pandas as pd, numpy as np, streamlit as st, matplotlib.pyplot as plt, cv2
from core import court, db, events, reco, context, identity

st.set_page_config(page_title="SmartPlay AI", layout="wide")
st.title("SmartPlay AI")
st.caption("All outputs are model predictions on your video. Stroke labels come from an unvalidated rule-based baseline.")
sb = st.sidebar
conf = sb.slider("Detection confidence", 0.05, 0.9, 0.25, 0.05)
stride = sb.slider("Process 1 frame out of", 1, 6, 2)
max_frames = sb.number_input("Max processed frames", 50, 3000, 600, 50)
hand = sb.radio("Dominant hand (pose)", ["right", "left"])
pose_w = sb.text_input("Pose weights", "yolov8n-pose.pt")
ball_w = sb.text_input("Ball weights", "yolov8n.pt")
ball_cls = sb.text_input("Ball class id (blank = all classes)", "32")
txt = sb.text_area("Court corners in pixels (near-left, near-right, far-right, far-left)", placeholder="120,640 1160,640 900,220 380,220")
nums = re.findall(r"-?\d+\.?\d*", txt)
H = court.court_homography(np.array(nums, float).reshape(4, 2)) if len(nums) == 8 else None
if txt.strip() and H is None: sb.warning("Need exactly 4 corners (8 numbers). Distances stay in pixels.")

t_run, t_pl, t_ball, t_pose, t_in, t_id, t_db = st.tabs(["Run", "Players & court", "Ball", "Pose & strokes", "Insights", "Identity", "Sessions"])
up = t_run.file_uploader("Padel video", type=["mp4", "mov", "avi", "mkv"])
if up:
    path = tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(up.name).suffix).name
    open(path, "wb").write(up.getvalue())
    ok, f0 = cv2.VideoCapture(path).read()
    if ok:
        fig, ax = plt.subplots(figsize=(6, 3.5)); ax.imshow(cv2.cvtColor(f0, cv2.COLOR_BGR2RGB))
        ax.set_title("First frame: read the pixel coordinates of the 4 court corners"); t_run.pyplot(fig); plt.close(fig)
    if t_run.button("Run analysis"):
        bar = t_run.progress(0.0)
        try:
            from core.pipeline import run_video
            res = run_video(path, pose_w, ball_w, int(ball_cls) if ball_cls.strip() else None, conf, stride, int(max_frames), hand, bar.progress)
            st.session_state["res"] = res
            sid = db.save_session(res, up.name, H is not None)
            t_run.success(f"Session {sid} saved: {res['frames']} frames, {res['tracks']['track_id'].nunique()} tracked players, ball seen in {res['n_ball_obs']} frames.")
        except Exception as e:
            t_run.error(f"Analysis failed: {e}")
res = st.session_state.get("res")
for t in (t_pl, t_ball, t_pose):
    if not res: t.info("Run an analysis first (Run tab).")
if res:
    tr = res["tracks"]
    fig, ax = plt.subplots(figsize=(4, 7) if H is not None else (7, 4))
    for tid, g in tr.groupby("track_id"):
        P = g[["x", "y"]].to_numpy(float); P = court.to_court(H, P) if H is not None else P
        ax.plot(P[:, 0], P[:, 1], label=f"Player {tid}")
    if H is not None: ax.add_patch(plt.Rectangle((0, 0), 10, 20, fill=False, lw=2)); ax.axhline(10, color="gray"); ax.set_xlim(-1, 11); ax.set_ylim(-1, 21)
    else: ax.invert_yaxis()
    ax.set_aspect("equal"); ax.legend(fontsize=7)
    t_pl.pyplot(fig); plt.close(fig)
    t_pl.dataframe(court.metrics(tr, H))
    if H is None: t_pl.warning("No court calibration: distances are in pixels and speeds are unavailable.")
    t_pl.download_button("Tracks CSV", tr.to_csv(index=False), "tracks.csv")
    bl = res["ball"]
    if len(bl):
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(bl.x, bl.y, color="lightgray")
        o = bl[bl.observed.astype(bool)]; ax.scatter(o.x, o.y, s=12, c="#E4572E", label="observed")
        ax.scatter(bl[~bl.observed.astype(bool)].x, bl[~bl.observed.astype(bool)].y, s=12, c="#13A89E", label="Kalman-predicted"); ax.invert_yaxis(); ax.legend()
        t_ball.pyplot(fig); plt.close(fig)
    t_ball.write(f"Ball detected in {res['n_ball_obs']} processed frames; {len(bl)} trajectory points. The default COCO model is weak on padel balls: use custom weights for real work.")
    if res["previews"]:
        i = t_pose.slider("Preview frame", 0, len(res["previews"]) - 1, 0) if len(res["previews"]) > 1 else 0; t_pose.image(res["previews"][i])
    t_pose.dataframe(res["strokes"]); t_pose.caption("Rule-based, unvalidated hypotheses; 'Uncertain' = missing keypoints.")
    t_pose.download_button("Strokes CSV", res["strokes"].to_csv(index=False), "strokes.csv")
t_db.dataframe(db.list_sessions())

with t_in:
    st.subheader("Context (optional, self-reported)")
    c1, c2, c3 = st.columns(3); fat = c1.slider("Fatigue (1-10)", 1, 10, 5); dur = c2.number_input("Session minutes", 0, 300, 60)
    lat, lon = c3.number_input("Latitude", value=36.8), c3.number_input("Longitude", value=10.18)
    wx = context.fetch_weather(lat, lon) if c3.checkbox("Fetch weather (internet)") else None
    ctx = context.session_context(fat, dur, wx); [st.info(n) for n in context.notes(ctx)]
    if not res: st.info("Run an analysis to see events and recommendations.")
    else:
        ev = events.detect_events(res["ball"].to_dict("records"), res["fps_eff"])
        st.subheader("Ball event candidates"); st.dataframe(pd.DataFrame(ev)); st.caption("Single-camera, image-space CANDIDATES with confidence; not verified shots or bounces.")
        st.subheader("Recommendations (demo catalog is SYNTHETIC)")
        if H is None: st.warning("Style needs court calibration (zones and speeds).")
        for _, row in court.metrics(res["tracks"], H).iterrows():
            sty = reco.play_style(row.to_dict()) if H is not None else None
            st.markdown(f"**Player {row.track_id}** - style: {sty or 'unavailable'}")
            if sty:
                for k in ("equipment", "exercise"): st.write(k, [(s, i["name"]) for s, i in reco.recommend(reco.STYLE_TEXT[sty], k)])
            for tip in reco.exercise_tips(res["strokes"][res["strokes"].track_id == row.track_id]): st.info(tip)
        st.caption("Partner matching: reco.match_partners(target, candidates) needs real, consented player profiles (none are invented here).")
with t_id:
    st.warning("Face identification is OFF unless you enrol a player with explicit consent. Embeddings are biometric data: delete them when no longer needed.")
    ok_c = st.checkbox("The player gave explicit consent"); nm = st.text_input("Player name"); im = st.file_uploader("Face photo", type=["jpg", "png"])
    if im and st.button("Enrol / match"):
        e = identity.embed_face(cv2.imdecode(np.frombuffer(im.getvalue(), np.uint8), 1))
        if e is None: st.error("No face found, or insightface is not installed (py -m pip install insightface onnxruntime).")
        else:
            try:
                if nm: identity.enroll(nm, e, ok_c); st.success("Enrolled.")
                st.write("Best match:", identity.match(e))
            except PermissionError as ex: st.error(str(ex))
    st.write("Enrolled:", identity.list_players()); d = st.text_input("Delete player (name)")
    if d and st.button("Delete"): st.write(identity.delete_player(d), "deleted")
