# Implementation report
Sources: the thesis (RAIA 02.26) and the improvements from our slides. The Slidesgo deck was never provided.
| Requirement | Source | Approach | Files | Verification |
|---|---|---|---|---|
| Player detection + ByteTrack | thesis | YOLOv8-pose `.track(bytetrack)` | `pipeline.py` | ran end-to-end with FAKE models (`test_pipeline_mock`); never with real YOLO/video |
| Court homography, distance, speed, zones | thesis | 4-corner DLT, speed cap 12 m/s | `court.py` | unit-tested |
| Ball detection | thesis + slides | generic COCO class (weak); TrackNet-lite model | `pipeline.py`, `tracknet.py` | pipeline plumbing mock-tested; TrackNet-lite NOT executed (no torch) |
| Kalman trajectory | slides | constant-velocity KF, gating, reset | `ball.py` | unit-tested + mock pipeline |
| Bounce/shot events | slides | direction-change candidates | `events.py` | unit-tested (synthetic path) |
| Pose, angles, 4 strokes | slides | rules over windows + Uncertain | `pose_utils.py` | unit-tested on SYNTHETIC skeletons; mock pipeline |
| Identity | thesis | embeddings, cosine, consent, delete, vote | `identity.py` | unit-tested with random vectors |
| SQLite | thesis | schema with FKs | `db.py` | unit-tested |
| Style rules, TF-IDF | thesis | thesis rules; smoothed TF-IDF on a SYNTHETIC catalog | `reco.py` | unit-tested |
| Collaborative filtering (user-based + SVD) | thesis | numpy hybrid alpha=0.5, cold start, leave-one-out HR@k | `cf.py` | tested on SYNTHETIC ratings only |
| NeuMF | thesis | PyTorch GMF+MLP, refuses < 200 interactions | `neumf.py` | NOT executed (no torch) |
| RTMPose backend | slides | - | - | NOT done |
| Partner matching | thesis | heuristic | `reco.py` | unit-tested, no real profiles |
| Weather / fatigue | slides | optional inputs | `context.py` | partly tested (offline) |
| Dashboard | thesis | Streamlit, 7 tabs | `app.py` | rendered with Streamlit AppTest (empty, injected results, calibrated); never used in a browser |
| Datasets, training, evaluation | prompt | scripts + metrics code | `scripts/`, `evaluation.py` | utilities unit-tested; no dataset downloaded, nothing trained |
