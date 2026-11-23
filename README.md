# SmartPlay AI (rebuild)
Install (Python 3.10-3.12 recommended for PyTorch): `py -m pip install -r requirements.txt`
Run: `py -m streamlit run app.py`   Tests: `py -m pytest -q` (15 pass, 2 skip without PyTorch)
Flow: upload video -> optional 4 court corners -> Run. Tabs: Run, Players & court, Ball, Pose & strokes, Insights (events, recommendations, context), Identity (opt-in, consent), Sessions.
Optional: `py -m pip install insightface onnxruntime` for face embeddings. Fine-tuning/evaluation: see `scripts/` and `docs/`.
Honest status: unit tests, a fake-model run of the video pipeline and a dashboard render test were executed. Real YOLO on real footage, PyTorch models (NeuMF, TrackNet-lite), training and any accuracy figure were NOT. Datasets found (PadelTracker100, Roboflow) but not downloaded: see `docs/DATASETS_AND_MODELS.md`. Read `docs/RESULTS_AND_LIMITATIONS.md` first.
