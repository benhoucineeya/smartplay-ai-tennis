# Results and limitations
| Level | What |
|---|---|
| Unit-tested (`py -m pytest -q`: 15 passed, 2 skipped because PyTorch is missing in the build environment) | homography, Kalman, speed filter, SQLite, strokes (synthetic), events, TF-IDF/style/partners, identity logic, context validation, evaluation metrics, dataset split + label validation + COCO->YOLO, CF on synthetic ratings |
| Executed with fakes | `run_video` with a scripted fake YOLO (checks tracking dict building, NaN handling, Kalman bridging, stroke windows); Streamlit AppTest rendering (found and fixed one real bug: slider with a single preview frame) |
| Written but NOT executed | `neumf.py`, `tracknet.py` (their tests skip without torch), `train_models.py`, `embed_face` |
| Executed on real YOLO/real padel video | NOTHING |
| Evaluated on labelled padel data | NOTHING: no accuracy, mAP, IDF1, FPS or memory figure exists |
| Not done | RTMPose backend, real dataset download and preparation, any training, runtime benchmarks |
Known risks: generic ball class misses most padel balls; strokes need steady pose and real thresholds; homography needs accurate corner clicks and ground-plane feet; face threshold 0.45 is a placeholder; events are candidates; CF/NeuMF need real interaction data (none exists).
Next steps: (1) install torch+ultralytics on your PC and run a real clip; (2) download PadelTracker100 annotations and a Roboflow ball dataset (`docs/DATASETS_AND_MODELS.md`); (3) fine-tune with `scripts/train_models.py`, evaluate on held-out matches; (4) label strokes and run `scripts/evaluate_models.py strokes`.
