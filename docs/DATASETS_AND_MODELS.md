# Datasets and models
**Status key:** found = seen in a web search on 2026-10-16 (search snippets only, pages not opened by me, nothing downloaded: Zenodo/Roboflow are not reachable from the build environment). Re-check every licence on the official page before use.
## Padel data (researched)
| Resource | Source | Licence shown | Content | Use here | Status |
|---|---|---|---|---|---|
| PadelTracker100 (Bada-Nerin et al., NTT Data / Univ. Oviedo, *Data in Brief* vol. 65) | https://doi.org/10.5281/zenodo.17020011 (v2, 2025-09-01; v1: 10.5281/zenodo.14653706) | CC BY 4.0 | ~100k frames from two 2022 World Padel Tour Finals matches (1920x1080, 30 fps, single main camera); annotations: ball box, players, pose keypoints, shot events; COCO-format JSON; record is ~7.1 GB | ball fine-tuning, pose and shot-event EVALUATION | found, not downloaded. The article says the source videos (from YouTube) are NOT included: check rights before obtaining footage |
| Roboflow `padel-detection-5` / `-4` / `-2` (workspace padel-rbqpp) | https://universe.roboflow.com/padel-rbqpp/padel-detection-5 | CC BY 4.0 | ball (and a second class) detection | ball detector fine-tuning | found, not downloaded |
| Roboflow `padel-ball-detection-i1vvv` (awecesar) | https://universe.roboflow.com/awecesar/padel-ball-detection-i1vvv | CC BY 4.0 | 21,071 images (version 3) | ball detector fine-tuning | found, not downloaded |
| Roboflow `padel_vv` (najla02) | https://universe.roboflow.com/najla02/padel_vv | CC BY 4.0 | 13 classes incl. ball, player, racket, net, glass wall, court | court/equipment detection | found, not downloaded |
| Roboflow `tfm-padel-training` (tfm-azkdo) | https://universe.roboflow.com/tfm-azkdo/tfm-padel-training | not checked | 481 images, ball/player/net/racket, keypoint model | small experiments only | found, not downloaded |
Caveats: Roboflow datasets need an account/API key and each page must be checked and cited; image counts after augmentation can inflate sizes; PadelTracker100 covers only two professional matches from one camera (domain gap vs club cameras). Splits must be by match/video. Steps: `python -m scripts.download_datasets`; COCO -> YOLO with `scripts.prepare_datasets.coco_to_yolo`.
## Models and libraries
| Resource | Source | Licence (verify) | Role | Status |
|---|---|---|---|---|
| YOLOv8 / YOLOv8-pose | https://docs.ultralytics.com | AGPL-3.0 (commercial licence exists) | players, pose, generic ball class | used in code; weights not downloaded here |
| COCO | https://cocodataset.org | annotations CC BY 4.0 | pretraining of the weights above | not used directly |
| ByteTrack | https://github.com/ifzhang/ByteTrack (Ultralytics ships its own) | MIT / AGPL | player tracking | used in code, never run on real video |
| RTMPose (MMPose) | https://github.com/open-mmlab/mmpose | Apache-2.0 | alternative pose backend | NOT integrated |
| InsightFace | https://github.com/deepinsight/insightface | code MIT; pretrained models non-commercial | `identity.embed_face` | optional, not run |
| TrackNet family | search the TrackNetV2/V3 repositories, read each licence | varies | reference for `core/tracknet.py` (own small implementation) | pretrained weights NOT used |
| Open-Meteo | https://open-meteo.com | free non-commercial use | optional weather | untested offline |
