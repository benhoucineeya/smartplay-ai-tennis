
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
