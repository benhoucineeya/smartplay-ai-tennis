"""Prints the manual steps: automatic download is NOT possible for these sources (licences, API keys, 7 GB files)."""
STEPS = """
1) PadelTracker100 (CC BY 4.0): https://doi.org/10.5281/zenodo.17020011  (Zenodo, ~7.1 GB, ball/player/pose/shot annotations).
   The paper states the source videos are NOT included (WPT 2022 Finals, from YouTube): check the record contents and the
   broadcaster's rights before obtaining any footage. Annotations use COCO format -> scripts.prepare_datasets.coco_to_yolo.
2) Roboflow Universe padel ball datasets (shown as CC BY 4.0 on their pages; check each page and cite it):
   https://universe.roboflow.com/padel-rbqpp/padel-detection-5 | /awecesar/padel-ball-detection-i1vvv | /najla02/padel_vv
   Create a free account, choose Download Dataset -> YOLOv8 format, unzip into datasets/<name>/.
3) Then: python -m scripts.prepare_datasets validate datasets/<name>/train/labels --classes N
   and split BY VIDEO / match (never by random frames), then scripts/train_models.py.
"""
if __name__ == "__main__": print(STEPS)
