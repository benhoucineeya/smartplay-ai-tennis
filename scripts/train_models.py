"""Fine-tune a YOLO detector (e.g. padel ball). NOT executed in this repo: needs a labelled dataset (data.yaml).
Example: python -m scripts.train_models --data data/ball/data.yaml --base yolov8n.pt --epochs 50 --imgsz 1280
Small objects need a large --imgsz. Keep train/val/test split by VIDEO (see prepare_datasets.split_by_video)."""
import argparse
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--base", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--imgsz", type=int, default=1280); ap.add_argument("--device", default="cpu")
    g = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(g.base); m.train(data=g.data, epochs=g.epochs, imgsz=g.imgsz, device=g.device, project="models", name="finetune")
    print(m.val(data=g.data, split="test"))   # held-out test split must exist in data.yaml
