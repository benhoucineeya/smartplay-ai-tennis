"""Dataset utilities. Usage: python -m scripts.prepare_datasets frames|split|validate ..."""
import argparse, random, pathlib, collections, json, json

def extract_frames(video, out_dir, every=5):
    import cv2
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True); cap = cv2.VideoCapture(str(video)); i = n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % every == 0: cv2.imwrite(str(out / f"{pathlib.Path(video).stem}_{i:06d}.jpg"), f); n += 1
        i += 1
    return n

def split_by_video(videos, seed=0, ratios=(.7, .15, .15)):
    """Whole videos go to one split only (no leakage between adjacent frames)."""
    v = sorted(videos); random.Random(seed).shuffle(v); n = len(v)
    if n < 3: return dict(train=v, val=[], test=[])
    a = max(1, min(int(n * ratios[0]), n - 2)); b = max(a + 1, min(int(n * (ratios[0] + ratios[1])), n - 1))
    return dict(train=v[:a], val=v[a:b], test=v[b:])

def coco_to_yolo(coco_json, out_dir, keep=None):
    """Standard COCO detection JSON (images, annotations[bbox=x,y,w,h], categories) -> YOLO txt labels. Returns class names in id order."""
    d = json.load(open(coco_json)); out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cats = {c["id"]: c["name"] for c in d["categories"]}; names = [n for n in cats.values() if keep is None or n in keep]
    cid = {n: i for i, n in enumerate(names)}; imgs = {i["id"]: i for i in d["images"]}; lines = {}
    for a in d["annotations"]:
        n = cats.get(a["category_id"]); x, y, w, h = a["bbox"]
        if n not in cid or w <= 0 or h <= 0: continue
        im = imgs[a["image_id"]]
        lines.setdefault(pathlib.Path(im["file_name"]).stem, []).append(
            f'{cid[n]} {(x + w / 2) / im["width"]:.6f} {(y + h / 2) / im["height"]:.6f} {w / im["width"]:.6f} {h / im["height"]:.6f}')
    for stem, l in lines.items(): (out / f"{stem}.txt").write_text("\n".join(l) + "\n")
    return names

def validate_yolo_labels(label_dir, n_classes):
    problems, counts = [], collections.Counter()
    for f in pathlib.Path(label_dir).glob("*.txt"):
        for ln, line in enumerate(f.read_text().splitlines(), 1):
            p = line.split()
            if len(p) != 5: problems.append(f"{f.name}:{ln} expected 5 fields"); continue
            try: c, x, y, w, h = int(p[0]), *map(float, p[1:])
            except ValueError: problems.append(f"{f.name}:{ln} non-numeric"); continue
            if not 0 <= c < n_classes: problems.append(f"{f.name}:{ln} bad class {c}")
            elif not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1): problems.append(f"{f.name}:{ln} box out of range")
            else: counts[c] += 1
    return problems, counts

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("frames"); a.add_argument("video"); a.add_argument("out"); a.add_argument("--every", type=int, default=5)
    b = sp.add_parser("split"); b.add_argument("videos", nargs="+")
    c = sp.add_parser("validate"); c.add_argument("labels"); c.add_argument("--classes", type=int, required=True)
    g = ap.parse_args()
    if g.cmd == "frames": print(extract_frames(g.video, g.out, g.every), "frames written")
    elif g.cmd == "split": print(json.dumps(split_by_video(g.videos), indent=1))
    else: pr, ct = validate_yolo_labels(g.labels, g.classes); print("problems:", pr[:20], "| class counts:", dict(ct))
