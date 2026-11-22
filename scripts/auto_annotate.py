"""Pra-anotasi bounding box otomatis (semi-automated pre-annotation).

Menghasilkan label awal format YOLO (.txt) untuk foto baru, lalu foto + label
diunggah ke Roboflow untuk dicek manual (human-in-the-loop).

Dua mode:
  --mode model  : pakai model ReSnap sendiri (best.pt). Paling akurat untuk 30 kelas
                  yang sudah ada (model-assisted labeling).
  --mode world  : pakai YOLO-World (open-vocabulary, prompt teks per kelas).
                  Dipakai untuk kelas yang belum dikenali model, mis. menambah data udang.

Contoh:
  python scripts/auto_annotate.py --images foto_baru/ --out prelabel/ --mode model
  python scripts/auto_annotate.py --images foto_udang/ --out prelabel_udang/ --mode world --classes shrimp
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detector import DEFAULT_WEIGHTS  # noqa: E402

# Urutan kelas WAJIB sama dengan data.yaml training (v2/v3).
CLASS_NAMES = [
    "tomato", "egg", "onion", "shallot", "garlic", "chili", "carrot", "potato", "cabbage",
    "broccoli", "cauliflower", "cucumber", "eggplant", "mushroom", "spinach", "green_onion",
    "green_beans", "long_beans", "corn", "pumpkin", "lime", "ginger", "turmeric", "galangal",
    "lemongrass", "tofu", "tempeh", "chicken", "fish", "shrimp",
]
# Prompt teks untuk YOLO-World (lebih deskriptif = lebih akurat).
WORLD_PROMPTS = {
    "tomato": "tomato", "egg": "egg", "onion": "onion", "shallot": "shallot",
    "garlic": "garlic bulb", "chili": "chili pepper", "carrot": "carrot", "potato": "potato",
    "cabbage": "cabbage", "broccoli": "broccoli", "cauliflower": "cauliflower",
    "cucumber": "cucumber", "eggplant": "eggplant", "mushroom": "mushroom", "spinach": "spinach",
    "green_onion": "green onion", "green_beans": "green beans", "long_beans": "long beans",
    "corn": "corn cob", "pumpkin": "pumpkin", "lime": "lime", "ginger": "ginger root",
    "turmeric": "turmeric root", "galangal": "galangal root", "lemongrass": "lemongrass stalk",
    "tofu": "tofu block", "tempeh": "tempeh", "chicken": "raw chicken meat",
    "fish": "raw fish", "shrimp": "raw shrimp",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_model(mode: str, weights: str | None, classes: list[str]):
    from ultralytics import YOLO

    if mode == "model":
        model = YOLO(weights or str(DEFAULT_WEIGHTS), task="detect")
        names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        if [names[i] for i in range(len(names))] != CLASS_NAMES:
            raise ValueError("Urutan kelas model berbeda dari CLASS_NAMES.")
        keep = {CLASS_NAMES.index(c) for c in classes}
        return model, lambda cid: cid if cid in keep else None
    model = YOLO(weights or "yolov8s-worldv2.pt")
    model.set_classes([WORLD_PROMPTS[c] for c in classes])
    # indeks YOLO-World mengikuti urutan `classes` -> petakan ke id dataset
    return model, lambda cid: CLASS_NAMES.index(classes[cid])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mode", choices=["model", "world"], default="model")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--classes", nargs="*", default=CLASS_NAMES, help="batasi kelas yang dilabeli")
    ap.add_argument("--conf", type=float, default=0.30, help="rendah agar objek jarang tidak terlewat")
    ap.add_argument("--review-below", type=float, default=0.60,
                    help="foto dengan kotak di bawah nilai ini ditandai wajib cek")
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    unknown = set(args.classes) - set(CLASS_NAMES)
    if unknown:
        ap.error(f"Kelas tidak dikenal: {sorted(unknown)}")
    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMG_EXT)
    if not images:
        ap.error(f"Tidak ada gambar di {args.images}")

    model, to_dataset_id = load_model(args.mode, args.weights, args.classes)
    img_dir, lbl_dir = args.out / "images", args.out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(0, len(images), 16):
        batch = images[i:i + 16]
        results = model.predict([str(p) for p in batch], conf=args.conf, iou=args.iou,
                                imgsz=640, device=args.device, verbose=False)
        for path, res in zip(batch, results):
            lines, confs, labels = [], [], []
            for (cx, cy, w, h), c, cid in zip(res.boxes.xywhn.tolist(), res.boxes.conf.tolist(),
                                              res.boxes.cls.int().tolist()):
                did = to_dataset_id(cid)
                if did is None:
                    continue
                lines.append(f"{did} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                confs.append(c)
                labels.append(CLASS_NAMES[did])
            stem = path.stem
            shutil.copy2(path, img_dir / path.name)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            status = "kosong" if not lines else ("cek" if min(confs) < args.review_below else "ok")
            rows.append({
                "gambar": path.name, "jumlah_kotak": len(lines),
                "kelas": ";".join(sorted(set(labels))),
                "conf_min": round(min(confs), 3) if confs else "",
                "status_review": status,
            })

    (args.out / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    (args.out / "data.yaml").write_text(
        "train: images\nval: images\nnc: 30\nnames: [" + ", ".join(CLASS_NAMES) + "]\n", encoding="utf-8")
    with (args.out / "review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = {s: sum(r["status_review"] == s for r in rows) for s in ("ok", "cek", "kosong")}
    print(f"{len(rows)} foto dilabeli -> {args.out}")
    print(f"ok: {counts['ok']}, wajib cek: {counts['cek']}, tanpa kotak: {counts['kosong']}")
    print("Unggah folder ini ke Roboflow (format YOLOv8), lalu koreksi foto berstatus 'cek' dan 'kosong'.")


if __name__ == "__main__":
    main()
