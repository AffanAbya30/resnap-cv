"""Evaluasi akhir + export model ReSnap.

Menghasilkan untuk satu split (val/test):
  - metrics.json   : Precision, Recall, F1, mAP@0.5, mAP@0.75, mAP@0.5:0.95
  - per_class.csv  : metrik per kelas + jumlah instance
  - grafik bawaan Ultralytics: confusion_matrix(_normalized).png, PR_curve.png,
    F1_curve.png, P_curve.png, R_curve.png
  - (opsional) best.onnx
  - (opsional) ablasi: skor ulang dengan foto yang sudah melalui denoise + CLAHE
    (src.preprocessing.enhance), untuk menjawab apakah preprocessing membantu deteksi.

Contoh:
  python scripts/evaluate_export.py --weights models/best.pt --data dataset/data.yaml \
      --split test --out laporan/test_v3 --onnx --ablation
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def evaluate(weights: str, data_yaml: Path, split: str, out: Path, name: str, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights, task="detect")
    seen = {}
    model.add_callback("on_val_end", lambda v: seen.update(images=int(v.seen)))
    m = model.val(data=str(data_yaml), split=split, imgsz=640, batch=8, conf=0.001, iou=0.7,
                  rect=True, plots=True, project=str(out), name=name, exist_ok=True,
                  device=device, verbose=False)
    box = m.box
    names = m.names
    p, r = float(box.mp), float(box.mr)
    summary = {
        "split": split,
        "images": seen.get("images", 0),
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "f1_macro": float(box.f1.mean()) if len(box.f1) else 0.0,
        "mAP50": float(box.map50),
        "mAP75": float(box.map75),
        "mAP50_95": float(box.map),
    }
    nt = getattr(m, "nt_per_class", None)
    rows = []
    for i, cid in enumerate(box.ap_class_index):
        pc, rc = float(box.p[i]), float(box.r[i])
        rows.append({
            "class_id": int(cid), "class_name": names[int(cid)],
            "instances": int(nt[int(cid)]) if nt is not None else "",
            "precision": round(pc, 4), "recall": round(rc, 4),
            "f1": round(2 * pc * rc / (pc + rc), 4) if pc + rc else 0.0,
            "AP50": round(float(box.ap50[i]), 4), "AP50_95": round(float(box.ap[i]), 4),
        })
    run_dir = out / name
    with (run_dir / "per_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def enhanced_copy(data_yaml: Path, split: str, tmp: Path) -> Path:
    """Salin split ke folder sementara dengan foto yang sudah di-enhance; label tidak berubah."""
    import cv2

    from src.preprocessing import enhance

    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base = Path(cfg.get("path") or data_yaml.parent)
    src_images = base / cfg[split]
    if src_images.is_file():
        raise ValueError("Ablasi butuh split berupa folder gambar, bukan daftar .txt")
    src_labels = Path(str(src_images).replace("images", "labels"))
    dst_images, dst_labels = tmp / split / "images", tmp / split / "labels"
    dst_images.mkdir(parents=True)
    shutil.copytree(src_labels, dst_labels)
    n = 0
    for p in sorted(src_images.iterdir()):
        img = cv2.imread(str(p))
        if img is None:
            continue
        cv2.imwrite(str(dst_images / p.name), enhance(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    new_yaml = tmp / "data.yaml"
    new_yaml.write_text(yaml.safe_dump({"path": str(tmp), "train": f"{split}/images", split: f"{split}/images",
                                        "val": f"{split}/images", "names": cfg["names"]}), encoding="utf-8")
    print(f"Ablasi: {n} foto {split} di-enhance (Gaussian+Median+CLAHE)")
    return new_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="0")
    ap.add_argument("--onnx", action="store_true", help="export best.onnx (opset 12, 640)")
    ap.add_argument("--ablation", action="store_true", help="bandingkan foto mentah vs denoise+CLAHE")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = {"raw": evaluate(args.weights, args.data, args.split, args.out, f"{args.split}_raw", args.device)}

    if args.ablation:
        with tempfile.TemporaryDirectory() as tmp:
            yaml_enh = enhanced_copy(args.data, args.split, Path(tmp))
            results["enhanced"] = evaluate(args.weights, yaml_enh, args.split, args.out,
                                           f"{args.split}_enhanced", args.device)
        results["delta_pp"] = {k: round(100 * (results["enhanced"][k] - results["raw"][k]), 2)
                               for k in ("precision", "recall", "f1", "mAP50", "mAP50_95")}

    summary_path = args.out / "ringkasan.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if args.onnx:
        from ultralytics import YOLO

        onnx_path = YOLO(args.weights).export(format="onnx", imgsz=640, opset=12, simplify=True)
        shutil.copy2(onnx_path, args.out / "best.onnx")
        results["onnx"] = str(args.out / "best.onnx")

    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
