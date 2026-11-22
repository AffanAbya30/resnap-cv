"""Training YOLOv8 ReSnap (transfer learning dari bobot pralatih COCO).

Setelan default = setelan run v3 (resnap_yolov8n_30kelas_v3_udang_b1) supaya
hasil bisa direproduksi. Contoh:

  python scripts/train_yolo.py --data dataset/data.yaml --name resnap_v4
  python scripts/train_yolo.py --data dataset/data.yaml --name resnap_v4 --resume
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RECIPE = dict(
    epochs=50, patience=15, batch=8, imgsz=640, save=True, save_period=5, cache=False,
    workers=2, pretrained=True, optimizer="auto", seed=42, deterministic=True,
    single_cls=False, rect=False, cos_lr=False, close_mosaic=10, amp=True, fraction=1.0,
    multi_scale=False, val=True, plots=True, lr0=0.01, lrf=0.01, momentum=0.937,
    weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    box=7.5, cls=0.5, dfl=1.5, nbs=64,
    # augmentasi (setara Roboflow: flip, rotasi, HSV shift, mosaic; blur dilakukan saat export dataset)
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, degrees=10.0, translate=0.1, scale=0.4,
    shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0,
    mosaic=0.5, mixup=0.0, cutmix=0.0, copy_paste=0.0,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="data.yaml (train/val/names)")
    ap.add_argument("--model", default="yolov8n.pt", help="bobot awal, default pralatih COCO")
    ap.add_argument("--project", default="runs/resnap")
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--resume", action="store_true", help="lanjutkan dari weights/last.pt")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                    help="override setelan, mis. --set epochs=80 batch=16")
    args = ap.parse_args()

    from ultralytics import YOLO, __version__

    run_dir = Path(args.project) / args.name
    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.is_file():
            ap.error(f"Tidak ada {last} untuk dilanjutkan.")
        YOLO(str(last)).train(resume=True, device=args.device)
    else:
        if run_dir.exists():
            ap.error(f"{run_dir} sudah ada. Pakai --name lain atau --resume.")
        recipe = dict(RECIPE)
        for item in args.set:
            k, v = item.split("=", 1)
            if k not in recipe:
                ap.error(f"Setelan tidak dikenal: {k}")
            recipe[k] = type(recipe[k])(v) if not isinstance(recipe[k], bool) else v.lower() == "true"
        run_dir.mkdir(parents=True)
        (run_dir / "experiment.json").write_text(json.dumps(dict(
            data=str(args.data.resolve()), base_model=args.model, ultralytics=__version__, settings=recipe,
        ), indent=2), encoding="utf-8")
        YOLO(args.model, task="detect").train(
            data=str(args.data), project=args.project, name=args.name, exist_ok=True,
            device=args.device, **recipe)

    best = run_dir / "weights" / "best.pt"
    if best.is_file():
        (run_dir / "training_complete.json").write_text(
            json.dumps({"weights_sha256": sha256(best)}, indent=2), encoding="utf-8")
        print("Selesai:", best)


if __name__ == "__main__":
    main()
