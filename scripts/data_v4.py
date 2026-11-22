"""Bangun dataset v4: dataset v3 + foto Open Images V7 untuk kelas lemah.

Dipanggil dari notebooks/ReSnap_Retrain_V4.ipynb. Fungsi di sini tidak bergantung
pada FiftyOne: notebook mengubah sampel FiftyOne menjadi dict sederhana

    {"filepath": str, "id": str,
     "detections": [{"label": "Potato", "bbox": [x, y, w, h] (relatif, kiri-atas),
                     "group": bool, "depiction": bool}, ...]}

Langkah penjagaan kualitas:
  1. Hanya label Open Images yang punya padanan pasti dengan kelas ReSnap yang dipakai.
  2. Foto dengan kotak "group of" (satu kotak untuk setumpuk bahan) atau gambar
     ilustrasi/kartun dibuang, karena dataset ReSnap memberi kotak per objek.
  3. Foto yang hampir identik (dHash) dengan foto train/val/test lama dibuang,
     supaya val dan test tidak bocor.
  4. Foto yang menurut model v3 berisi bahan ReSnap lain tanpa label dibuang,
     supaya model tidak diajari bahwa bahan itu adalah latar belakang.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = [
    "tomato", "egg", "onion", "shallot", "garlic", "chili", "carrot", "potato", "cabbage",
    "broccoli", "cauliflower", "cucumber", "eggplant", "mushroom", "spinach", "green_onion",
    "green_beans", "long_beans", "corn", "pumpkin", "lime", "ginger", "turmeric", "galangal",
    "lemongrass", "tofu", "tempeh", "chicken", "fish", "shrimp",
]
# Label Open Images -> id kelas ReSnap. Hanya padanan yang maknanya sama.
# Sengaja tidak dipetakan: "Fish" dan "Chicken" (di Open Images kebanyakan hewan hidup),
# "Lemon" (bukan jeruk nipis), dan label induk seperti "Vegetable"/"Seafood".
OI_TO_RESNAP = {
    "Potato": 7, "Shrimp": 29, "Mushroom": 13, "Tomato": 0, "Carrot": 6, "Broccoli": 9,
    "Cabbage": 8, "Cucumber": 11, "Pumpkin": 19, "Egg (Food)": 1,
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# Hash kemiripan ----------------------------------------------------------------

def dhash(path: str | Path, size: int = 8, mirror: bool = False) -> int | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if mirror:
        img = img[:, ::-1]
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_folder(folder: Path) -> dict[str, int]:
    out = {}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            h = dhash(p)
            if h is not None:
                out[str(p)] = h
    return out


def near_duplicates(candidates: dict[str, int], reference: dict[str, int], max_dist: int = 6) -> dict[str, str]:
    """Kembalikan {kandidat: referensi_terdekat} untuk pasangan dengan jarak Hamming <= max_dist."""
    if not candidates or not reference:
        return {}
    ref_keys = list(reference)
    ref = np.array([reference[k] for k in ref_keys], dtype=np.uint64)
    hits = {}
    for key, h in candidates.items():
        x = np.bitwise_xor(ref, np.uint64(h))
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        i = int(dist.argmin())
        if dist[i] <= max_dist:
            hits[key] = ref_keys[i]
    return hits


# Konversi Open Images -> YOLO ------------------------------------------------------

def convert_sample(sample: dict, target_labels: set[str], min_area: float = 0.0004) -> tuple[list[str] | None, str]:
    """Ubah satu sampel menjadi baris label YOLO. None + alasan bila foto harus dibuang."""
    lines, has_target = [], False
    for det in sample["detections"]:
        label = det["label"]
        if label not in OI_TO_RESNAP:
            continue
        if det.get("depiction"):
            return None, "ilustrasi"
        if det.get("group"):
            return None, "group_of"
        x, y, w, h = det["bbox"]
        x, y = max(0.0, x), max(0.0, y)
        w, h = min(w, 1.0 - x), min(h, 1.0 - y)
        if w * h < min_area:
            continue
        lines.append(f"{OI_TO_RESNAP[label]} {x + w / 2:.6f} {y + h / 2:.6f} {w:.6f} {h:.6f}")
        has_target |= label in target_labels
    if not has_target:
        return None, "tanpa_kelas_target"
    return lines, "ok"


def coverage(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bagian luas kotak a yang tertutup kotak b (IoA), a: (N,4), b: (M,4) xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return inter / (area_a[:, None] + 1e-9)


def yolo_lines_to_xyxy(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not lines:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    arr = np.array([[float(v) for v in ln.split()] for ln in lines])
    cls, cx, cy, w, h = arr.T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), cls.astype(int)


def unlabeled_objects(model, image_path: str, lines: list[str], conf: float = 0.6,
                      min_cover: float = 0.5, ignore: set[int] = frozenset()) -> list[str]:
    """Deteksi v3 yakin (>= conf) yang kurang dari separuh luasnya tertutup kotak label.

    Kelas di `ignore` (kelas yang memang sedang diperbaiki, mis. potato/shrimp/mushroom)
    tidak dihitung, karena v3 justru sering salah pada kelas itu.
    """
    res = model.predict(image_path, conf=conf, imgsz=640, verbose=False)[0]
    if len(res.boxes) == 0:
        return []
    pred = res.boxes.xyxyn.cpu().numpy()
    pcls = res.boxes.cls.cpu().numpy().astype(int)
    gt, _ = yolo_lines_to_xyxy(lines)
    covered = coverage(pred, gt).max(axis=1) if len(gt) else np.zeros(len(pred))
    return sorted({CLASS_NAMES[c] for c, o in zip(pcls, covered) if o < min_cover and c not in ignore})


def write_pair(image_src: Path, lines: list[str], images_dir: Path, labels_dir: Path, stem: str) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_src))
    if img is None:
        raise ValueError(f"Gagal membaca {image_src}")
    cv2.imwrite(str(images_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_oi_split(samples: list[dict], target_labels: set[str], out_dir: Path, prefix: str,
                   reference_hashes: dict[str, int], model=None, ignore_cls: set[int] = frozenset(),
                   seen_hashes: dict[str, int] | None = None) -> list[dict]:
    """Saring + tulis satu kumpulan sampel Open Images. Mengembalikan log per foto."""
    log, kept_hashes = [], dict(seen_hashes or {})
    for s in samples:
        row = {"oi_id": s["id"], "file": Path(s["filepath"]).name, "status": "", "detail": ""}
        lines, reason = convert_sample(s, target_labels)
        if lines is None:
            row["status"] = reason
            log.append(row)
            continue
        h = dhash(s["filepath"])
        if h is None:
            row["status"] = "tidak_terbaca"
            log.append(row)
            continue
        pool = {**reference_hashes, **kept_hashes}
        dup = near_duplicates({s["filepath"]: h}, pool) or near_duplicates(
            {s["filepath"]: dhash(s["filepath"], mirror=True)}, pool)
        if dup:
            row.update(status="mirip_foto_lain", detail=Path(next(iter(dup.values()))).name)
            log.append(row)
            continue
        if model is not None:
            missing = unlabeled_objects(model, s["filepath"], lines, ignore=ignore_cls)
            if missing:
                row.update(status="bahan_lain_tanpa_label", detail=";".join(missing))
                log.append(row)
                continue
        stem = f"{prefix}_{s['id']}"
        write_pair(Path(s["filepath"]), lines, out_dir / "images", out_dir / "labels", stem)
        kept_hashes[s["filepath"]] = h
        row.update(status="dipakai", detail=f"{len(lines)} kotak")
        log.append(row)
    return log


# Penggabungan + oversampling -------------------------------------------------------

def label_classes(label_path: Path) -> set[int]:
    text = label_path.read_text(encoding="utf-8-sig") if label_path.is_file() else ""
    return {int(ln.split()[0]) for ln in text.splitlines() if ln.strip()}


def oversample(train_dir: Path, class_ids: set[int], copies: int, prefix_skip: str = "oi_") -> int:
    """Salin foto train LAMA yang memuat kelas lemah sebanyak `copies` kali tambahan.

    Foto Open Images tidak diduplikasi (sudah beragam). Duplikat diberi akhiran _dupN.
    Dibatasi kecil (2x) karena duplikat berlebihan memicu overfitting pada foto itu.
    """
    images, labels = train_dir / "images", train_dir / "labels"
    added = 0
    for img in sorted(images.iterdir()):
        if img.stem.startswith(prefix_skip) or "_dup" in img.stem:
            continue
        lbl = labels / f"{img.stem}.txt"
        if not (label_classes(lbl) & class_ids):
            continue
        for k in range(1, copies + 1):
            shutil.copy2(img, images / f"{img.stem}_dup{k}{img.suffix}")
            shutil.copy2(lbl, labels / f"{img.stem}_dup{k}.txt")
            added += 1
    return added


def count_split(split_dir: Path) -> dict:
    imgs = [p for p in (split_dir / "images").iterdir() if p.suffix.lower() in IMG_EXT]
    inst, img_per_cls = Counter(), Counter()
    for p in imgs:
        lbl = split_dir / "labels" / f"{p.stem}.txt"
        if not lbl.is_file():
            continue
        cls = [int(ln.split()[0]) for ln in lbl.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
        inst.update(cls)
        img_per_cls.update(set(cls))
    return {"images": len(imgs),
            "per_class": {CLASS_NAMES[c]: {"images": img_per_cls[c], "instances": inst[c]}
                          for c in range(len(CLASS_NAMES))}}


def write_yaml(path: Path, root: Path, train: str, val: str, extra: dict | None = None) -> Path:
    import yaml

    cfg = {"path": str(root), "train": train, "val": val, "names": dict(enumerate(CLASS_NAMES))}
    cfg.update(extra or {})
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def write_log(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
