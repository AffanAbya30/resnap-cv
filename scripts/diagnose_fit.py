"""Diagnosa overfitting / underfitting dari satu run YOLOv8.

Tiga sumber bukti:
  1. Kurva results.csv: apakah mAP val masih naik di akhir (belum konvergen),
     apakah loss val naik lagi sementara loss train terus turun (overfitting).
  2. Gap generalisasi: mAP50-95 pada subset train (tanpa augmentasi) dikurangi
     mAP50-95 val. Dibandingkan dengan gap model acuan (v3), bukan angka mutlak,
     karena detektor wajar punya gap.
  3. Skor pada data yang tidak pernah dilihat (holdout Open Images) per kelas lemah.

Contoh:
  python scripts/diagnose_fit.py --results runs/x/results.csv --max-epochs 100 \
      --train-map 0.61 --val-map 0.47 --ref-gap 0.12 --plot kurva.png
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Ambang keputusan (ditulis eksplisit supaya bisa dikutip di laporan).
SLOPE_STILL_IMPROVING = 0.001   # kenaikan mAP50-95 val per epoch pada 10 epoch terakhir
VAL_LOSS_RISE = 0.10            # loss val naik >10% dari titik terendahnya
BEST_EPOCH_MARGIN = 10          # epoch terbaik jauh sebelum akhir
GAP_EXTRA_OVERFIT = 0.05        # gap lebih besar 5 poin dari gap model acuan
LOW_TRAIN_MAP = 0.55            # mAP50-95 subset train yang terlalu rendah = kapasitas kurang
SMALL_GAP = 0.03


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [{k.strip(): float(v) for k, v in r.items() if v not in (None, "")} for r in csv.DictReader(f)]
    if not rows:
        raise ValueError(f"{path} kosong")
    return rows


def _slope(ys: list[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xm, ym = (n - 1) / 2, sum(ys) / n
    num = sum((i - xm) * (y - ym) for i, y in enumerate(ys))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den


def analyze_curves(rows: list[dict], max_epochs: int) -> dict:
    epochs = [int(r["epoch"]) for r in rows]
    m50 = [r["metrics/mAP50(B)"] for r in rows]
    m5095 = [r["metrics/mAP50-95(B)"] for r in rows]
    fitness = [0.1 * a + 0.9 * b for a, b in zip(m50, m5095)]  # sama dengan pemilih best.pt Ultralytics
    val_loss = [r["val/box_loss"] + r["val/cls_loss"] + r["val/dfl_loss"] for r in rows]
    train_loss = [r["train/box_loss"] + r["train/cls_loss"] + r["train/dfl_loss"] for r in rows]

    best_i = max(range(len(rows)), key=fitness.__getitem__)
    min_vl_i = min(range(len(rows)), key=val_loss.__getitem__)
    tail = slice(max(0, len(rows) - 10), len(rows))
    return {
        "epochs_run": epochs[-1],
        "max_epochs": max_epochs,
        "early_stopped": epochs[-1] < max_epochs,
        "best_epoch": epochs[best_i],
        "best_mAP50": round(m50[best_i], 4),
        "best_mAP50_95": round(m5095[best_i], 4),
        "val_map_slope_last10": round(_slope(m5095[tail]), 5),
        "train_loss_slope_last10": round(_slope(train_loss[tail]), 5),
        "val_loss_min_epoch": epochs[min_vl_i],
        "val_loss_rise_from_min": round(val_loss[-1] / val_loss[min_vl_i] - 1, 4),
    }


def verdict(curves: dict, train_map: float | None = None, val_map: float | None = None,
            ref_gap: float | None = None) -> dict:
    findings, under, over = [], False, False

    if not curves["early_stopped"] and curves["val_map_slope_last10"] > SLOPE_STILL_IMPROVING:
        under = True
        findings.append(f"mAP50-95 val masih naik {curves['val_map_slope_last10']:.4f}/epoch di 10 epoch "
                        "terakhir dan training berhenti karena batas epoch: belum konvergen. "
                        "Lanjutkan training (resume) dengan epoch lebih banyak.")
    else:
        findings.append("mAP val sudah mendatar di akhir training (konvergen).")

    if (curves["val_loss_rise_from_min"] > VAL_LOSS_RISE and curves["train_loss_slope_last10"] < 0
            and curves["epochs_run"] - curves["val_loss_min_epoch"] >= BEST_EPOCH_MARGIN):
        over = True
        findings.append(f"Loss val naik {curves['val_loss_rise_from_min']:.0%} dari titik terendah (epoch "
                        f"{curves['val_loss_min_epoch']}) sementara loss train terus turun: tanda overfitting. "
                        f"best.pt tetap diambil dari epoch {curves['best_epoch']}, sebelum gejala itu.")
    else:
        findings.append(f"Loss val tidak naik berarti (naik {curves['val_loss_rise_from_min']:.0%} dari minimum).")

    gap = None
    if train_map is not None and val_map is not None:
        gap = train_map - val_map
        limit = (ref_gap if ref_gap is not None else 0.15) + GAP_EXTRA_OVERFIT
        if gap > limit:
            over = True
            findings.append(f"Gap mAP50-95 train-val {gap:.3f} melebihi batas {limit:.3f}"
                            + (f" (gap acuan {ref_gap:.3f})" if ref_gap is not None else "") + ": overfitting.")
        else:
            findings.append(f"Gap mAP50-95 train-val {gap:.3f} dalam batas {limit:.3f}.")
        if train_map < LOW_TRAIN_MAP and gap < SMALL_GAP:
            under = True
            findings.append(f"mAP50-95 di data train sendiri hanya {train_map:.3f} dan hampir sama dengan val: "
                            "model kurang kapasitas (underfit). Pertimbangkan model lebih besar.")

    label = ("overfit dan belum konvergen" if over and under else
             "overfit" if over else "underfit" if under else "seimbang")
    return {"vonis": label, "gap_train_val": None if gap is None else round(gap, 4), "temuan": findings}


def plot(rows: list[dict], out: Path, best_epoch: int | None = None, title: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, part in zip(axes.flat[:3], ("box", "cls", "dfl")):
        ax.plot(ep, [r[f"train/{part}_loss"] for r in rows], label="train")
        ax.plot(ep, [r[f"val/{part}_loss"] for r in rows], label="val")
        ax.set_title(f"{part} loss")
        ax.legend()
    ax = axes.flat[3]
    ax.plot(ep, [r["metrics/mAP50(B)"] for r in rows], label="mAP50 val")
    ax.plot(ep, [r["metrics/mAP50-95(B)"] for r in rows], label="mAP50-95 val")
    ax.set_title("metrik val")
    ax.legend()
    for a in axes.flat:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        if best_epoch:
            a.axvline(best_epoch, color="gray", ls="--", lw=1)
    fig.suptitle(title or "Kurva training (garis putus-putus = epoch best.pt)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--max-epochs", required=True, type=int)
    ap.add_argument("--train-map", type=float)
    ap.add_argument("--val-map", type=float)
    ap.add_argument("--ref-gap", type=float)
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args()

    rows = load_results(args.results)
    curves = analyze_curves(rows, args.max_epochs)
    out = {"kurva": curves, **verdict(curves, args.train_map, args.val_map, args.ref_gap)}
    if args.plot:
        plot(rows, args.plot, curves["best_epoch"])
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
