# ReSnap — Recipe from a Snap

Foto bahan makanan -> pra-pemrosesan OpenCV -> deteksi YOLOv8 (30 kelas) -> bahan bisa dikoreksi -> Top-5 resep dari Recipe API.

## Menjalankan

```bash
pip install -r requirements.txt
# opsional, kalau pakai Spoonacular; tanpa key otomatis pakai TheMealDB
export SPOONACULAR_API_KEY=xxxx
streamlit run app.py
```

Model `models/best.pt` = **v4b** (YOLOv8s, `resnap_yolov8s_30kelas_v4b_reg`, sha256 `1410f11d…6809`). Model lama tersimpan sebagai `models/best_v3.pt`.

## Struktur

| File | Modul PRD | Pemilik |
|---|---|---|
| `src/preprocessing.py` | 3.1 Resize/HSV, Gaussian+Median, CLAHE LAB, Otsu, morfologi+kontur, letterbox+normalisasi | Sulthan |
| `app.py` | 3.4 UI Streamlit (kamera/unggah, bbox, tag bahan, kartu resep) | Sulthan |
| `src/detector.py` | 3.2 Inference YOLOv8, NMS, filter confidence, himpunan bahan B | Affan |
| `scripts/auto_annotate.py` | 3.2 Pra-anotasi otomatis (model sendiri / YOLO-World) -> Roboflow | Affan |
| `scripts/train_yolo.py` | 3.2 Training transfer learning (setelan v3) | Affan |
| `scripts/evaluate_export.py` | 3.2 Evaluasi P/R/F1/mAP, confusion matrix, PR curve, ablasi, ONNX | Affan |
| `notebooks/ReSnap_Evaluasi_Final_Test.ipynb` | Evaluasi test set final di Colab | Affan |
| `src/recipe_local.py` + `data/recipes.json.gz` | 3.3 Pencocokan resep lokal (tanpa internet/LLM) | Affan |
| `scripts/build_recipe_db.py` | Membangun database resep dari dataset Cookpad | Affan |
| `src/recipe_fallback.py` | Cadangan online (Spoonacular/TheMealDB) | cadangan |

## Resep lokal (versi murni, tanpa API/LLM)

`app.py` memakai database lokal `data/recipes.json.gz`; API online hanya dipakai bila file itu tidak ada. (Hook `src/recipe_service.py` tetap ada bila suatu saat modul resep lain ingin dipasang.)

- Data: Kaggle *Indonesian Food Recipes* (resep Cookpad, 2018), 14.945 resep dalam 8 kategori (ayam, ikan, kambing, sapi, tahu, telur, tempe, udang). Setelah dibersihkan: 14.658 resep.
- Setiap bahan dipetakan ke 30 kelas deteksi (mis. "cabe rawit" -> cabai, "laos" -> lengkuas, "telur ayam" -> telur), bumbu dasar (garam, gula, minyak, air, lada, kecap, kemiri, ketumbar, daun salam, dll.) dianggap selalu ada, dan bahan yang tidak bisa dideteksi (daging sapi, santan, tepung) tetap dihitung sebagai bahan kurang.
- Skor PRD `|B ∩ R| / |R|`; seri -> bahan kurang lebih sedikit, lalu bahan cocok lebih banyak, lalu resep terpopuler. Resep yang memakai minimal 2 bahan terdeteksi ditampilkan lebih dulu; judul kembar dan resep berbahan utama tunggal disaring.
- Bangun ulang: `python scripts/build_recipe_db.py --csv Indonesian_Food_Recipes.csv`. Uji: `python -m pytest tests/ -q`.

## Kontrak modul resep

Semua modul resep (`recipe_local`, `recipe_fallback`) mengikuti kontrak yang sama:

```python
def recommend(ingredients: list[str], top_k: int = 5) -> list[dict]:
    # ingredients: nama kelas YOLO, mis. ["egg", "tomato", "green_onion"]
    # return per resep:
    # {"judul": str, "gambar": url|None, "skor": float 0-1,
    #  "bahan_tersedia": [str], "bahan_kurang": [str], "langkah": [str], "sumber": url}
```

Pembagian: Sulthan (preprocessing + UI + integrasi), Affan (dataset, YOLOv8, pencocokan resep lokal), Ezekiel (laporan dan slide).

## Catatan desain preprocessing

- Resize mempertahankan rasio aspek lalu letterbox 640x640 (sama seperti saat training). Resize paksa 640x640 akan membuat bahan gepeng.
- Input YOLO berupa tensor RGB float [0, 1] hasil `normalize()`, dan koordinat kotak dipetakan balik ke foto asli lewat `LetterboxInfo.to_original()`.
- Model dilatih dengan foto mentah, jadi denoise + CLAHE untuk deteksi default-nya mati (toggle di sidebar). Pengaruhnya diukur di notebook evaluasi final (ablasi).
- Mask Otsu + kontur ditampilkan sebagai analisis kandidat area bahan; tidak dipakai untuk memotong input YOLO.

## Hasil model

Val (1.069 foto, sama untuk semua versi):

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Gap train-val* |
|---|---|---|---|---|---|
| v3 (YOLOv8n) | 0.704 | 0.605 | 0.648 | 0.426 | 0.218 |
| v4 (YOLOv8s + Open Images) | 0.713 | 0.651 | 0.687 | 0.480 | 0.355 |
| **v4b** (v4 + regularisasi) | 0.717 | 0.657 | 0.694 | 0.481 | 0.274 |

*mAP50-95 pada 600 foto train tanpa augmentasi dikurangi mAP50-95 val.

Holdout Open Images (tidak ikut training), AP50: kentang 0.01 → 0.53, udang 0.14 → 0.45, jamur 0.24 → 0.71 (v3 → v4b).

Test set (654 foto, dievaluasi sekali untuk model terpilih): v4b mAP@0.5 0.694, mAP@0.5:0.95 0.465 (v3: 0.679 / 0.437). Smoothing + CLAHE sebelum deteksi menurunkan skor v4b di val dan test (mAP50 −3,9 dan −5,1 poin), sehingga dimatikan secara default. Peragaan tiga materi (histogram, smoothing, segmentasi + morfologi) dan analisis ablasinya ada di `notebooks/ReSnap_Implementasi_3_Materi.ipynb`.

Status v4b: lolos 4 dari 5 syarat. Gap train-val 0.274 sedikit di atas batas 0.268 (gap v3 + 0.05). Di kurva training, loss train dan val berjalan berdampingan sampai epoch 59 (epoch best.pt); keduanya baru berpisah setelah mosaic dimatikan di 10 epoch terakhir. Sisa gap sebagian berasal dari kapasitas YOLOv8s yang lebih besar dari YOLOv8n.

Kelas terlemah di test: kunyit (AP50 0.19), udang (0.28, hanya 6 objek), lengkuas (0.35), brokoli (0.37), jamur (0.42).

## Retrain v4 (urutan kerja)

1. `notebooks/ReSnap_Retrain_V4.ipynb` (Colab, T4, sekitar 3–5 jam): tambah foto kentang/udang/jamur dari Open Images V7 ke train, oversampling ringan, YOLOv8s. Val tetap sama dengan v3.
2. Notebook yang sama mendiagnosa overfit/underfit (`scripts/diagnose_fit.py`), membandingkan v3 dan v4 di val dan holdout Open Images, lalu memutuskan v4 diterima atau tidak berdasarkan 4 syarat tertulis.
3. Bila diterima, `models/best.pt` diganti v4 (v3 disimpan sebagai `best_v3.pt`), lalu jalankan `ReSnap_Evaluasi_Final_Test.ipynb` dengan `MODEL = 'v4'`. Test set dipakai sekali, setelah model dipilih.

Logika penyaringan data ada di `scripts/data_v4.py`.
