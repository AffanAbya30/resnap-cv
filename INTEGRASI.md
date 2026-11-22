# Integrasi ReSnap ke web app

Deteksi bahan (YOLOv8 v4b) + rekomendasi resep lokal dalam satu fungsi Python. Tidak butuh internet, API, atau LLM.

## Isi paket

| File | Fungsi |
|---|---|
| `resnap_pipeline.py` | Pintu masuk: `scan()` dan `rekomendasi()` |
| `src/preprocessing.py`, `src/detector.py`, `src/recipe_local.py` | Modul yang dipanggil `resnap_pipeline.py` |
| `models/best.pt` | Model deteksi v4b (YOLOv8s, 30 kelas, 22,5 MB) |
| `data/recipes.json.gz` | Database 14.658 resep Cookpad (3,9 MB) |
| `notebooks/ReSnap_Demo_Pipeline.ipynb` | Demo yang sudah dijalankan |
| `contoh/contoh_output.json`, `contoh/contoh_output.jpg` | Contoh keluaran `scan()` dari `contoh/foto_udang.jpg` |
| `contoh/contoh_rekomendasi_edit_bahan.json` | Contoh keluaran `rekomendasi()` setelah bahan diedit |

Struktur folder harus dipertahankan (model di `models/`, database di `data/`).

## Setup

```bash
pip install -r requirements.txt      # Python 3.10+, ultralytics==8.3.228
python resnap_pipeline.py contoh/foto_udang.jpg     # cek cepat, mencetak JSON
```

## Pemakaian

```python
from resnap_pipeline import scan, rekomendasi

hasil = scan("foto.jpg")              # path, bytes hasil upload, atau array BGR OpenCV
hasil = scan(file_bytes, conf=0.5, top_k=5)

# user menghapus/menambah bahan di frontend -> kirim nama KELAS (bukan nama Indonesia)
resep = rekomendasi(["egg", "tomato", "shallot", "chili"], top_k=5)
```

Contoh endpoint (FastAPI):

```python
from fastapi import FastAPI, UploadFile
from resnap_pipeline import scan, rekomendasi

app = FastAPI()
scan("contoh/foto_udang.jpg")          # pemanasan: model dimuat sekali saat server mulai

@app.post("/scan")
async def scan_endpoint(foto: UploadFile, conf: float = 0.5):
    return scan(await foto.read(), conf=conf)

@app.post("/rekomendasi")
def rekomendasi_endpoint(bahan: list[str]):
    return rekomendasi(bahan)
```

## Format keluaran `scan()`

```jsonc
{
  "gambar":  {"lebar": 960, "tinggi": 720},
  "deteksi": [                                  // satu entri per kotak
    {"kelas": "shrimp", "nama": "Udang", "confidence": 0.8652, "box": [x1, y1, x2, y2]}
  ],
  "bahan": [                                    // bahan unik, urut confidence tertinggi
    {"kelas": "shrimp", "nama": "Udang", "jumlah": 3, "confidence_maks": 0.8652}
  ],
  "resep": [                                    // Top-k, sudah terurut
    {
      "judul": "Udang Bakar Madu Simple",
      "skor": 0.5,                              // 0..1, tampilkan sebagai persen
      "bahan_tersedia": ["Udang"],
      "bahan_kurang": ["madu"],
      "bahan_resep": ["Udang 1/4 kg yang besar", "3 sendok makan madu", "..."],
      "langkah": ["Cuci udang ...", "..."],
      "kategori": "udang",
      "loves": 12,
      "sumber": "https://cookpad.com/id/resep/..."
    }
  ],
  "pengaturan": {"conf": 0.5, "iou": 0.7, "top_k": 5, "enhance": false, "model": "v4b (YOLOv8s, 30 kelas)"},
  "waktu_ms": {"preprocessing": 117, "deteksi": 260, "resep": 50}
}
```

`rekomendasi()` mengembalikan list yang sama dengan bagian `resep` di atas.

Catatan untuk frontend:
- `box` dalam piksel foto asli (x1, y1 kiri atas; x2, y2 kanan bawah). Untuk menggambar di foto yang diperkecil, kalikan dengan `lebar_tampil / gambar.lebar`.
- `jumlah` = jumlah kotak terdeteksi, bukan jumlah bahan sebenarnya (pada tumpukan, banyak objek tidak terdeteksi).
- `resep` bisa kosong bila tidak ada bahan terdeteksi; tampilkan ajakan menambah bahan manual.
- Resep tidak punya gambar (dataset Cookpad tidak menyertakannya); `sumber` bisa dipakai sebagai tautan.
- Panggilan pertama lebih lama (sekitar 2 detik, memuat model); berikutnya sekitar 0,3 detik di CPU.
- `conf` default 0,5 sesuai PRD. Bisa disediakan slider 0,25–0,9.

## Daftar kelas bahan

| kelas | nama | kelas | nama | kelas | nama |
|---|---|---|---|---|---|
| tomato | Tomat | broccoli | Brokoli | lime | Jeruk nipis |
| egg | Telur | cauliflower | Kembang kol | ginger | Jahe |
| onion | Bawang bombay | cucumber | Timun | turmeric | Kunyit |
| shallot | Bawang merah | eggplant | Terong | galangal | Lengkuas |
| garlic | Bawang putih | mushroom | Jamur | lemongrass | Serai |
| chili | Cabai | spinach | Bayam | tofu | Tahu |
| carrot | Wortel | green_onion | Daun bawang | tempeh | Tempe |
| potato | Kentang | green_beans | Buncis | chicken | Ayam |
| cabbage | Kubis | long_beans | Kacang panjang | fish | Ikan |
| corn | Jagung | pumpkin | Labu | shrimp | Udang |
