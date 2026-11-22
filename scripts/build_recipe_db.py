"""Bangun database resep lokal (data/recipes.json) dari dataset Cookpad Indonesia.

Sumber: Kaggle "Indonesian Food Recipes" (14.945 resep, 8 kategori), file
Indonesian_Food_Recipes.csv dengan kolom 'Ingredients Cleaned'.

Setiap token bahan dipetakan ke salah satu dari:
  - kelas deteksi ReSnap (30 kelas)  -> bisa dicocokkan dengan hasil YOLO
  - bumbu dasar (pantry)             -> dianggap selalu ada, tidak dihitung di skor
  - bahan lain                       -> tetap dihitung sebagai bahan yang kurang
  - bukan bahan (sisa kata pembersihan seperti "potong", "haluskan") -> dibuang

Contoh:
  python scripts/build_recipe_db.py --csv Indonesian_Food_Recipes.csv --out data/recipes.json.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

# Urutan penting: aturan yang lebih spesifik ditulis dulu.
CLASS_RULES = [
    (r"kaldu|royco|masako|penyedap|bubuk rasa|saus|saos|kecap|terasi|petis", None),
    (r"kerupuk|\bebi\b", "other"),
    (r"kembang kol", "cauliflower"),
    (r"daun bawang|bawang daun|loncang|prei|bawang prei", "green_onion"),
    (r"bawang bomb", "onion"),
    (r"bawang merah", "shallot"),
    (r"bawang putih", "garlic"),
    (r"\btelu?r\b|\btelor\b|kuning telur|putih telur", "egg"),
    (r"\btomat\b", "tomato"),
    (r"\bcab[ea]i?\b|\bcabe\b|\brawit\b|\blombok\b|\bcengek\b", "chili"),
    (r"\bwortel\b", "carrot"),
    (r"\bkentang\b", "potato"),
    (r"\bkol\b|\bkubis\b", "cabbage"),
    (r"\bbrok?oli\b", "broccoli"),
    (r"\bk?etimun\b|\btimun\b|\bmentimun\b", "cucumber"),
    (r"\bteron?g\b|\bterung\b", "eggplant"),
    (r"\bjamur\b", "mushroom"),
    (r"\bbayam\b", "spinach"),
    (r"\bbuncis\b", "green_beans"),
    (r"kacang panjang", "long_beans"),
    (r"\bjagung\b", "corn"),
    (r"labu siam", "other"),
    (r"\blabu\b|\bwaluh\b", "pumpkin"),
    (r"jeruk nipis|jeruk limau|\blimau\b", "lime"),
    (r"\bjahe\b", "ginger"),
    (r"\bkunyit\b|\bkunir\b", "turmeric"),
    (r"\blengkuas\b|\blaos\b", "galangal"),
    (r"\bserai\b|\bsereh\b|\bsere\b", "lemongrass"),
    (r"\btahu\b", "tofu"),
    (r"\btempe\b", "tempeh"),
    (r"\budang\b", "shrimp"),
    (r"\bayam\b", "chicken"),
    (r"\bikan\b|\btongkol\b|\blele\b|\bnila\b|\bbandeng\b|\bkakap\b|\btuna\b|\bsalmon\b|\bteri\b|\bpatin\b"
     r"|\bgurame\b|\bkembung\b|\bdori\b|\bmujair\b|\bcakalang\b|\btenggiri\b", "fish"),
]
# Bumbu dasar dapur Indonesia yang dianggap selalu tersedia (PRD: garam, gula, minyak, air, lada; diperluas).
PANTRY = (r"\bgaram\b|\bgula\b|\bminyak\b|\bair\b|\blada\b|\bmerica\b|penyedap|kaldu|royco|masako|micin|\bmsg\b"
          r"|\bkecap\b|\bsaus\b|\bsaos\b|\bsambal\b|\bkemiri\b|\bketumbar\b|daun salam|\bsalam\b|daun jeruk"
          r"|\bterasi\b|\bpala\b|\bcengkeh\b|kayu manis|\bjinten\b|\bjintan\b|kapulaga|bunga lawang|asam jawa"
          r"|\bmentega\b|\bmargarin\b|\bmargarine\b|bawang goreng|\bladaku\b|\bmerica\b|\bcuka\b|\bes batu\b"
          r"|\bes\b|\bvetsin\b|\bsasa\b|\bajinomoto\b|\bbumbu\b")
NOT_INGREDIENT = (r"^(potong|tipis|geprek|haluskan|halus|cincang|iris|serong|memarkan|belah|kupas|rebus|goreng"
                  r"|sangrai|pelengkap|tambahan|isian|dihaluskan|secukupnya|sesuai selera|bumbu halus"
                  r"|bumbu dihaluskan|bumbu haluskan|bumbu pelengkap|potong potong|potong dadu|potong kotak"
                  r"|cincang halus|cincang kasar|tusuk sate|daun pisang|bahan|optional|opsional|utk|untuk"
                  r"|bersihkan|kocok lepas|dadu|kotak|panjang|bulat|utuh|matang|segar|kasar|besar|kecil)$")

CLASS_NAMES = [
    "tomato", "egg", "onion", "shallot", "garlic", "chili", "carrot", "potato", "cabbage", "broccoli",
    "cauliflower", "cucumber", "eggplant", "mushroom", "spinach", "green_onion", "green_beans",
    "long_beans", "corn", "pumpkin", "lime", "ginger", "turmeric", "galangal", "lemongrass", "tofu",
    "tempeh", "chicken", "fish", "shrimp",
]


NOISE_WORDS = r"\b(buang|mangkok|mangkuk|piring|wadah|plastik|panci|wajan|sendok|gelas|sesuai|secukup\w*|sck)\b"


def classify(token: str) -> str | None:
    """Kembalikan nama kelas, 'pantry', 'other', atau None (bukan bahan)."""
    t = re.sub(r"\s+", " ", token.lower()).strip()
    if not t or len(t) < 2 or re.fullmatch(NOT_INGREDIENT, t):
        return None
    if re.search(NOISE_WORDS, t) and classify_core(t) == "other":
        return None
    return classify_core(t)


def classify_core(t: str) -> str:
    for pattern, cls in CLASS_RULES:
        if re.search(pattern, t):
            if cls is None:
                return "pantry"
            return cls
    if re.search(PANTRY, t):
        return "pantry"
    return "other"


def parse_steps(text: str) -> list[str]:
    steps = []
    for line in re.split(r"\n+", text or ""):
        line = re.sub(r"^\s*\d+\)\s*", "", line).strip()
        if line:
            steps.append(line)
    return steps


def build(csv_path: Path) -> tuple[list[dict], dict]:
    recipes, stats = [], Counter()
    unknown = Counter()
    seen_urls = set()
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = row["URL"].strip()
            if url in seen_urls:
                stats["duplikat"] += 1
                continue
            seen_urls.add(url)
            tokens = [t.strip() for t in (row["Ingredients Cleaned"] or "").split(",") if t.strip()]
            main, other = [], []
            for tok in tokens:
                c = classify(tok)
                if c is None or c == "pantry":
                    continue
                if c == "other":
                    if tok not in other:
                        other.append(tok)
                        unknown[tok] += 1
                elif c not in main:
                    main.append(c)
            steps = parse_steps(row["Steps"])
            if not main or len(steps) < 2:
                stats["dibuang_tanpa_kelas_atau_langkah"] += 1
                continue
            recipes.append({
                "id": len(recipes),
                "judul": row["Title"].strip(),
                "kategori": row["Category"].strip(),
                "bahan_kelas": main,          # bahan utama yang bisa dideteksi
                "bahan_lain": other,          # bahan utama yang tidak bisa dideteksi
                "bahan_asli": [b.strip() for b in row["Ingredients"].split("--") if b.strip()],
                "langkah": steps,
                "loves": int(float(row["Loves"] or 0)),
                "sumber": url,
            })
            stats["dipakai"] += 1
    stats["bahan_lain_teratas"] = unknown.most_common(40)
    return recipes, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", default=Path(__file__).resolve().parent.parent / "data" / "recipes.json.gz", type=Path)
    args = ap.parse_args()

    recipes, stats = build(args.csv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"sumber": "Kaggle Indonesian Food Recipes (Cookpad, 2018)", "resep": recipes},
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    args.out.write_bytes(gzip.compress(payload, mtime=0) if args.out.suffix == ".gz" else payload)
    coverage = Counter(c for r in recipes for c in r["bahan_kelas"])
    print(json.dumps({k: v for k, v in stats.items() if k != "bahan_lain_teratas"}, indent=2))
    print("Resep per kelas:", {c: coverage[c] for c in CLASS_NAMES})
    print("Bahan lain teratas:", stats["bahan_lain_teratas"][:25])
    print(f"Tersimpan: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
