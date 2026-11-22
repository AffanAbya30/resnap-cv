"""Uji modul resep lokal dengan database mini. Jalankan: python -m pytest tests/ -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest  # noqa: E402

from build_recipe_db import classify  # noqa: E402
from src import recipe_local  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    def _use(recipes):
        rows = [{"id": i, "judul": judul, "kategori": "x", "bahan_kelas": kelas, "bahan_lain": lain,
                 "bahan_asli": [], "langkah": ["a", "b"], "loves": loves, "sumber": f"u{i}"}
                for i, (judul, kelas, lain, loves) in enumerate(recipes)]
        path = tmp_path / f"db{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps({"resep": rows}), encoding="utf-8")
        monkeypatch.setattr(recipe_local, "DB_PATH", path)
    return _use


def test_skor_prd(db):
    db([("Telur Balado", ["egg", "chili", "tomato", "shallot"], ["santan"], 0)])
    r = recipe_local.recommend(["egg", "chili", "tomato"])[0]
    assert r["skor"] == pytest.approx(3 / 5)          # |B∩R| / |R|, santan dihitung kurang
    assert r["bahan_kurang"] == ["Bawang merah", "santan"]


def test_seri_bahan_kurang_lebih_sedikit_menang(db):
    db([("Besar", ["egg", "chili", "tomato", "garlic"], [], 0),   # 2/4 = 0.5, kurang 2
        ("Kecil", ["egg", "garlic"], [], 0)])                       # 1/2 = 0.5, kurang 1
    out = recipe_local.recommend(["egg", "chili"])
    assert [o["judul"] for o in out] == ["Besar", "Kecil"]        # Besar memakai 2 bahan -> kelompok pertama
    out = recipe_local.recommend(["egg"])
    assert [o["judul"] for o in out] == ["Kecil", "Besar"]        # satu kelompok: kurang lebih sedikit menang


def test_judul_kembar_dibuang_dan_resep_satu_bahan_disaring(db):
    db([("Telur Balado", ["egg", "chili"], [], 5), ("telur balado!", ["egg", "chili"], [], 1),
        ("Telur Rebus", ["egg"], [], 99)])
    out = recipe_local.recommend(["egg", "chili"])
    assert [o["judul"] for o in out] == ["Telur Balado"]


def test_kosong(db):
    db([("A", ["egg", "chili"], [], 0)])
    assert recipe_local.recommend([]) == []
    assert recipe_local.recommend(["shrimp"]) == []


@pytest.mark.parametrize("token,expected", [
    ("telur ayam", "egg"), ("kaldu ayam", "pantry"), ("kembang kol", "cauliflower"), ("kol", "cabbage"),
    ("bawang bombai", "onion"), ("papan tempe", "tempeh"), ("labu siam", "other"), ("air jeruk nipis", "lime"),
    ("terasi udang", "pantry"), ("kerupuk udang", "other"), ("potong", None), ("buang kepala", None),
])
def test_pemetaan_bahan(token, expected):
    assert classify(token) == expected
