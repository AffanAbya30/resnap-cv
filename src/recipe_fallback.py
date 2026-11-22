"""Cadangan Modul 3 (pencocokan resep) supaya app.py bisa didemokan end-to-end.

Modul utama milik Ezekiel ada di src/recipe_service.py. app.py memakai modul
itu bila tersedia dan mengikuti kontrak yang sama:

    recommend(ingredients: list[str], top_k: int = 5) -> list[dict]

`ingredients` berisi nama kelas YOLO (mis. "green_onion"). Tiap dict resep:
    {"judul", "gambar", "skor" (0-1), "bahan_tersedia": [..], "bahan_kurang": [..],
     "langkah": [..], "sumber": url, "api": "spoonacular" | "themealdb"}

Skor: skor(r) = |B ∩ R| / |R|, R tanpa bumbu dasar (pantry). Seri -> bahan kurang
paling sedikit lebih dulu.
"""
from __future__ import annotations

import os
import re

import requests

from .detector import query_name

PANTRY = {
    "salt", "sugar", "oil", "water", "pepper", "black pepper", "olive oil", "vegetable oil",
    "butter", "flour", "garam", "gula", "minyak", "air", "lada", "merica",
}
ALIASES = {
    "green onion": {"spring onion", "scallion", "green onions", "spring onions", "scallions"},
    "chili pepper": {"chili", "chilli", "red chilli", "green chilli", "chillies", "cayenne"},
    "shallot": {"shallots"},
    "eggplant": {"aubergine"},
    "green beans": {"french beans", "string beans"},
    "long beans": {"yardlong beans", "snake beans"},
    "shrimp": {"prawns", "prawn", "king prawns", "shrimps"},
    "fish": {"salmon", "cod", "tuna", "white fish", "haddock", "mackerel", "tilapia"},
    "chicken": {"chicken breast", "chicken thighs", "chicken legs", "chicken wings"},
    "corn": {"sweetcorn", "sweet corn"},
    "cilantro": {"coriander"},
}
TIMEOUT = 10


def _singular(word: str) -> str:
    if word.endswith("oes") or word.endswith("ches") or word.endswith("shes"):
        return word[:-2]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _norm(text: str) -> str:
    words = re.sub(r"[^a-z ]", " ", text.lower()).split()
    return " ".join(_singular(w) for w in words)


def _matches(recipe_ing: str, detected_query: str) -> bool:
    r = _norm(recipe_ing)
    names = {_norm(n) for n in {detected_query} | ALIASES.get(detected_query, set())}
    return any(n == r or re.search(rf"\b{re.escape(n)}\b", r) for n in names)


def _is_pantry(ing: str) -> bool:
    r = _norm(ing)
    return r in PANTRY or any(r.endswith(" " + p) and len(r.split()) <= 3 for p in ("salt", "oil", "water"))


def score_recipe(recipe_ingredients: list[str], detected: list[str]) -> dict:
    """Hitung skor PRD untuk satu resep. `detected` = nama kelas YOLO."""
    queries = {cls: query_name(cls) for cls in detected}
    main = [i for i in dict.fromkeys(recipe_ingredients) if i and not _is_pantry(i)]
    available, missing = [], []
    for ing in main:
        (available if any(_matches(ing, q) for q in queries.values()) else missing).append(ing)
    total = len(main)
    return {
        "skor": len(available) / total if total else 0.0,
        "bahan_tersedia": available,
        "bahan_kurang": missing,
    }


def rank(recipes: list[dict], top_k: int = 5) -> list[dict]:
    return sorted(recipes, key=lambda r: (-r["skor"], len(r["bahan_kurang"]), r["judul"]))[:top_k]


# Spoonacular -----------------------------------------------------------------

def _spoonacular(detected: list[str], key: str, top_k: int) -> list[dict]:
    base = "https://api.spoonacular.com"
    resp = requests.get(f"{base}/recipes/findByIngredients", timeout=TIMEOUT, params={
        "ingredients": ",".join(query_name(c) for c in detected),
        "number": 20, "ranking": 1, "ignorePantry": "true", "apiKey": key,
    })
    resp.raise_for_status()
    out = []
    for item in resp.json():
        names = [i["name"] for i in item.get("usedIngredients", []) + item.get("missedIngredients", [])]
        out.append({
            "id": item["id"], "judul": item["title"], "gambar": item.get("image"),
            "sumber": f"https://spoonacular.com/recipes/-{item['id']}", "api": "spoonacular",
            "langkah": [], **score_recipe(names, detected),
        })
    top = rank(out, top_k)
    for r in top:  # langkah hanya diambil untuk resep yang ditampilkan (hemat kuota)
        try:
            steps = requests.get(f"{base}/recipes/{r['id']}/analyzedInstructions",
                                 params={"apiKey": key}, timeout=TIMEOUT).json()
            r["langkah"] = [s["step"] for block in steps for s in block.get("steps", [])]
        except (requests.RequestException, ValueError, KeyError, TypeError):
            pass
    return top


# TheMealDB -------------------------------------------------------------------

def _mealdb(detected: list[str], top_k: int, max_candidates: int = 25) -> list[dict]:
    base = "https://www.themealdb.com/api/json/v1/1"
    hits: dict[str, int] = {}
    for cls in detected:
        q = query_name(cls).replace(" ", "_")
        try:
            meals = requests.get(f"{base}/filter.php", params={"i": q}, timeout=TIMEOUT).json().get("meals") or []
        except (requests.RequestException, ValueError):
            continue
        for m in meals:
            hits[m["idMeal"]] = hits.get(m["idMeal"], 0) + 1
    # Kandidat yang memuat paling banyak bahan terdeteksi diperiksa lebih dulu.
    candidates = sorted(hits, key=hits.get, reverse=True)[:max_candidates]
    out = []
    for meal_id in candidates:
        try:
            meal = (requests.get(f"{base}/lookup.php", params={"i": meal_id}, timeout=TIMEOUT).json().get("meals") or [None])[0]
        except (requests.RequestException, ValueError):
            continue
        if not meal:
            continue
        names = [meal.get(f"strIngredient{i}") or "" for i in range(1, 21)]
        names = [n.strip() for n in names if n and n.strip()]
        steps = [s.strip() for s in re.split(r"\r?\n+", meal.get("strInstructions") or "") if s.strip()]
        out.append({
            "id": meal_id, "judul": meal["strMeal"], "gambar": meal.get("strMealThumb"),
            "sumber": meal.get("strSource") or f"https://www.themealdb.com/meal/{meal_id}",
            "api": "themealdb", "langkah": steps, **score_recipe(names, detected),
        })
    return rank(out, top_k)


def recommend(ingredients: list[str], top_k: int = 5) -> list[dict]:
    detected = [i for i in dict.fromkeys(ingredients) if i]
    if not detected:
        return []
    key = os.environ.get("SPOONACULAR_API_KEY", "").strip()
    if key:
        try:
            result = _spoonacular(detected, key, top_k)
            if result:
                return result
        except (requests.RequestException, ValueError, KeyError):
            pass  # kuota habis / jaringan -> jatuh ke TheMealDB
    return _mealdb(detected, top_k)
