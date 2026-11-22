"""ReSnap — Recipe from a Snap (Streamlit UI).

Jalankan:  streamlit run app.py
Alur: foto -> preprocessing (OpenCV) -> YOLOv8 -> bahan (bisa diedit) -> Recipe API -> Top-5 resep.
"""
from __future__ import annotations

import time

import streamlit as st

from src.detector import INGREDIENTS, IngredientDetector, display_name, draw_detections
from src.preprocessing import decode_image, run_pipeline

from src import recipe_local

try:  # modul Ezekiel bila sudah ada
    from src.recipe_service import recommend
    RECIPE_SOURCE = "recipe_service (Ezekiel)"
except ImportError:
    if recipe_local.available():  # database resep lokal, tanpa internet
        recommend = recipe_local.recommend
        RECIPE_SOURCE = "database lokal (Cookpad, 14 ribu resep)"
    else:
        from src.recipe_fallback import recommend
        RECIPE_SOURCE = "API online (Spoonacular/TheMealDB)"

st.set_page_config(page_title="ReSnap", layout="wide")


@st.cache_resource(show_spinner="Memuat model YOLOv8...")
def load_detector() -> IngredientDetector:
    return IngredientDetector()


@st.cache_data(show_spinner=False, ttl=3600)
def cached_recommend(ingredients: tuple[str, ...], top_k: int) -> list[dict]:
    return recommend(list(ingredients), top_k=top_k)


# Sidebar ---------------------------------------------------------------------
with st.sidebar:
    st.header("Pengaturan")
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.50, 0.05,
                     help="PRD: deteksi di bawah 0.5 dibuang. Turunkan bila ada bahan terlewat.")
    iou = st.slider("IoU NMS", 0.30, 0.90, 0.70, 0.05)
    enhance = st.toggle("Denoise + CLAHE sebelum deteksi", value=False,
                        help="Membantu foto dapur yang redup. Model dilatih dengan foto mentah, "
                             "jadi default-nya mati.")
    top_k = st.number_input("Jumlah resep", 1, 10, 5)
    st.caption(f"Sumber resep: {RECIPE_SOURCE}")

st.title("ReSnap")
st.caption("Foto bahan di dapur, dapatkan resep yang paling cocok.")

# Input -----------------------------------------------------------------------
tab_upload, tab_camera = st.tabs(["Unggah foto", "Kamera"])
with tab_upload:
    uploaded = st.file_uploader("Pilih foto bahan", type=["jpg", "jpeg", "png", "webp"])
with tab_camera:
    captured = st.camera_input("Ambil foto")

source = captured or uploaded
if source is None:
    st.info("Unggah atau ambil foto bahan makanan untuk mulai.")
    st.stop()

try:
    image_bgr = decode_image(source.getvalue())
except ValueError as e:
    st.error(str(e))
    st.stop()

# Preprocessing + deteksi -----------------------------------------------------
t0 = time.perf_counter()
pre = run_pipeline(image_bgr, enhance_for_detection=enhance)
t1 = time.perf_counter()
result = load_detector().detect(pre, conf=conf, iou=iou)
t2 = time.perf_counter()

col_a, col_b = st.columns(2)
col_a.image(pre.original_bgr[:, :, ::-1], caption="Foto asli", width="stretch")
col_b.image(draw_detections(image_bgr, result),
            caption=f"{len(result.detections)} objek terdeteksi", width="stretch")
st.caption(f"Preprocessing {1000 * (t1 - t0):.0f} ms, deteksi {1000 * (t2 - t1):.0f} ms")

with st.expander("Tahapan pra-pemrosesan citra"):
    stages = pre.stages()
    cols = st.columns(4)
    for i, (name, img) in enumerate(stages.items()):
        cols[i % 4].image(img, caption=name, width="stretch", clamp=True)
    st.json(pre.stats)

with st.expander("Detail deteksi"):
    if result.detections:
        st.dataframe(
            [{"bahan": display_name(d.cls), "kelas": d.cls, "confidence": round(d.conf, 3),
              "box_xyxy": [round(v) for v in d.box]} for d in result.detections],
            width="stretch",
        )
    else:
        st.write("Tidak ada objek di atas threshold.")

# Konfirmasi bahan ------------------------------------------------------------
st.subheader("Bahan yang kamu punya")
detected = sorted(result.ingredients, key=lambda c: -result.ingredients[c]["conf_maks"])
key = f"bahan_{hash(source.getvalue())}_{conf}_{iou}_{enhance}"
chosen = st.multiselect(
    "Hapus yang salah deteksi atau tambahkan yang terlewat",
    options=list(INGREDIENTS),
    default=detected,
    format_func=display_name,
    key=key,
)
if not chosen:
    st.warning("Belum ada bahan. Tambahkan minimal satu bahan secara manual.")
    st.stop()

# Rekomendasi resep -----------------------------------------------------------
st.subheader("Rekomendasi resep")
with st.spinner("Mencari resep..."):
    try:
        recipes = cached_recommend(tuple(sorted(chosen)), int(top_k))
    except Exception as e:  # jaringan/kuota API
        st.error(f"Gagal menghubungi Recipe API: {e}")
        st.stop()

if not recipes:
    st.info("Belum ada resep yang cocok untuk kombinasi bahan ini.")
    st.stop()

for r in recipes:
    with st.container(border=True):
        c_img, c_body = st.columns([1, 3])
        if r.get("gambar"):
            c_img.image(r["gambar"], width="stretch")
        c_body.markdown(f"#### {r['judul']}")
        c_body.progress(min(max(r["skor"], 0.0), 1.0), text=f"Kecocokan {r['skor']:.0%}")
        have = ", ".join(r.get("bahan_tersedia", [])) or "-"
        missing = ", ".join(r.get("bahan_kurang", [])) or "tidak ada"
        c_body.markdown(f":green-background[Tersedia] {have}")
        c_body.markdown(f":orange-background[Kurang] {missing}")
        if r.get("bahan_resep"):
            with c_body.expander("Bahan lengkap"):
                for b in r["bahan_resep"]:
                    st.markdown(f"- {b}")
        if r.get("langkah"):
            with c_body.expander("Langkah memasak"):
                for i, step in enumerate(r["langkah"], 1):
                    st.markdown(f"{i}. {step}")
        if r.get("sumber"):
            c_body.markdown(f"[Resep lengkap]({r['sumber']})")
