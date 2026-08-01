import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.config import OUTPUT_DIR
from src.api import router as api_router

# ─────────────────────────────────────────────────────────────
# 1. Inisialisasi Aplikasi FastAPI
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="BrainScan AI Framework API",
    description="API untuk analisis otomatis CT-Scan & MRI menggunakan arsitektur Hybrid CNN-Transformer",
    version="1.0",
)

# ─────────────────────────────────────────────────────────────
# 2. CORS — biar frontend (domain/origin manapun) bisa akses API ini
# ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
# 3. Folder output yang diperlukan saat runtime
# ─────────────────────────────────────────────────────────────
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 3b. Health check -- beberapa platform hosting (SnapDeploy, dll)
#     rutin ngecek endpoint ini buat mastiin container sehat
# ─────────────────────────────────────────────────────────────
@app.get("/health")
@app.get("/healthz")
def health_check():
    return {"status": "ok"}

# ─────────────────────────────────────────────────────────────
# 4. Sambungkan semua endpoint dari api.py
#    (mengimpor src.api otomatis menjalankan proses load model
#    di dalamnya, lihat komentar di bagian atas api.py)
# ─────────────────────────────────────────────────────────────
app.include_router(api_router)

# ─────────────────────────────────────────────────────────────
# 5. Mount static files
#    - /outputs/figures -> hasil heatmap Grad-CAM (fallback akses langsung)
#    - /                 -> frontend (index.html, app.js, index.css)
#    Urutan penting: mount("/") harus PALING TERAKHIR supaya tidak
#    "menutupi" route API yang sudah didaftarkan lewat include_router.
# ─────────────────────────────────────────────────────────────
app.mount(
    "/outputs/figures",
    StaticFiles(directory=os.path.join(OUTPUT_DIR, "figures")),
    name="figures",
)
app.mount("/", StaticFiles(directory="src/static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    # Jalankan dari ROOT proyek (bukan dari dalam folder src/):
    #   uvicorn src.app:app --reload
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=True)
