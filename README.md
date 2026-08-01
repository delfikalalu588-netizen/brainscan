# BrainScan AI — Final Project

## Arsitektur Deployment
- **Model AI**: Hugging Face (`delfidev/brain-hybrid-efficientnet-vit`, format **ONNX**) — gratis, model registry saja
- **Backend API** (FastAPI, `src/app.py`, runtime **ONNX Runtime** — bukan PyTorch, biar ringan): **Render** (free tier)
- **Database** (data pasien & riwayat scan): **Turso** — SQLite-compatible, dipisah dari backend supaya permanen
- **Frontend**: hosting terpisah (**InfinityFree**) — sesuaikan `API_BASE_URL` di `src/static/app.js` ke URL Render kamu

## Kenapa ONNX, bukan PyTorch?
Render free tier RAM-nya terbatas (~512MB). PyTorch + torchvision ukurannya
ratusan MB dan makan banyak RAM saat load model. ONNX Runtime jauh lebih
ringan, jadi lebih mungkin muat di free tier. Konsekuensinya: fitur heatmap
("Peta Atensi Model") sekarang dibaca langsung dari attention weight yang
di-export sebagai output tambahan model (`src/explainability.py`), BUKAN
Grad-CAM klasik (yang butuh backward pass, tidak didukung ONNX Runtime).

## Deploy ke Render
1. Buat akun Render, klik **New > Web Service**, hubungkan ke repo Git project ini
2. **Build Command**: `pip install -r requirements.txt`
3. **Start Command**: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`
4. Set environment variables di Settings > Environment (lihat bawah)
5. Deploy — Render otomatis kasih URL publik (`https://nama-app.onrender.com`)

## Setup Turso (database terpisah)
1. Buat akun di https://turso.tech, buat database baru
2. Ambil `TURSO_DATABASE_URL` dan `turso db tokens create nama-db` buat dapetin `TURSO_AUTH_TOKEN`
3. (Opsional) migrasi data lama dari `brainscan.db`: `turso db shell nama-db < brainscan.db`

## Environment variables yang perlu di-set di Render
```
TURSO_DATABASE_URL=libsql://nama-db-kamu.turso.io
TURSO_AUTH_TOKEN=isi-token-turso-kamu
GEMINI_API_KEY=isi-key-gemini-kamu   # opsional, fallback ke laporan lokal kalau kosong
```

## Setup Frontend (InfinityFree)
1. Upload isi folder `src/static/` (index.html, app.js, index.css) ke hosting InfinityFree
2. Buka `src/static/app.js`, ganti baris `API_BASE_URL` ke URL Render kamu, misal:
   ```js
   const API_BASE_URL = "https://nama-app.onrender.com";
   ```

## Cara jalanin LOKAL untuk testing (dari root folder ini)
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

export TURSO_DATABASE_URL="..."          # wajib, biar fitur riwayat pasien jalan
export TURSO_AUTH_TOKEN="..."
export GEMINI_API_KEY="isi-key-kamu"     # opsional

uvicorn src.app:app --reload
```
Buka `http://127.0.0.1:8000/docs` untuk dokumentasi endpoint interaktif.

## Yang perlu diperhatikan
- Model **ONNX** (bukan `.pth` lagi) otomatis di-download dari Hugging Face
  (`delfidev/brain-hybrid-efficientnet-vit`) pas server pertama kali start.
- Model hybrid punya **2 output** (`logits` + `attention`) — lihat
  `reexport_onnx_with_attention.py` kalau perlu re-export ulang.
- Kalau `best_precheck_model.onnx` belum ada di HF repo, precheck otomatis
  dilewati (semua gambar dianggap valid) — ada warning di log, bukan error.
- File training (`train_classifier.py`, `data_loader.py`, dll) sengaja
  tidak diikutkan di sini karena tidak dipakai untuk menjalankan API;
  itu tetap ada di notebook/project training terpisah.
