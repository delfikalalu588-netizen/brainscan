"""
preprocess.py  (versi ONNX — tanpa torch/torchvision)
------------------------------------------------------
Backend sekarang full ONNX Runtime (biar ringan buat Render free tier),
jadi preprocessing gambar ditulis ulang pakai PIL + numpy murni,
menghasilkan array [1, 3, H, W] float32 — persis format yang dibutuhkan
ONNX Runtime (menggantikan torchvision.transforms yang butuh torch).

Statistik normalisasi & urutan operasi TETAP SAMA PERSIS dengan versi
PyTorch sebelumnya, supaya hasil inference tidak berubah.
"""

import numpy as np
from PIL import Image

from src.config import IMG_SIZE

# Sama seperti sebelumnya: statistik ImageNet untuk model hybrid
# (backbone EfficientNet-B3 di-pretrain pakai statistik ini)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Model precheck dilatih terpisah pakai normalisasi [0.5,0.5,0.5]
PRECHECK_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PRECHECK_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _pil_to_chw_array(image: Image.Image, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Resize -> [0,1] float -> normalize -> ubah HWC jadi CHW -> tambah
    dimensi batch. Setara dengan T.Resize + T.ToTensor + T.Normalize."""
    image = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0        # HWC, [0,1]
    arr = (arr - mean) / std                                  # normalize per-channel
    arr = arr.transpose(2, 0, 1)                               # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)                          # tambah batch dim -> [1,3,H,W]
    return arr.astype(np.float32)


def val_transforms(image: Image.Image) -> np.ndarray:
    """Preprocessing untuk model hybrid utama (normalisasi ImageNet)."""
    return _pil_to_chw_array(image, IMAGENET_MEAN, IMAGENET_STD)


def precheck_transforms(image: Image.Image) -> np.ndarray:
    """Preprocessing untuk model precheck (normalisasi [-1,1])."""
    return _pil_to_chw_array(image, PRECHECK_MEAN, PRECHECK_STD)
