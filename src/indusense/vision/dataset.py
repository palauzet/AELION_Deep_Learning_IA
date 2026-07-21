"""Chargement et préparation du dataset MVTec AD (catégorie `pill`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DATA_ROOT = Path(__file__).resolve().parents[3] / "pill"
DEFAULT_IMAGE_SIZE = (256, 256)


def list_defects(data_root: Path = DATA_ROOT) -> list[str]:
    """Liste les catégories de défauts disponibles (sous-dossiers de `test/`, hors `good`)."""
    return sorted(p.name for p in (data_root / "test").iterdir() if p.is_dir() and p.name != "good")


def _load_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        resized = img.convert("RGB").resize(image_size)
        return np.asarray(resized, dtype=np.float32) / 255.0


def _load_folder(folder: Path, image_size: tuple[int, int]) -> np.ndarray:
    paths = sorted(folder.glob("*.png"))
    return np.stack([_load_image(p, image_size) for p in paths])


def load_good_images(
    split: str = "train",
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    data_root: Path = DATA_ROOT,
) -> np.ndarray:
    """Charge les images saines (`good`) d'un split (`train` ou `test`), normalisées dans [0, 1]."""
    return _load_folder(data_root / split / "good", image_size)


def load_defect_images(
    defect: str,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    data_root: Path = DATA_ROOT,
) -> np.ndarray:
    """Charge les images défectueuses d'une catégorie (`test/<defect>/`), normalisées [0, 1]."""
    return _load_folder(data_root / "test" / defect, image_size)


def train_val_split(
    images: np.ndarray, val_fraction: float = 0.15, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Réserve une fraction des images saines pour la validation (calibrage du seuil, partie 2)."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(images))
    n_val = int(len(images) * val_fraction)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    return images[train_idx], images[val_idx]
