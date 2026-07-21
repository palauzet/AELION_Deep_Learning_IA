"""Pipeline d'augmentation (Albumentations) pour les images saines d'entraînement."""

from __future__ import annotations

import albumentations as A
import numpy as np


def build_augmentation_pipeline() -> A.Compose:
    """Augmentations légères : rotation/décalage modérés, variations d'éclairage.

    Amplitudes volontairement modérées : `pill` est un objet centré et aligné,
    une augmentation géométrique trop forte élargirait la notion de "normal"
    au point de rendre l'auto-encodeur tolérant aux défauts (voir notebook 02).
    Pas de flip horizontal/vertical : la gravure « FF » n'est pas symétrique,
    un flip en inverserait la lecture en miroir — une configuration qu'aucune
    caméra fixe ne peut observer en réalité.
    """
    return A.Compose(
        [
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=0, border_mode=0, p=0.5
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        ]
    )


def augment_image(image: np.ndarray, pipeline: A.Compose, n: int = 4) -> list[np.ndarray]:
    """Génère `n` versions augmentées d'une image (float32 dans [0, 1])."""
    return [pipeline(image=image)["image"] for _ in range(n)]
