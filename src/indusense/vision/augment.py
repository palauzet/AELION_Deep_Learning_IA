"""Pipeline d'augmentation (Albumentations) pour les images saines d'entraînement."""

from __future__ import annotations

import math

import albumentations as A
import numpy as np


def build_augmentation_pipeline() -> A.ReplayCompose:
    """Augmentations légères : rotation/décalage modérés, variations d'éclairage.

    Amplitudes volontairement modérées : `pill` est un objet centré et aligné,
    une augmentation géométrique trop forte élargirait la notion de "normal"
    au point de rendre l'auto-encodeur tolérant aux défauts (voir notebook 02).
    Pas de flip horizontal/vertical : la gravure « FF » n'est pas symétrique,
    un flip en inverserait la lecture en miroir — une configuration qu'aucune
    caméra fixe ne peut observer en réalité.

    `ReplayCompose` (plutôt que `Compose`) enregistre les paramètres tirés à
    chaque appel, pour pouvoir décrire précisément ce qui a été appliqué
    (cf. `describe_replay`).
    """
    return A.ReplayCompose(
        [
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.Affine(translate_percent=(-0.05, 0.05), scale=(0.95, 1.05), border_mode=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        ]
    )


def augment_image(image: np.ndarray, pipeline: A.ReplayCompose, n: int = 4) -> list[dict]:
    """Génère `n` versions augmentées d'une image (float32 dans [0, 1]).

    Chaque élément est le dict brut renvoyé par le pipeline (`image` + `replay`),
    pour permettre à l'appelant de décrire les transformations réellement tirées
    via `describe_replay`.
    """
    return [pipeline(image=image) for _ in range(n)]


def describe_replay(replay: dict) -> str:
    """Décrit en une phrase les transformations réellement tirées pour un appel du pipeline."""
    parts = []
    for t in replay["transforms"]:
        if not t["applied"]:
            continue
        name = t["__class_fullname__"]
        params = t["params"]
        if name == "HorizontalFlip":
            parts.append("flip horizontal (miroir gauche/droite)")
        elif name == "VerticalFlip":
            parts.append("flip vertical (miroir haut/bas)")
        elif name == "Rotate":
            matrix = params["matrix"]
            angle = math.degrees(math.atan2(matrix[0, 1], matrix[0, 0]))
            parts.append(f"rotation {angle:+.1f}°")
        elif name == "Affine":
            matrix = params["matrix"]
            height, width = params["shape"][:2]
            tx_pct = matrix[0, 2] / width * 100
            ty_pct = matrix[1, 2] / height * 100
            scale = params["scale"]
            parts.append(
                f"translation ({tx_pct:+.1f} %, {ty_pct:+.1f} %), "
                f"échelle (×{scale['x']:.2f}, ×{scale['y']:.2f})"
            )
        elif name == "RandomBrightnessContrast":
            parts.append(f"contraste ×{params['alpha']:.2f}, luminosité {params['beta']:+.2f}")
    return "; ".join(parts) if parts else "aucune transformation tirée (probabilités défavorables)"
