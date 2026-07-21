"""Score d'anomalie, seuil de détection et métriques d'évaluation (AUROC, heatmaps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import gaussian
from sklearn.metrics import confusion_matrix, roc_auc_score
from tensorflow import keras

from indusense.vision.dataset import DATA_ROOT, DEFAULT_IMAGE_SIZE


def reconstruct(model: keras.Model, images: np.ndarray) -> np.ndarray:
    """Passe les images dans l'auto-encodeur pour obtenir leur reconstruction."""
    return model.predict(images, verbose=0)


def error_maps(images: np.ndarray, reconstructions: np.ndarray) -> np.ndarray:
    """Erreur de reconstruction par pixel (MSE moyennée sur les canaux couleur)."""
    return np.mean((images - reconstructions) ** 2, axis=-1)


def image_scores(maps: np.ndarray) -> np.ndarray:
    """Score d'anomalie par image = erreur moyenne sur tous les pixels de l'image."""
    return maps.mean(axis=(1, 2))


def calibrate_threshold(
    val_scores: np.ndarray, method: str = "percentile", q: float = 99.0, k: float = 3.0
) -> float:
    """Calibre un seuil de détection à partir des scores des **saines de validation**.

    Décision : deux méthodes disponibles (`percentile` ou `mean_std`) plutôt qu'une
    seule figée. Pourquoi : l'énoncé propose lui-même les deux (centile 99, ou
    moyenne + k·écart-type) et le TP demande de discuter l'arbitrage
    rappel/fausses-alertes — le notebook 05 compare les deux plutôt que d'en imposer
    une seule. Le seuil est calibré **uniquement** sur la validation saine, jamais sur
    le test : l'utiliser sur du test reviendrait à calibrer avec les données qu'on
    évalue ensuite (fuite).
    """
    if method == "percentile":
        return float(np.percentile(val_scores, q))
    if method == "mean_std":
        return float(val_scores.mean() + k * val_scores.std())
    raise ValueError(f"méthode de calibration inconnue : {method!r}")


def predict_flags(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Indique quelles images dépassent le seuil (True = signalée comme anomalie)."""
    return scores > threshold


def image_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC niveau image (labels : 0 = sain, 1 = défaut), indépendant du seuil choisi."""
    return roc_auc_score(labels, scores)


def pixel_auroc(maps: np.ndarray, masks: np.ndarray) -> float:
    """AUROC niveau pixel : concatène toutes les cartes d'erreur et tous les masques GT."""
    return roc_auc_score(masks.flatten(), maps.flatten())


def confusion(scores: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """Matrice de confusion (0 = sain, 1 = défaut) au seuil donné."""
    return confusion_matrix(labels, predict_flags(scores, threshold))


def heatmap(error_map: np.ndarray, blur_sigma: float = 4.0) -> np.ndarray:
    """Carte d'erreur lissée et normalisée dans `[0, 1]`, pour l'affichage.

    Pourquoi le flou gaussien : l'erreur pixel brute est bruitée (variations locales
    aléatoires) ; un lissage léger fait ressortir les zones contiguës d'erreur élevée
    (les défauts), plus lisibles qu'une carte pixel brute.
    """
    smoothed = gaussian(error_map, sigma=blur_sigma)
    smoothed = smoothed - smoothed.min()
    max_value = smoothed.max()
    return smoothed / max_value if max_value > 0 else smoothed


def load_masks(
    defect: str,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    data_root: Path = DATA_ROOT,
) -> np.ndarray:
    """Charge les masques de vérité terrain d'une catégorie de défaut, en binaire.

    Redimensionnement en `NEAREST` (pas bicubique) pour ne pas introduire de valeurs
    intermédiaires dans un masque censé rester binaire.
    """
    paths = sorted((data_root / "ground_truth" / defect).glob("*.png"))
    masks = []
    for path in paths:
        with Image.open(path) as img:
            resized = img.convert("L").resize(image_size, Image.Resampling.NEAREST)
            masks.append(np.asarray(resized) > 0)
    return np.stack(masks)
