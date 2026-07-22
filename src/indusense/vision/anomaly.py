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


def healthy_baseline(val_maps: np.ndarray, smooth_sigma: float = 4.0) -> np.ndarray:
    """Carte de référence saine : moyenne, pixel à pixel, des cartes d'erreur lissées
    des saines de validation.

    Décision : lisser **chaque carte individuellement** avant de moyenner, jamais
    flouter le tableau 3D `(n_images, H, W)` d'un coup (cela mélangerait aussi selon
    l'axe des images, sans rapport avec l'espace).

    Pourquoi cette référence : certaines zones (contour ovale, gravure embossée
    « FF ») ont une erreur de reconstruction systématiquement élevée sur **toute**
    pièce saine (bottleneck compressé ×12, cf. notebook 03) — sans référence, un
    score basé sur le maximum ou un centile brut de la carte d'erreur confond ce
    bruit de fond normal avec un vrai défaut. Cette carte sert à le soustraire
    (`image_scores_pooled`) avant de chercher un pic anormal.
    """
    smoothed = np.stack([gaussian(m, sigma=smooth_sigma) for m in val_maps])
    return smoothed.mean(axis=0)


def image_scores_pooled(
    maps: np.ndarray, baseline: np.ndarray, q: float = 99.5, smooth_sigma: float = 4.0
) -> np.ndarray:
    """Score d'anomalie par image = centile élevé de l'erreur **après soustraction**
    de la référence saine (`healthy_baseline`), plutôt qu'une simple moyenne globale.

    Décision : soustraction de la moyenne saine, **sans** division par un
    écart-type par pixel. Pourquoi : une normalisation par écart-type par pixel a été
    testée et écartée — avec seulement ~40 images de validation, l'écart-type sur
    65 536 positions de pixels indépendantes est bien trop bruité pour généraliser
    (mesuré : 100 % de fausses alertes sur les saines de test). Une simple
    soustraction de moyenne, plus robuste, reste stable (0 % de fausse alerte) tout
    en corrigeant le bruit de fond décrit dans `healthy_baseline`.

    Décision : centile élevé (`q`, défaut 99,5) plutôt que la moyenne globale
    (`image_scores`). Pourquoi : la moyenne globale dilue un défaut localisé sur
    quelques centaines de pixels dans une moyenne sur 65 536 pixels (cf. écart
    AUROC pixel vs image du notebook 06) ; un centile élevé se concentre sur les
    quelques pixels les plus anormaux de l'image, là où se trouve un défaut localisé.
    """
    smoothed = np.stack([gaussian(m, sigma=smooth_sigma) for m in maps])
    residual = np.maximum(smoothed - baseline, 0)
    return np.percentile(residual.reshape(len(residual), -1), q, axis=1)


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
