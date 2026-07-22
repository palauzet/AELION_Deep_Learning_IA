"""Détection d'anomalie par comparaison à une banque de patches sains (PatchCore).

Contrairement à l'auto-encodeur (`model.py`/`train.py`), cette approche ne reconstruit
rien : elle compare les caractéristiques locales (patches) de chaque image à une
banque de patches issus des pièces saines, extraites par un réseau **pré-entraîné et
gelé**. Aucune rétropropagation sur nos données — extraction de features + calcul de
distances uniquement, de l'ordre de quelques minutes.
"""

from __future__ import annotations

import numpy as np
from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input

_LAYER_SHALLOW = "conv3_block4_out"
_LAYER_DEEP = "conv4_block6_out"


def build_feature_extractor(image_size: tuple[int, int] = (256, 256)) -> keras.Model:
    """Construit l'extracteur de features : ResNet50 pré-entraîné (ImageNet), gelé.

    Décision : rester dans l'écosystème Keras/TensorFlow déjà en place
    (`keras.applications.ResNet50`) plutôt qu'ajouter torchvision/timm (PyTorch)
    comme nouvelle dépendance lourde — aucun changement de `pyproject.toml`.

    Décision : sorties = deux couches intermédiaires, `conv3_block4_out` (32×32×512,
    équivalent de la couche "layer2" de l'article PatchCore original) et
    `conv4_block6_out` (16×16×1024, équivalent "layer3"). Pourquoi ces deux couches :
    des features de milieu de réseau, ni trop bas niveau (bords/couleurs, peu
    discriminant) ni trop haut niveau (trop abstrait, perd la localisation spatiale
    fine nécessaire pour repérer un petit défaut) — le compromis standard de
    l'approche PatchCore.

    Nécessite un téléchargement unique des poids ImageNet (≈94 Mo, mis en cache dans
    `~/.keras/models`) — un accès réseau la première fois, pas un entraînement.
    """
    base = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=(*image_size, 3)
    )
    base.trainable = False
    return keras.Model(
        inputs=base.input,
        outputs=[base.get_layer(_LAYER_SHALLOW).output, base.get_layer(_LAYER_DEEP).output],
    )


def extract_patch_features(extractor: keras.Model, images: np.ndarray) -> np.ndarray:
    """Extrait les features de patches (deux couches concaténées) pour un lot d'images.

    Décision : prétraitement `preprocess_input(images * 255.0)` **obligatoire**.
    Pourquoi : le pipeline du projet charge les images en `float32` dans [0, 1]
    (`dataset.py`), mais `preprocess_input` de ResNet50 attend du [0, 255] RGB qu'il
    convertit en BGR et centre sur les statistiques ImageNet — sans ce prétraitement,
    les features produites sont dégradées (quasi constantes) et inutilisables.

    Décision : ré-échantillonner la sortie de la couche profonde (16×16) à la
    résolution de la couche superficielle (32×32) puis concaténer sur l'axe des
    canaux → un tenseur (n_images, 32, 32, 1536). Simplification documentée : l'article
    original ajoute aussi un moyennage local 3×3 par position ("locally aware patch
    features") en plus de la concaténation multi-couches ; ce moyennage est omis ici
    par souci de simplicité — la concaténation multi-échelle seule apporte déjà du
    contexte spatial, mais ce n'est pas une reproduction fidèle à 100 % de l'article.
    """
    preprocessed = preprocess_input(images * 255.0)
    shallow, deep = extractor.predict(preprocessed, verbose=0)
    target_shape = shallow.shape[1:3]
    deep_resized = np.stack(
        [resize(image, target_shape, order=1, preserve_range=True) for image in deep]
    )
    return np.concatenate([shallow, deep_resized], axis=-1)


def build_memory_bank(
    patch_features: np.ndarray,
    coreset_size: int = 5000,
    pre_sample_size: int = 50_000,
    seed: int = 42,
) -> np.ndarray:
    """Construit la banque de mémoire des patches sains, réduite par coreset glouton.

    Décision : coreset glouton (k-center) — sélection itérative du vecteur le plus
    éloigné de la banque déjà sélectionnée, avec des distances minimales mises à jour
    de façon **incrémentale** (jamais de matrice de distances N×N complète : pour
    ~227 images d'entraînement × 1024 patches ≈ 232 000 vecteurs, une matrice N×N
    dépasserait 50 To et ferait planter la machine). Les distances sont calculées via
    la décomposition `‖a-b‖² = ‖a‖² + ‖b‖² - 2·a·b` pour profiter de produits
    matriciels optimisés (BLAS) plutôt que des soustractions explicites répétées.

    Décision : sous-échantillonnage aléatoire préalable à `pre_sample_size` (défaut
    50 000) avant le coreset glouton, si la banque brute est plus grande. Pourquoi :
    le coreset glouton coûte `O(coreset_size × N)` — sur la banque brute complète
    (~232 000 vecteurs), `coreset_size=5000` prendrait de l'ordre de 10 minutes de
    calcul ; un sous-échantillonnage aléatoire préalable (rapide, `O(N)`) ramène ce
    coût à quelques minutes sans changer la méthode de sélection elle-même. Un
    sous-tirage aléatoire de 50 000 patches parmi ~227 images reste représentatif de
    l'apparence normale (~220 patches par image en moyenne).

    Pourquoi cette approche plutôt que celle de l'article : PatchCore original utilise
    une projection aléatoire pour accélérer le coreset sur des jeux de données bien
    plus grands ; ce jeu de données étant petit (quelques centaines de milliers de
    vecteurs), un sous-échantillonnage aléatoire direct suivi d'un coreset glouton
    reste praticable sans cette optimisation plus complexe — un choix pragmatique, pas
    une fidélité totale à l'article. `coreset_size=5000` (~2 % de la banque brute) est
    un compromis coût/fidélité, ajustable si le calcul reste trop lent en pratique.
    """
    n_images, height, width, channels = patch_features.shape
    bank = patch_features.reshape(n_images * height * width, channels)
    if len(bank) <= coreset_size:
        return bank

    rng = np.random.default_rng(seed)
    if len(bank) > pre_sample_size:
        pre_sample_idx = rng.choice(len(bank), size=pre_sample_size, replace=False)
        bank = bank[pre_sample_idx]

    bank_sq_norm = np.sum(bank**2, axis=1)
    first_idx = int(rng.integers(len(bank)))
    selected = [first_idx]
    min_dist_sq = bank_sq_norm + bank_sq_norm[first_idx] - 2.0 * (bank @ bank[first_idx])

    for _ in range(coreset_size - 1):
        next_idx = int(np.argmax(min_dist_sq))
        selected.append(next_idx)
        dist_sq_to_new = bank_sq_norm + bank_sq_norm[next_idx] - 2.0 * (bank @ bank[next_idx])
        min_dist_sq = np.minimum(min_dist_sq, dist_sq_to_new)

    return bank[selected]


def patch_anomaly_map(query_features: np.ndarray, memory_bank: np.ndarray) -> np.ndarray:
    """Carte d'anomalie par patch = distance euclidienne au plus proche voisin dans la
    banque de mémoire, pour chaque image.

    Décision : traiter les images une par une (pas toutes en une seule matrice de
    requêtes) pour garder un pic mémoire raisonnable — (n_patches_image × taille du
    coreset) plutôt que (n_patches_total × taille du coreset), qui serait bien plus
    volumineux pour un grand jeu de test.
    """
    n_images, height, width, channels = query_features.shape
    bank_sq_norm = np.sum(memory_bank**2, axis=1)
    maps = np.empty((n_images, height, width), dtype=np.float32)
    for i in range(n_images):
        queries = query_features[i].reshape(height * width, channels)
        query_sq_norm = np.sum(queries**2, axis=1, keepdims=True)
        dist_sq = query_sq_norm + bank_sq_norm[np.newaxis, :] - 2.0 * (queries @ memory_bank.T)
        dist_sq = np.maximum(dist_sq, 0)
        maps[i] = np.sqrt(dist_sq.min(axis=1)).reshape(height, width)
    return maps


def image_score_from_patches(patch_map: np.ndarray) -> np.ndarray:
    """Score d'anomalie par image = distance maximale sur la carte de patches.

    Pourquoi le maximum (pas la moyenne) : agrégation standard de l'approche
    PatchCore — un seul patch très anormal (le défaut) suffit à signaler l'image,
    contrairement à la moyenne globale de l'auto-encodeur qui dilue un défaut
    localisé (cf. diagnostic notebook 07).
    """
    return patch_map.reshape(len(patch_map), -1).max(axis=1)


def upsample_patch_map(patch_map: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Ré-échantillonne une carte de patches à la résolution de l'image d'origine.

    Pourquoi : pour rester comparable aux cartes d'erreur de l'auto-encodeur
    (`error_maps`, résolution image complète) — même convention que `heatmap()`
    dans `anomaly.py` — et permettre de réutiliser `pixel_auroc`/`load_masks` sans
    modification.
    """
    return np.stack([resize(m, image_size, order=1, preserve_range=True) for m in patch_map])
