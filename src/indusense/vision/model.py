"""Auto-encodeur CNN pour la détection d'anomalies (`pill`) : reconstruire le sain."""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(
    image_size: tuple[int, int] = (256, 256),
    channels: int = 3,
    latent_channels: int = 16,
) -> keras.Model:
    """Construit l'auto-encodeur : 3 conv (encodeur) + 3 déconv (décodeur), sortie sigmoid.

    Décision : goulot d'étranglement à 16 canaux (ratio de compression ≈ ×12, cf.
    `compression_ratio`) plutôt qu'un goulot plus large. Pourquoi : un ratio trop
    proche de 1 laisserait le modèle apprendre une quasi-identité et bien reconstruire
    aussi les défauts (le piège central du TP) ; ×12 vise une réelle séparation
    sain/défaut. Architecture pleinement convolutionnelle (pas de `Flatten`+`Dense`)
    pour rester fidèle à la consigne "3 conv + 3 déconv" et garder un `summary()`
    lisible. Modèle retourné **non compilé** : la perte (MSE/SSIM) est un choix
    d'entraînement, pas d'architecture (cf. `train.py`).
    """
    height, width = image_size
    inputs = keras.Input(shape=(height, width, channels), name="image")

    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu", name="enc_conv1")(
        inputs
    )
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu", name="enc_conv2")(x)
    latent = layers.Conv2D(
        latent_channels, 3, strides=2, padding="same", activation="relu", name="latent"
    )(x)

    x = layers.Conv2DTranspose(
        64, 3, strides=2, padding="same", activation="relu", name="dec_deconv1"
    )(latent)
    x = layers.Conv2DTranspose(
        32, 3, strides=2, padding="same", activation="relu", name="dec_deconv2"
    )(x)
    outputs = layers.Conv2DTranspose(
        channels, 3, strides=2, padding="same", activation="sigmoid", name="reconstruction"
    )(x)

    return keras.Model(inputs, outputs, name="pill_autoencoder")


def compression_ratio(model: keras.Model) -> float:
    """Ratio (valeurs en entrée / valeurs dans le goulot), calculé depuis le modèle réel.

    Décision : ne pas coder ce ratio en dur mais le recalculer depuis les vraies
    shapes de `model` — le notebook 03 doit l'obtenir en inspectant le modèle qu'il
    vient de construire, conformément au point de réflexion de l'énoncé ("calculez
    le ratio de compression").
    """
    latent_layer = model.get_layer("latent")
    input_size = math.prod(model.input_shape[1:])
    latent_size = math.prod(latent_layer.output.shape[1:])
    return input_size / latent_size


def ssim_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Perte `1 - SSIM moyen` : plus sensible aux altérations de texture que la MSE.

    Proposée par l'énoncé en alternative à la MSE ("essayez aussi SSIM"), utilisée
    uniquement pour le second run d'entraînement (cf. `train.py`) ; le score
    d'anomalie reste toujours basé sur la MSE pour rester comparable entre les runs
    (cf. `anomaly.py`).
    """
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
