"""Entraînement de l'auto-encodeur (cible = entrée) avec suivi MLflow."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from tensorflow import keras

from indusense.vision.augment import augment_image, build_augmentation_pipeline
from indusense.vision.dataset import DEFAULT_IMAGE_SIZE, load_good_images, train_val_split
from indusense.vision.model import compression_ratio, ssim_loss

ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "ml"
MODEL_PATH = ARTIFACTS_DIR / "models" / "autoencoder_pill.keras"
MLFLOW_TRACKING_URI = f"sqlite:///{(ARTIFACTS_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_LOCATION = (ARTIFACTS_DIR / "mlflow_artifacts").resolve().as_uri()
EXPERIMENT_NAME = "indusense-vision-autoencoder"


def build_training_set(
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    n_aug: int = 4,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Construit `(x_train, x_val)` : saines d'entraînement augmentées + validation brute.

    Décision : `x_val` n'est **jamais** augmenté. Pourquoi : la validation sert à
    calibrer le seuil de détection (étape 3), qui doit refléter la distribution réelle
    des pièces saines — l'augmenter fausserait ce calibrage (même logique de fuite que
    le split train/val de la partie 1). Seul `x_train` est enrichi via le pipeline
    Albumentations de la partie 1 (`build_augmentation_pipeline`), qui exclut déjà les
    flips (gravure « FF » non symétrique, cf. notebook 02).
    """
    good_train = load_good_images("train", image_size)
    train_raw, val_good = train_val_split(good_train, val_fraction=val_fraction, seed=seed)

    pipeline = build_augmentation_pipeline()
    augmented = [
        result["image"]
        for image in train_raw
        for result in augment_image(image, pipeline, n=n_aug)
    ]
    x_train = np.concatenate([train_raw, np.stack(augmented)], axis=0)
    return x_train, val_good


def train_autoencoder(
    model: keras.Model,
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    loss: str = "mse",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    run_name: str | None = None,
    log_mlflow: bool = True,
) -> tuple[keras.Model, Any, float]:
    """Entraîne l'auto-encodeur (cible = entrée) et journalise le run dans MLflow.

    Décision : `EarlyStopping` sur `val_loss` (patience 5, poids restaurés). Pourquoi :
    le jeu d'entraînement est minuscule (~227 saines, avant augmentation), le risque
    de sur-apprentissage est élevé et rien ne garantit un nombre d'epochs optimal a
    priori. Décision : suivi MLflow explicitement exigé par l'énoncé (étape 2) — params,
    métriques par epoch et modèle sont journalisés dans un run.

    Décision : la **durée d'entraînement** (secondes, mesurée autour de `model.fit`)
    est mesurée et journalisée systématiquement. Pourquoi : elle est nécessaire pour
    comparer objectivement deux runs (ex. MSE vs SSIM — SSIM est plus coûteux à
    calculer par batch) et pour documenter le coût réel de tout futur réentraînement,
    pas seulement sa qualité.

    Retourne `(model, history, duration_seconds)`.
    """
    loss_fn = ssim_loss if loss == "ssim" else "mse"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_fn)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]

    run_ctx = nullcontext()
    if log_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        _ensure_experiment(EXPERIMENT_NAME, MLFLOW_ARTIFACT_LOCATION)
        mlflow.set_experiment(EXPERIMENT_NAME)
        run_ctx = mlflow.start_run(run_name=run_name or f"autoencoder-{loss}")

    with run_ctx:
        if log_mlflow:
            mlflow.log_params(
                {
                    "loss": loss,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "latent_channels": model.get_layer("latent").output.shape[-1],
                    "compression_ratio": compression_ratio(model),
                    "n_train": len(x_train),
                    "n_val": len(x_val),
                }
            )
        start_time = time.perf_counter()
        history = model.fit(
            x_train,
            x_train,
            validation_data=(x_val, x_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )
        duration_seconds = time.perf_counter() - start_time
        if log_mlflow:
            for epoch, (train_loss, val_loss) in enumerate(
                zip(history.history["loss"], history.history["val_loss"], strict=True)
            ):
                mlflow.log_metrics({"loss": train_loss, "val_loss": val_loss}, step=epoch)
            mlflow.log_metric("train_duration_seconds", duration_seconds)
            mlflow.tensorflow.log_model(model, name="model")

    return model, history, duration_seconds


def _ensure_experiment(name: str, artifact_location: str) -> None:
    """Crée l'expérience MLflow avec un artifact store explicite si elle n'existe pas."""
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=artifact_location)


def save_model(model: keras.Model, path: Path = MODEL_PATH) -> None:
    """Sauvegarde le modèle au format Keras natif — source de vérité pour les autres notebooks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_trained_model(path: Path = MODEL_PATH) -> keras.Model:
    """Recharge le modèle sauvegardé (gère la perte `ssim_loss` si utilisée à l'entraînement)."""
    return keras.models.load_model(path, custom_objects={"ssim_loss": ssim_loss})
