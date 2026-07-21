# AELION — Deep Learning IA (TP B6)

Détection de défauts visuels sur pièces industrielles par **auto-encodeur CNN**
(apprentissage non supervisé : reconstruire uniquement des pièces saines).

Énoncé complet : voir [`TP_B6_deep_learning_donnees.md`](TP_B6_deep_learning_donnees.md).

**Partie 1 (en cours) : préparation des données.** Catégorie MVTec AD retenue : `pill`.

## Mise en route

Environnement géré par **uv** (Python **3.13.x**, cf. `.python-version`).

```powershell
# Dépendances Deep Learning (tensorflow, albumentations, opencv, scikit-image, pillow)
uv sync --group dl

# Notebooks
uv run jupyter lab
```

## Structure

```
pill/                       # dataset MVTec AD (train/good, test/*, ground_truth/*)
src/indusense/vision/       # dataset.py (chargement), augment.py (augmentation Albumentations)
notebooks/
├── 01_chargement_donnees.ipynb  # étape 1 : chargement, resize, normalisation, split
└── 02_augmentation.ipynb        # étape 2 : pipeline Albumentations
reports/figures/            # figures livrables
```
