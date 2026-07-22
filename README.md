# AELION — Deep Learning IA (TP B6)

Détection de défauts visuels sur pièces industrielles par **auto-encodeur CNN**
(apprentissage non supervisé : reconstruire uniquement des pièces saines, puis
utiliser l'erreur de reconstruction comme score d'anomalie). Catégorie MVTec AD
retenue : `pill`.

Énoncés complets :
- Partie 1 (données) : [`TP_B6_deep_learning_1_donnees.md`](TP_B6_deep_learning_1_donnees.md)
- Partie 2 (auto-encodeur) : [`TP_B6_deep_learning_2_auto_encodeur.md`](TP_B6_deep_learning_2_auto_encodeur.md)

**Statut : parties 1 et 2 terminées.**

## Mise en route

Environnement géré par **uv** (Python **3.13.x**, cf. `.python-version`).

```powershell
# Dépendances Deep Learning (tensorflow, albumentations, opencv, scikit-image,
# pillow, mlflow, scikit-learn)
uv sync --group dl

# Notebooks
uv run jupyter lab
```

## Structure

```
pill/                       # dataset MVTec AD (train/good, test/*, ground_truth/*)
src/indusense/vision/
├── dataset.py                # chargement, resize, normalisation, split train/val
├── augment.py                # pipeline d'augmentation Albumentations
├── model.py                  # architecture de l'auto-encodeur CNN
├── train.py                  # entraînement, suivi MLflow, sauvegarde du modèle
└── anomaly.py                # score d'anomalie, seuil, AUROC, heatmaps
notebooks/
├── 01_chargement_donnees.ipynb        # partie 1 — chargement, resize, normalisation, split
├── 02_augmentation.ipynb              # partie 1 — pipeline Albumentations
├── 03_conception_autoencodeur.ipynb   # partie 2 — architecture, ratio de compression
├── 04_entrainement.ipynb              # partie 2 — entraînement (MSE + SSIM), MLflow
├── 05_score_seuil.ipynb               # partie 2 — score d'anomalie, calibration du seuil
└── 06_heatmaps_evaluation.ipynb       # partie 2 — heatmaps, AUROC, matrice de confusion
reports/figures/            # figures livrables
artifacts/ml/               # modèle entraîné (.keras) + suivi MLflow (gitignoré)
```

## Résultats clés (partie 2)

- Auto-encodeur pleinement convolutionnel, bottleneck 16 canaux (32×32×16) —
  **ratio de compression ×12**.
- Perte retenue pour l'entraînement : **MSE** (cohérente avec le score d'anomalie,
  lui-même basé sur l'erreur de reconstruction MSE) ; un run **SSIM** est conservé à
  titre de comparaison.
- Seuil de détection calibré sur les pièces saines de validation (percentile 99).
- **AUROC image-level 0,688** / **AUROC pixel-level 0,832** : l'agrégation du score
  par moyenne sur l'image entière dilue le signal des défauts localisés — analyse
  détaillée au notebook 06.
