# Prédiction Automatique de Landmarks Anatomiques (AFIDs) par SIFT-RANSAC Multi-Atlas

Ce projet implémente un pipeline complet pour la prédiction de points de repère anatomiques (Landmarks/AFIDs) sur des images IRM cérébrales. La méthode utilise l'extraction de features SIFT, un recalage affine robuste via RANSAC, et une fusion multi-atlas basée sur la qualité des correspondances (Top-K).

## Prérequis

* **Python 3.8+**
* **Exécutable SIFT :** Le binaire `featMatchMultiple` (non inclus) doit être présent.
* **Dépendances Python :**
  ```bash
  pip install -r requirements.txt
  ```
  *(Contenu requis : `numpy`, `scikit-learn`, `matplotlib`, `argparse`)*

---

## Structure des Données

Le pipeline attend une structure de fichiers standardisée (type BIDS simplifié) :
* **Images/Features :** `sub-ID_T1w.key` (Fichiers de features SIFT extraits)
* **Vérité Terrain (GT) :** `sub-ID_space-T1w_desc-groundtruth_afids.fcsv` (Uniquement requis pour la validation)

---

## Description des Scripts et Utilisation

Le projet contient 3 scripts principaux, correspondant aux étapes de **Génération**, **Validation**, et **Prédiction**.

### 1. `generate_matches.py` (Génération des Matches)

**Description :**
Ce script automatise l'exécution de l'exécutable C `featMatchMultiple`. Il prend un dossier de patients cibles et un dossier d'atlas, calcule toutes les paires de correspondances possibles, et nettoie automatiquement les nombreux fichiers temporaires générés par le binaire. Il gère intelligemment les doublons et évite de calculer `Patient A` vs `Patient A`.

**Arguments :**
* `--patients` : Dossier des `.key` cibles.
* `--atlases` : Dossier des `.key` sources.
* `--output` : Dossier des `.matches.txt` de sortie.
* `--exe` : Emplacement de l'executable featMatchMultiple.
* `--no_rotation` : (Optionnel) Ajoute le flag `-r-` pour désactiver l'invariance en rotation.

**Exemple d'utilisation :**
```bash
python generate_matches.py \
  --patients "data/AFIDs-OASIS" \
  --atlases "data/AFIDs-HCP" \
  --output "Resultats_Matches_HCP_vers_OASIS" \
  --exe "./featExtract1.6/featMatchMultiple.mac"
```

---

### 2. `top_K.py` (Validation & Analyse)

**Description :**
Ce script sert à valider la méthode lorsque la Vérité Terrain (GT) est connue. Il parcourt les dossiers de matches générés par le script précédent, applique l'algorithme RANSAC pour trouver la transformation, et compare la position prédite avec la position réelle (GT). Il génère un fichier CSV statistique et une courbe montrant l'erreur moyenne (TRE) en fonction du nombre d'atlas utilisés ($K$).

**Arguments principaux :**
* `results` : Le dossier contenant les fichiers `.txt` générés à l'étape 1.
* `gt_target` / `gt_source` : Dossiers contenant les fichiers `.fcsv`.
* `--name_png` : Nom du graphique à générer.

**Exemple d'utilisation :**
```bash
python analyze_topK.py \
  "Resultats_Matches_HCP_vers_OASIS" \
  "data/AFIDs-OASIS" \
  "data/AFIDs-HCP" \
  --output_dir "Analyses_Stats" \
  --name_csv "stats_validation.csv" \
  --name_png "courbe_erreur_K.png"
```

---

### 3. `predict_landmarks.py` (Prédiction / Production)

**Description :**
C'est le script final d'inférence. Il permet de prédire les landmarks sur de **nouveaux patients** (dont on ne connait pas la GT).
Il effectue le pipeline complet en une seule commande :
1. Matching SIFT contre la base d'atlas.
2. Calcul des matrices de transformation via RANSAC.
3. Sélection des $K$ meilleurs atlas (basé sur le nombre d'inliers).
4. Fusion des résultats et génération du fichier `.fcsv`.

**Arguments principaux :**
* `--input` : Un fichier `.key` unique ou un dossier de `.key`.
* `--atlas_dir` : Emplacement du dossier des atlas & de leurs GroundTruth.
* `--k` : Nombre d'atlas à utiliser pour la fusion (Défaut : 12).
* `--threshold` : Seuil de tolérance RANSAC en mm (Défaut : 15.0).

**Exemple d'utilisation :**
```bash
python predict_landmarks.py \
  --input "Nouveaux_Patients/sub-001.key" \
  --atlas_dir "data/AFIDs-HCP" \
  --output "Predictions_Finales" \
  --exe "./featExtract1.6/featMatchMultiple.mac" \
  --k 12
```

---

## Méthodologie

* **RANSAC :** Utilisé avec un raffinement par moindres carrés sur les inliers pour garantir une transformation affine précise malgré le bruit.
* **Top-K Fusion :** Au lieu de faire la moyenne de tous les atlas, l'algorithme ne conserve que les $K$ atlas ayant le plus de correspondances valides (inliers), et calcule la **médiane** spatiale des prédictions pour éliminer les outliers.

## Auteur
Mathys Claudel & Antonin Chauvet
Professeur : Matthew Toews
Projet réalisé dans le cadre du cours SYS818.