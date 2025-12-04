# Prédiction Automatique de Landmarks Anatomiques (AFIDs) par SIFT-RANSAC Multi-Atlas

Ce projet implémente un pipeline robuste pour la prédiction de points de repère anatomiques (Landmarks/AFIDs) sur des images IRM cérébrales. La méthode repose sur l'extraction de caractéristiques invariantes (SIFT), le calcul de transformations affines robustes (RANSAC) et une stratégie de fusion multi-atlas (Top-K).

## 🚀 Fonctionnalités

* **Matching Robuste :** Génération automatisée de correspondances SIFT entre patients et atlas.
* **Alignement RANSAC :** Calcul de transformation affine avec rejet d'outliers et raffinement par moindres carrés.
* **Fusion Multi-Atlas :** Sélection des $K$ meilleurs atlas (basé sur le nombre d'inliers) et fusion des prédictions par médiane géométrique.
* **Analyse de Performance :** Outils pour évaluer l'erreur (TRE) et tracer des courbes d'influence du paramètre $K$.
* **Interopérabilité :** Sortie des prédictions au format `.fcsv` (compatible 3D Slicer).

## 📋 Prérequis

* **Python 3.8+**
* **Exécutable SIFT :** Le binaire `featMatchMultiple` (non inclus dans ce dépôt) doit être accessible.

### Dépendances Python
Installez les librairies nécessaires via :
```bash
pip install -r requirements.txt