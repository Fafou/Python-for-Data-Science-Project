## Spotify Mode Classification - Machine Learning Project

## Equipe
- **Membres** : Farah Charfeddine, Ranim Khalifa, Chaima Ben Radhia, Asma Jleili
- **Tuteur** : Haythem Ghazouani

---

## Description du Projet
Ce projet vise a classifier le **mode musical** (majeur/mineur) des chansons Spotify a partir de leurs caracteristiques audio. L'objectif est de mettre en oeuvre un pipeline Machine Learning complet, de l'exploration des donnees au deploiement, en suivant les bonnes pratiques MLOps.

---

## Objectifs
- Analyser les caracteristiques audio des chansons Spotify
- Predire le **mode musical** (majeur/mineur) a partir des features audio
- Mettre en place un pipeline ML complet avec tracking des experiences

---

## Dataset
- **Source** : Hugging Face - ConquestAce/spotify-songs
- **Taille initiale** : 28 MB, 161,455 lignes, 21 colonnes
- **Apres nettoyage** : 45,952 lignes
- **Features principales** : 
  - *Audio features* : danceability, energy, acousticness, tempo, valence, instrumentalness, liveness, loudness, speechiness, mode, key
  - *Metadonnees* : track_id, name, artist, album, duration_ms
- **Target finale** : `mode` (0 = mineur, 1 = majeur)

## Distribution de la target
- **Mode 0 (mineur)** : 17,814 chansons (38.8%)
- **Mode 1 (majeur)** : 28,138 chansons (61.2%)

---

## Roadmap (7 semaines)

### Phase 1 : Fondations (Semaine 1 - COMPLÉTÉE)
- [x] Choix et validation du dataset (Spotify Songs)
- [x] Structure GitHub
- [x] README et documentation
- [x] Analyse Exploratoire des Données (EDA)
- [x] Premier nettoyage des données

### Phase 2 : Pipeline ML (Semaine 2 - COMPLÉTÉE)
- [x] Nettoyage avancé des données (doublons, valeurs manquantes, outliers)
- [x] Feature engineering (création de 5 nouvelles features)
- [x] Split argumenté (80/20 avec stratification)
- [x] Normalisation des données (StandardScaler)
- [x] Sauvegarde des données préparées

### Phase 2 : Pipeline ML (Semaine 3 - COMPLÉTÉE)
- [x] Changement de stratégie : passage de la classification des **genres** (10% de précision) à la classification du **mode** (majeur/mineur)
- [x] Chargement des données préparées pour le mode
- [x] Modélisation avec 3 algorithmes :
  - RandomForest (GridSearchCV)
  - XGBoost (GridSearchCV)
  - LogisticRegression (GridSearchCV)
- [x] Tracking des expériences avec **MLflow**
- [x] Évaluation des performances
- [x] Sauvegarde du meilleur modèle
- [x] Génération des graphiques (matrices de confusion, courbes ROC, feature importance)

### Phase 3 : Deploiement (Semaine 4 - COMPLETEE)
- [x] API FastAPI avec le modele RandomForest
- [x] Endpoints : /predict, /predict_batch, /health
- [x] Documentation Swagger automatique

---

## Resultats de la Modelisation

### Performance des modeles
| Modele | Train Accuracy | Test Accuracy | F1-score | AUC |
|--------|---------------|---------------|----------|-----|
| **RandomForest** | **100.00%** | **98.24%** | **0.9857** | **0.999** |
| XGBoost | 92.96% | 89.99% | 0.9223 | 0.968 |
| LogisticRegression | 60.33% | 60.03% | 0.7425 | 0.575 |
| Dummy (baseline) | - | 61.20% | - | - |

### Meilleur modele : RandomForest
## Parametres optimaux :
- n_estimators: 200
- max_depth: None
- min_samples_split: 5
- class_weight: balanced

