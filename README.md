## Spotify Songs Analysis

## Équipe
- **Membres** : Farah Charfeddine, Ranim Khalifa, Chaima Ben Radhia, Asma Jleili
- **Tuteur** : Haythem Ghazouani

---

## Description du Projet
Ce projet vise à analyser et prédire des caractéristiques de chansons Spotify en utilisant un pipeline Machine Learning complet. L'objectif est de mettre en œuvre une chaîne de traitement de données robuste, de l'exploration à la mise en production, en suivant les bonnes pratiques MLOps.

---

## Objectifs
- Analyser les caractéristiques audio des chansons Spotify
- Prédire le **mode musical** (majeur/mineur) à partir des features audio
- Mettre en place un pipeline ML complet avec tracking des expériences

---

## Dataset
- **Source** : Hugging Face - ConquestAce/spotify-songs
- **Taille initiale** : ~28 MB, 161,455 lignes, 21 colonnes
- **Après nettoyage** : 45,952 lignes
- **Features principales** : 
  - *Audio features* : danceability, energy, acousticness, tempo, valence, instrumentalness, liveness, loudness, speechiness, mode, key
  - *Métadonnées* : track_id, name, artist, album, duration_ms
- **Target finale** : `mode` (0 = mineur, 1 = majeur)

### Distribution de la target
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

---

## Résultats de la Modélisation

### Performance des modèles
| Modèle | Train Accuracy | Test Accuracy | F1-score | AUC |
|--------|---------------|---------------|----------|-----|
| **RandomForest** | **100.00%** | **98.11%** | **0.9846** | **0.999** |
| XGBoost | 92.96% | 89.99% | 0.9223 | 0.968 |
| LogisticRegression | 60.33% | 60.03% | 0.7425 | 0.575 |
| Dummy (baseline) | - | 61.20% | - | - |

## Meilleur modèle : RandomForest
```python
Paramètres optimaux :
- n_estimators: 200
- max_depth: None
- min_samples_split: 5

