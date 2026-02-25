# Python-for-Data-Science-Project
# Spotify Songs Analysis

## Équipe
- Members : [Farah Charfeddine - Ranim Khalifa - Chaima Ben Radhia - Asma Jleili]
- Tuteur : Haythem Ghazouani

## Description du Projet
Ce projet vise à analyser et prédire des caractéristiques de chansons Spotify en utilisant un pipeline Machine Learning complet. L'objectif est de mettre en œuvre une chaîne de traitement de données robuste, de l'exploration à la mise en production.

## Objectifs
- Analyser les caractéristiques audio des chansons (danceability, energy, valence, etc.)
- Prédire : Le genre musical (classification)
  
## Dataset
- Source : Hugging Face - ConquestAce/spotify-songs
- Taille : ~28 MB, 161455 lignes, 21 colonnes
- Features principales : 
  *Audio features* : danceability, energy, acousticness, tempo, valence, instrumentalness, liveness, loudness, speechinees, mode, key, time_signature
  *Métadonnées* : track_id, name, artist, album, duration_ms, popularity
  *Target* : genre

## Roadmap (7 semaines)
### Phase 1 : Fondations (Semaine 1 - En cours)
- [x] Choix et validation du dataset
- [x] Structure GitHub
- [x] README et documentation
- [x] Analyse Exploratoire des Données (EDA)
- [ ] Premier nettoyage des données
### Phase 2 : Pipeline ML (Semaines 2-3)
- [ ] Prétraitement avancé et gestion du déséquilibre (SMOTE)
- [ ] Feature engineering
- [ ] Modélisation (Random Forest, XGBoost)
- [ ] Optimisation hyperparamètres (GridSearchCV)
- [ ] Tracking avec MLflow
### Phase 3 : Déploiement (Semaines 4-7)
- [ ] API FastAPI
- [ ] Dashboard React
- [ ] Conteneurisation Docker
- [ ] CI/CD (optionnel)

## Installation

```bash
# Cloner le repository
git clone https://github.com/votre-nom/Python-for-Data-Science-Project.git
cd Python-for-Data-Science-Project

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
