import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import joblib

# 1. CHARGEMENT
df = pd.read_csv("data/raw/spotify_songs.csv", low_memory=False)
initial_count = len(df)
print(f"Avant nettoyage : {initial_count} lignes")

# 2. NETTOYAGE

# 2.1 Supprimer les doublons
df = df.drop_duplicates()
print(f"Doublons supprimés : {initial_count - len(df)} lignes en moins")

# 2.2 Analyser les valeurs manquantes
print("Analyse des valeurs manquantes:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Colonne': missing.index,
    'Valeurs manquantes': missing.values,
    'Pourcentage': missing_pct.values
})
missing_df = missing_df[missing_df['Valeurs manquantes'] > 0].sort_values('Pourcentage', ascending=False)
if len(missing_df) > 0:
    print(missing_df)
else:
    print("Aucune valeur manquante trouvée !")

# 2.3 Traiter les valeurs manquantes 
# Liste des colonnes critiques (celles qu'on veut absolument garder)
critical_cols = ['danceability', 'energy', 'key', 'loudness', 'mode', 
                 'speechiness', 'acousticness', 'instrumentalness', 
                 'liveness', 'valence', 'tempo', 'duration_ms', 'genre']

# Vérifier quelles colonnes critiques existent
existing_critical = [col for col in critical_cols if col in df.columns]

# Supprimer les lignes qui n'ont PAS de genre (target)
if 'genre' in df.columns:
    before = len(df)
    df = df[df['genre'].notna()]
    print(f"Lignes sans genre supprimées : {before - len(df)}")

# Pour les autres colonnes critiques, remplir avec la médiane
for col in existing_critical:
    if col != 'genre' and col in df.columns and df[col].isnull().sum() > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"{col}: {df[col].isnull().sum()} valeurs remplacées par la médiane ({median_val:.3f})")

# 2.4 Supprimer les lignes avec encore des NaN (si elles existent)
remaining_na = df.isnull().sum().sum()
if remaining_na > 0:
    before = len(df)
    df = df.dropna()
    print(f"Lignes restantes avec NaN supprimées : {before - len(df)}")

# 2.5 Convertir les types numériques
numeric_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'duration_ms', 'key', 'mode']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 2.6 Supprimer les outliers évidents
print("Suppression des outliers:")

# Durée trop longue (> 1 heure)
before = len(df)
df = df[df['duration_ms'] < 3600000]
print(f"Chansons > 1h supprimées : {before - len(df)}")

# Durée trop courte (< 30 secondes)
before = len(df)
df = df[df['duration_ms'] > 30000]
print(f"Chansons < 30s supprimées : {before - len(df)}")

# Tempo raisonnable (entre 40 et 220 BPM)
if 'tempo' in df.columns:
    before = len(df)
    df = df[(df['tempo'] > 40) & (df['tempo'] < 220)]
    print(f"Tempo hors norme supprimés : {before - len(df)}")

# 3. FEATURE ENGINEERING

# 3.1 Durée en minutes
df['duration_min'] = df['duration_ms'] / 60000
print("Créé : duration_min")

# 3.2 Ratio énergie/acoustique
df['energy_acoustic_ratio'] = df['energy'] / (df['acousticness'] + 0.001)
print("Créé : energy_acoustic_ratio")

# 3.3 Produit danceability * valence
df['dance_valence'] = df['danceability'] * df['valence']
print("Créé : dance_valence")

# 3.4 Différence énergie - acousticness
df['energy_minus_acoustic'] = df['energy'] - df['acousticness']
print("Créé : energy_minus_acoustic")

# 3.5 Combinaison key + mode (pour plus tard)
df['key_mode'] = df['key'].astype(str) + '_' + df['mode'].astype(str)
print("Créé : key_mode")

# 4. PRÉPARATION POUR LE MODÈLE

feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence',
    'tempo', 'duration_min', 'energy_acoustic_ratio',
    'dance_valence', 'energy_minus_acoustic'
]

X = df[feature_cols]
y = df['genre']

print(f"Features sélectionnées : {len(feature_cols)}")
print(f"Target : {y.nunique()} genres")

# 5. SPLIT 

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train : {X_train.shape[0]} chansons")
print(f"Test  : {X_test.shape[0]} chansons")

# 6. NORMALISATION (StandardScaler)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir en DataFrame pour sauvegarde
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols)


os.makedirs("data/processed", exist_ok=True)

# Sauvegarder les données normalisées
X_train_scaled_df.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv("data/processed/X_test_scaled.csv", index=False)
pd.DataFrame(y_train).to_csv("data/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("data/processed/y_test.csv", index=False)

# Sauvegarder aussi les données brutes (optionnel)
X_train.to_csv("data/processed/X_train_raw.csv", index=False)
X_test.to_csv("data/processed/X_test_raw.csv", index=False)

# Sauvegarder le scaler pour l'utiliser plus tard (important pour l'API !)
joblib.dump(scaler, "data/processed/scaler.pkl")


