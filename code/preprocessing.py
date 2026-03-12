import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

print("="*60)
print("PREPROCESSING SPOTIFY - TARGET MODE")
print("="*60)

# Créer dossiers
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/models", exist_ok=True)

# 1. CHARGEMENT
df = pd.read_csv("data/raw/spotify_songs.csv", low_memory=False)
initial_count = len(df)
print(f"\nAvant nettoyage : {initial_count} lignes")

# 2. NETTOYAGE
print("\nNETTOYAGE")
print("-"*40)

# 2.1 Supprimer les doublons
df = df.drop_duplicates()
print(f"Doublons supprimés : {initial_count - len(df)}")

# 2.2 Supprimer les valeurs manquantes
df = df.dropna()
print(f"Valeurs manquantes supprimées")

# 2.3 Convertir les types numériques
numeric_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'duration_ms', 'mode']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 2.4 Supprimer les outliers
before = len(df)
df = df[df['duration_ms'] < 3600000]  # < 1h
df = df[df['duration_ms'] > 30000]    # > 30s
df = df[(df['tempo'] > 40) & (df['tempo'] < 220)]
print(f"Outliers supprimés : {before - len(df)}")

print(f"\nAprès nettoyage : {len(df)} lignes")

# 3. FEATURE ENGINEERING
print("\nFEATURE ENGINEERING")
print("-"*40)

# 3.1 Durée en minutes
df['duration_min'] = df['duration_ms'] / 60000
print("Créé: duration_min")

# 4. PRÉPARATION POUR MODE
print("\nPRÉPARATION TARGET MODE")
print("-"*40)

feature_cols = ['danceability', 'energy', 'loudness', 'speechiness',
                'acousticness', 'instrumentalness', 'liveness', 'valence',
                'tempo', 'duration_min']

X = df[feature_cols]
y = df['mode']  # Target = mode (0/1)

print(f"Features: {len(feature_cols)}")
print(f"\nDistribution mode:")
print(f"   Mode 0 (mineur): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"   Mode 1 (majeur): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

# 5. SPLIT
print("\nSPLIT TRAIN/TEST")
print("-"*40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")

# 6. NORMALISATION
print("\nNORMALISATION")
print("-"*40)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Moyennes après normalisation: {X_train_scaled.mean(axis=0).round(3)}")
print(f"Écarts-types après normalisation: {X_train_scaled.std(axis=0).round(3)}")

# 7. SAUVEGARDE
print("\nSAUVEGARDE")
print("-"*40)

pd.DataFrame(X_train_scaled, columns=feature_cols).to_csv("data/processed/X_train_mode.csv", index=False)
pd.DataFrame(X_test_scaled, columns=feature_cols).to_csv("data/processed/X_test_mode.csv", index=False)
pd.DataFrame(y_train).to_csv("data/processed/y_train_mode.csv", index=False)
pd.DataFrame(y_test).to_csv("data/processed/y_test_mode.csv", index=False)
joblib.dump(scaler, "data/models/scaler_mode.pkl")

print("X_train_mode.csv")
print("X_test_mode.csv")
print("y_train_mode.csv")
print("y_test_mode.csv")
print("scaler_mode.pkl")
