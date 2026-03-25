from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import io
import os

# ============================================
# CONFIGURATION
# ============================================

app = FastAPI(title="Spotify Mode Classification API")

# Enable CORS (pour permettre au frontend React d'appeler l'API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemins des modèles
MODEL_PATH = "data/models/best_mode_pipeline.pkl"
SCALER_PATH = "data/models/scaler_mode.pkl"

# Charger le modèle
if os.path.exists(MODEL_PATH):
    pipeline = joblib.load(MODEL_PATH)
    model = pipeline['model']
    feature_names = pipeline['feature_names']
    best_f1 = pipeline['best_f1']
    print(f"Modele charge: RandomForest avec F1 = {best_f1:.4f}")
    print(f"Features attendues: {feature_names}")
else:
    model = None
    feature_names = None
    print(f"Modele non trouve: {MODEL_PATH}")

# ============================================
# DEFINITION DES MODELES DE DONNEES
# ============================================

class SongFeatures(BaseModel):
    """
    Modele Pydantic pour les caracteristiques audio d'une chanson
    (identique aux features utilisees pour l'entrainement)
    """
    danceability: float = Field(..., ge=0, le=1, description="Dansabilite (0-1)")
    energy: float = Field(..., ge=0, le=1, description="Energie (0-1)")
    loudness: float = Field(..., ge=-60, le=0, description="Volume (dB)")
    speechiness: float = Field(..., ge=0, le=1, description="Presence de parole (0-1)")
    acousticness: float = Field(..., ge=0, le=1, description="Caractere acoustique (0-1)")
    instrumentalness: float = Field(..., ge=0, le=1, description="Caractere instrumental (0-1)")
    liveness: float = Field(..., ge=0, le=1, description="Presence de public (0-1)")
    valence: float = Field(..., ge=0, le=1, description="Positivite musicale (0-1)")
    tempo: float = Field(..., ge=40, le=220, description="Tempo (BPM)")
    duration_min: float = Field(..., ge=0.5, le=60, description="Duree en minutes")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/")
def root():
    """
    Page d'accueil de l'API
    """
    return {
        "name": "Spotify Mode Classification API",
        "version": "1.0.0",
        "description": "Predire si une chanson est en mode MAJEUR (1) ou MINEUR (0)",
        "model_info": {
            "type": "RandomForestClassifier",
            "f1_score": best_f1 if model else None,
            "features": feature_names
        },
        "endpoints": {
            "/health": "Verifier l'etat de l'API",
            "/predict": "Prediction pour une seule chanson",
            "/predict_batch": "Prediction pour plusieurs chansons (fichier CSV)"
        }
    }

@app.get("/health")
def health_check():
    """
    Health check endpoint to ensure API is running and model is loaded.
    """
    if model:
        return {
            "status": "ok",
            "model_loaded": True,
            "model_type": "RandomForestClassifier",
            "f1_score": best_f1
        }
    return {
        "status": "degraded",
        "model_loaded": False,
        "message": "Model not loaded"
    }

@app.post("/predict")
def predict_mode(data: SongFeatures):
    """
    Real-time prediction for a single song.
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model pipeline not available")
    
    # Convertir en DataFrame (ordre des colonnes identique a l'entrainement)
    input_df = pd.DataFrame([data.dict()])
    
    # Verifier l'ordre des colonnes
    input_df = input_df[feature_names]
    
    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    
    return {
        "mode_prediction": int(prediction),
        "mode": "Majeur" if prediction == 1 else "Mineur",
        "probabilities": {
            "mineur": float(probability[0]),
            "majeur": float(probability[1])
        }
    }

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Batch prediction endpoint expecting a CSV file.
    Returns the input CSV with two new columns: 'Mode_Prediction' and 'Mode_Probabilities'
    """
    if not model:
        raise HTTPException(status_code=503, detail="Model pipeline not available")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Verifier que toutes les colonnes necessaires sont presentes
        required_cols = feature_names
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise HTTPException(
                status_code=400, 
                detail=f"CSV doit contenir les colonnes: {required_cols}\nColonnes manquantes: {missing_cols}"
            )
        
        # Reordonner les colonnes
        df_input = df[required_cols]
        
        # Predictions
        predictions = model.predict(df_input)
        probabilities = model.predict_proba(df_input)
        
        # Ajouter les resultats
        df['Mode_Prediction'] = predictions
        df['Mode'] = df['Mode_Prediction'].map({0: 'Mineur', 1: 'Majeur'})
        df['Prob_Mineur'] = probabilities[:, 0]
        df['Prob_Majeur'] = probabilities[:, 1]
        
        return df.to_dict(orient='records')
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ============================================
# LANCEMENT
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)