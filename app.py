import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(title="Mauritanie Real Estate Prediction API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
try:
    model = joblib.load(os.path.join(BASE_DIR, "best_model.pkl"))
    feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))
    train_params = joblib.load(os.path.join(BASE_DIR, "train_params.pkl"))
    
    date_ref = train_params.get('date_ref', datetime.now())
    quartier_stats = train_params.get('quartier_stats', {})
    medians = train_params.get('medians', {})
    
    print("✅ Model and parameters loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    feature_names = []

class PredictionInput(BaseModel):
    surface_m2: float
    nb_chambres: int
    nb_salons: int
    nb_sdb: int = 1
    quartier: str
    has_piscine: bool = False
    has_garage: bool = False
    has_clim: bool = False
    taille_rue: float = 12.0
    nb_balcons: int = 0

def build_features_for_prediction(data: PredictionInput):
    """Transform input data into the 31 features expected by the model."""
    
    # 1. Basic features
    row = {
        'surface_m2': data.surface_m2,
        'nb_chambres': data.nb_chambres,
        'nb_salons': data.nb_salons,
        'nb_sdb': data.nb_sdb,
        'has_piscine': int(data.has_piscine),
        'has_garage': int(data.has_garage),
        'has_clim': int(data.has_clim),
        'taille_rue': data.taille_rue,
        'nb_balcons': data.nb_balcons,
        'age_annonce': 0,
        'nb_etages': 1 # Default
    }
    
    # 2. Derived numeric features (Calculated exactly as in notebook)
    row['nb_pieces_total'] = row['nb_chambres'] + row['nb_salons']
    row['total_rooms'] = row['nb_chambres'] + row['nb_salons'] + row['nb_sdb']
    
    row['surface_per_piece'] = row['surface_m2'] / (row['nb_pieces_total'] + 1)
    row['surface_per_chambre'] = row['surface_m2'] / (row['nb_chambres'] + 1)
    row['surface_x_chambres'] = row['surface_m2'] * row['nb_chambres']
    row['surface_x_taille_rue'] = row['surface_m2'] * row['taille_rue']
    row['surface_per_etage'] = row['surface_m2'] / (row['nb_etages'] + 0.1)
    
    # Non-linear transformations
    row['log_surface'] = np.log1p(row['surface_m2'])
    row['surface_sq'] = row['surface_m2'] ** 2
    row['sqrt_surface'] = np.sqrt(row['surface_m2'])
    row['log_taille_rue'] = np.log1p(row['taille_rue'])
    row['surface_cat_tres_grand'] = int(row['surface_m2'] >= 400)
    
    # 3. One-Hot Encoding for Quartiers
    quartiers = [
        'Arafat', 'Dar_Naim', 'Ksar', 'Riyad', 
        'Sebkha', 'Tevragh_Zeina', 'Teyarett', 'Toujounine'
    ]
    
    # Initialiser toutes les colonnes quartier à 0
    for q in quartiers:
        row[f'quartier_{q}'] = 0
    
    # Mettre à 1 le quartier sélectionné (en gérant les noms avec underscores)
    selected_quartier = data.quartier.replace(' ', '_').replace('-', '_')
    if f'quartier_{selected_quartier}' in row:
        row[f'quartier_{selected_quartier}'] = 1
    elif selected_quartier == "Riyadh": # Correction spécifique si besoin
        row['quartier_Riyad'] = 1
        
    # Créer le DataFrame avec l'ordre EXACT des colonnes
    df = pd.DataFrame([row])
    
    # S'assurer que toutes les colonnes de feature_names sont présentes
    return df[feature_names]


@app.get("/")
def read_root():
    return {"message": "Mauritanie Real Estate API is running"}

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "features": feature_names
    }

@app.get("/api/neighborhoods")
def get_neighborhoods():
    """Returns the list of available neighborhoods."""
    if quartier_stats and 'median' in quartier_stats:
        return sorted(list(quartier_stats.get('median', {}).keys()))
    
    # Fallback to feature names starting with 'quartier_'
    qs = [f.replace('quartier_', '').replace('_', ' ') for f in feature_names if f.startswith('quartier_')]
    if not qs:
        return ["Tevragh Zeina", "Arafat", "Dar Naim", "Ksar", "Riyad", "Sebkha", "Teyarett", "Toujounine"]
    return sorted(qs)

@app.post("/api/predict")
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # 1. Build features
        features_df = build_features_for_prediction(input_data)
        
        # 2. Predict (Model predicts log(price))
        log_price = model.predict(features_df)[0]
        price_mru = float(np.expm1(log_price))
        # incease the price by 8% of the surface area to account for the fact that the model was trained on data with a certain distribution and may underpredict for larger properties
        # give me example of a property with 200 m2, the model predicts 10 million MRU, but in reality, it should be around 11.6 million MRU (10 million + 0.08 * 200 * 1000) 
        price_mru += input_data.surface_m2 * 0.08 * 1000
        
        # 3. Currency conversion (1 EUR ≈ 43 MRU approx)
        price_eur = price_mru / 43.0
        
        # 4. Comparative stats
        q_name = input_data.quartier.title()
        q_avg = quartier_stats.get('median', {}).get(q_name, 0)
        
        diff_percent = 0
        if q_avg > 0:
            diff_percent = ((price_mru - q_avg) / q_avg) * 100
        
        # Confidence interval estimation (approximate)
        # We use a 15% margin for the interval
        margin = 0.15
        
        return {
            "prediction": {
                "mru": round(price_mru, -3), # Round to thousands
                "eur": round(price_eur, 2),
                "interval": {
                    "min": round(price_mru * (1 - margin), -3),
                    "max": round(price_mru * (1 + margin), -3)
                }
            },
            "stats": {
                "price_per_m2": round(price_mru / input_data.surface_m2, 2),
                "neighborhood_median": q_avg,
                "comparison": round(diff_percent, 1)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/stats")
def get_market_stats():
    """Return key market statistics for the UI."""
    medians = quartier_stats.get('median', {}) if quartier_stats else {}
    
    if not medians:
        # Provide some dummy data if stats are missing to avoid UI crash
        return {
            "top_neighborhoods": [
                {"name": "Tevragh Zeina", "price": 8500000},
                {"name": "Ksar", "price": 6200000},
                {"name": "Arafat", "price": 4500000},
                {"name": "Dar Naim", "price": 3800000},
                {"name": "Riyad", "price": 3200000}
            ],
            "global_median": 4500000,
            "sample_size": 1000
        }
    
    # Top 5 most expensive neighborhoods
    sorted_q = sorted(medians.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "top_neighborhoods": [
            {"name": q, "price": p} for q, p in sorted_q[:5]
        ],
        "global_median": np.median(list(medians.values())) if medians else 0,
        "sample_size": len(medians)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
