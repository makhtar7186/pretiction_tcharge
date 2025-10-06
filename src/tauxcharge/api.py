from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse

import pandas as pd
from datetime import datetime, timedelta
import os
import logging

from tauxcharge.elastic.elasticsearch import ElasticFetcher
from tauxcharge.model.lstm_regressor import CNNLSTMRegressor
from tauxcharge.preprocessing.sequence_generator import SequenceGenerator
from tauxcharge.preprocessing.custom_preprocessor import CustomPreprocessor

# ---------------------------------------
# Config logs
# ---------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tauxcharge_api")

app = FastAPI(title="TauxCharge Prediction API")

# ---------------------------------------
# Constants
# ---------------------------------------
LOOK_BACK = 2016
PREDICT_HORIZON = 1440
INTERVAL_MINUTES = 5

# ---------------------------------------
# Initialize objects once at startup
# ---------------------------------------
fetcher = ElasticFetcher()
elt_save = ElasticFetcher(index=os.getenv("ES_INDEX_dest"))

regressor = CNNLSTMRegressor()
all_models = regressor.load_model()
liste = regressor.get_models_name()

pre = CustomPreprocessor()
seq_generator = SequenceGenerator(look_back=LOOK_BACK)

# ---------------------------------------
# Pydantic models
# ---------------------------------------
class Prediction(BaseModel):
    timestamp_prediction: str
    name: str
    predicted_for: str
    predicted_taux_de_charge: float

class PredictResponse(BaseModel):
    predictions: List[Prediction]

# ---------------------------------------
# Core prediction function
# ---------------------------------------
def run_prediction() -> List[dict]:
    """
    Retourne TOUJOURS une liste de dict prédictions (évent. vide).
    Chaque dict correspond au schéma de 'Prediction'.
    """
    try:
        # 1) Charger & prétraiter les données réelles
        df_all = fetcher.fetch_last_days(day=15)
        if df_all is None or len(df_all) == 0:
            logger.warning("Aucune donnée récupérée depuis Elasticsearch.")
            return []

        df_all = pre.fit_transform(df_all)
        if "date_mesure" not in df_all.columns or "names" not in df_all.columns:
            logger.error("Colonnes essentielles manquantes après preprocessing.")
            return []

        df_all["date_mesure"] = pd.to_datetime(df_all["date_mesure"], errors="coerce")
        df_all = df_all.dropna(subset=["date_mesure"])
        if df_all.empty:
            logger.warning("DataFrame vide après coercition des dates.")
            return []

        df_all = df_all.sort_values(["names", "date_mesure"])
        now = df_all["date_mesure"].max()

        # 2) Fenêtrage temporel pour les pas de 5 minutes
        minute = (now.minute // INTERVAL_MINUTES + 1) * INTERVAL_MINUTES
        next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)

        pred_records: List[dict] = []
        end = now + timedelta(days=5)

        windows_by_name = {}   # name -> X_seq current (1, LOOK_BACK, F)
        scalers_by_name = {}   # name -> scaler target

        # 3) Boucle de blocs jusqu'à 'end'
        while next_time < end:
            # fin de bloc (cinq jours max)
            horizon_block_end = min(next_time + timedelta(days=5), end)
            steps_in_block = int((horizon_block_end - next_time).total_seconds() // (INTERVAL_MINUTES * 60))

            # Sélection des équipements pour lesquels on a des modèles
            valid_models = set(liste["models_name"].unique()) if "models_name" in liste.columns else set()

            for name in df_all["names"].unique():
                if name not in valid_models:
                    continue

                # --- Bootstrap de la fenêtre si première fois
                if name not in windows_by_name:
                    df_name = df_all[df_all["names"] == name].copy().sort_values("date_mesure")
                    seq_data = seq_generator.fit_transform(df_name)

                    # Sécurité sur la sortie du SequenceGenerator
                    if not isinstance(seq_data, dict) or "X_seq" not in seq_data or seq_data["X_seq"].shape[0] == 0:
                        logger.info(f"Pas assez de données pour '{name}' (bootstrap).")
                        continue

                    windows_by_name[name] = seq_data["X_seq"][-1].reshape(1, LOOK_BACK, -1)
                    scalers_by_name[name] = seq_data.get("scaler", None)

                # --- Prévision du bloc courant
                X_seq_cur = windows_by_name[name]
                scaler = scalers_by_name[name]

                try:
                    rec_out, X_seq_next = regressor.forecast_block(
                        regressor=all_models[name],
                        X_seq=X_seq_cur,
                        scaler=scaler,
                        start_time=next_time,
                        steps=steps_in_block,
                        interval_minutes=INTERVAL_MINUTES,
                        mode="direct",
                        exog_builder=None,
                    )
                except Exception as e:
                    logger.exception(f"Erreur forecast_block pour {name}: {e}")
                    continue

                # --- Mettre à jour la fenêtre
                windows_by_name[name] = X_seq_next

                # --- Collecter les résultats (rec_out = liste de ForecastResult)
                for res in rec_out:
                    # Compat: res peut être dataclass ou dict selon votre implémentation
                    if hasattr(res, "predicted_for"):
                        predicted_for_dt = res.predicted_for
                        pred_val = float(res.pred)
                    else:
                        predicted_for_dt = res["predicted_for"]
                        pred_val = float(res["pred"])

                    if isinstance(predicted_for_dt, datetime):
                        predicted_for_str = predicted_for_dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        # fallback si déjà string
                        predicted_for_str = str(predicted_for_dt)

                    pred_records.append(
                        {
                            "timestamp_prediction": now.strftime("%Y-%m-%d %H:%M:%S"),
                            "name": name,
                            "predicted_for": predicted_for_str,
                            "predicted_taux_de_charge": pred_val,
                        }
                    )

            # Bloc suivant
            next_time = horizon_block_end

        # 4) Retour toujours une LISTE (éventuellement vide)
        if pred_records:
                pred_record = pd.DataFrame(pred_records)
                pred_record['readable_date'] = pred_record['predicted_for']
                pred_record['nom2'] = pred_record['name']
                pred_record['readable_date'] = pred_record['readable_date'].astype(str)
                pred_record['predicted_for'] = pd.to_datetime(pred_record['predicted_for'])
                elt_save.save_to_es(pred_record)

        return pred_records

    except Exception as e:
        logger.exception(f"run_prediction() a levé une exception: {e}")
        # On retourne une liste vide pour éviter les TypeError plus loin
        return []

# ---------------------------------------
# Routes
# ---------------------------------------
@app.get("/")
async def root():
    return {"message": "TauxCharge Prediction API is running."}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/predict", response_model=PredictResponse)
async def predict(limit: int = Query(20, ge=1, le=1000)):
    """
    Ex: /predict?limit=20
    """
    results = run_prediction()  # Toujours une liste

    if not isinstance(results, list):
        # Ceinture & bretelles: on ne devrait jamais passer ici
        logger.error(f"run_prediction a renvoyé un type inattendu: {type(results)}")
        raise HTTPException(status_code=500, detail="Type de résultat inattendu.")

    if len(results) == 0:
        # Rien prédit -> 503 avec message explicite
        raise HTTPException(
            status_code=503,
            detail="Aucune prédiction disponible pour le moment (données insuffisantes ou modèles manquants).",
        )

    # Limiter proprement
    results = results[:limit]
    return {"predictions": results}
