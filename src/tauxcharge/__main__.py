# __main__.py
import time
import pandas as pd
from datetime import datetime, timedelta

from tauxcharge.elastic.elasticsearch import ElasticFetcher
from tauxcharge.model.lstm_regressor import CNNLSTMRegressor
from tauxcharge.preprocessing.sequence_generator import SequenceGenerator
from tauxcharge.preprocessing.custom_preprocessor import CustomPreprocessor
import os



LOOK_BACK = 2016
PREDICT_HORIZON = 1440
INTERVAL_MINUTES = 5

# Initialiser les objets
fetcher = ElasticFetcher()
elt_save = ElasticFetcher(index=os.getenv("ES_INDEX_dest"))
regressor = CNNLSTMRegressor()
all_models = regressor.load_model()
liste = regressor.get_models_name()

pre = CustomPreprocessor()
seq_generator = SequenceGenerator(look_back=LOOK_BACK)


print("✅ Début de la boucle de prédiction continue…")  





# Charger toutes les données
df_all = fetcher.fetch_last_days(day=13)
df_all = pre.fit_transform(df_all)

#now = datetime.utcnow()
now = df_all['date_mesure'].max()
print(f"\n🕒 Heure actuelle : {now:%Y-%m-%d %H:%M:%S} — récupération des données…")

df_all['date_mesure'] = pd.to_datetime(df_all['date_mesure'])
df_all = df_all.sort_values(['names', 'date_mesure'])

minute = (now.minute // INTERVAL_MINUTES + 1) * INTERVAL_MINUTES
next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)
pred_records = []
end = now + timedelta(days=5)
windows_by_name = {}   # name -> X_seq courant (1, LOOK_BACK, F)
scalers_by_name = {}   # name -> scaler cible


while next_time < end:
    
    # ⚠️ utiliser builtins.min pour éviter le shadowing de min()
    import builtins
    horizon_block_end = builtins.min(next_time + timedelta(days=5), end)
    steps_in_block = int((horizon_block_end - next_time).total_seconds() // (INTERVAL_MINUTES * 60))

    for name in df_all['names'].unique():
        if name not in liste.values:
            continue
        # 1) Bootstrap état
        if name not in windows_by_name:
            df_name = df_all[df_all['names'] == name].copy().sort_values('date_mesure')
            seq_data = seq_generator.fit_transform(df_name)
            if seq_data['X_seq'].shape[0] == 0:
                print(f"⛔ Pas assez de données pour '{name}' (bootstrap)")
                continue
            windows_by_name[name] = seq_data['X_seq'][-1].reshape(1, LOOK_BACK, -1)
            scalers_by_name[name] = seq_data['scaler']  # scaler cible

        # 2) Forecast bloc courant
        X_seq_cur = windows_by_name[name]
        scaler    = scalers_by_name[name]
        
        rec_out, X_seq_next = regressor.forecast_block(
            regressor=all_models[name],      # ← assure-toi que c’est bien ton dict de modèles
            X_seq=X_seq_cur,
            scaler=scaler,
            start_time=next_time,
            steps=steps_in_block,
            interval_minutes=INTERVAL_MINUTES,
            mode="direct",                  # ou "recursive"
            exog_builder=None               # ou exog_builder_default
        )

        # 3) Persister la fenêtre mise à jour
        windows_by_name[name] = X_seq_next

        # 4) Collecte
        for res in rec_out:  # res est un ForecastResult
            pred_records.append({
                "timestamp_prediction": now.strftime("%Y-%m-%d %H:%M:%S"),
                "name": name,
                "predicted_for": res.predicted_for.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_taux_de_charge": float(res.pred),  # par sécurité
            })

    # 5) Bloc suivant
    next_time = horizon_block_end

if pred_records:
    pred_record = pd.DataFrame(pred_records)
    pred_record['readable_date'] = pred_record['predicted_for']
    pred_record['nom2'] = pred_record['name']
    pred_record['readable_date'] = pred_record['readable_date'].astype(str)
    pred_record['predicted_for'] = pd.to_datetime(pred_record['predicted_for'])
    print(pred_records.info())
    # elt_save.save_to_es(pred_records)

else:
    print("code error")
