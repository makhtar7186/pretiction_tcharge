from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class SequenceGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, look_back=2016, horizon=1440, scale_target=True):
        self.look_back = look_back
        self.horizon = horizon
        self.scale_target = scale_target
        self.scaler_ = None
        self.feature_names_out_ = ['X_seq', 'name_seq', 'y_seq', 'date_seq']

    def fit(self, X, y=None):
        self.scaler_ = MinMaxScaler()
        if isinstance(X, pd.DataFrame):
            self.scaler_.fit(X['taux_de_charge'].values.reshape(-1, 1))
        else:
            raise ValueError("L'entrée doit être un DataFrame pandas")
        return self

    def transform(self, X):
        X_seq, name_seq, y_seq, date_seq = [], [], [], []

        # Assure un tri temporel correct (évite un tri lexicographique de str)
        if not np.issubdtype(X['date_mesure'].dtype, np.datetime64):
            X = X.assign(date_mesure=pd.to_datetime(X['date_mesure'], errors='coerce'))
            X = X.dropna(subset=['date_mesure'])

        for names, group in X.groupby('names'):
            group = group.sort_values('date_mesure')

            series = group['taux_de_charge'].values.astype('float32')
            dates  = group['date_mesure'].values
            series_scaled = self.scaler_.transform(series.reshape(-1, 1)).astype('float32')

            # fenêtre (look_back) -> cible multi-horizon (horizon)
            # limite: i + look_back + horizon <= len(series_scaled)
            last_start = len(series_scaled) - self.look_back - self.horizon
            if last_start < 0:
                continue

            for i in range(0, last_start + 1):
                X_seq.append(series_scaled[i : i + self.look_back])  # (look_back, 1)
                y_seq.append(
                    series_scaled[i + self.look_back : i + self.look_back + self.horizon, 0]
                )                                                     # (horizon,)
                name_seq.append(names)
                date_seq.append(dates[i + self.look_back : i + self.look_back + self.horizon])

        return {
            'X_seq'  : np.asarray(X_seq,  dtype='float32'),   # (N, look_back, 1)
            'name_seq': np.asarray(name_seq, dtype='object').reshape(-1, 1),  # (N, 1)
            'y_seq'  : np.asarray(y_seq,  dtype='float32'),   # (N, horizon)
            'date_seq': np.asarray(date_seq, dtype=object),
            'scaler' : self.scaler_
        }
