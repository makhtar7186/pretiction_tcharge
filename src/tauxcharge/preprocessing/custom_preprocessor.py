from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['date_mesure'] = X['date_mesure'].str.replace(r'\.0$', '', regex=True)
        X['charge_canal'] = X['tcp_dw'] + X['udp_dw'] + X['icmp_dw'] + X['other_dw']
        X['taux_de_charge'] = X['charge_canal'] / X['capacite_max']
        X['date_mesure'] = pd.to_datetime(X['date_mesure'])
        X = X[['names', 'date_mesure', 'taux_de_charge']].dropna().sort_values('date_mesure')
        X = X.drop_duplicates(subset=['date_mesure', 'names'], keep='first')
        return X
    
    def get_feature_names_out(self, input_features=None):
        # retourne les noms des colonnes en sortie
        if input_features is None:
            input_features = [f"feature_{i}" for i in range(self.n_features_)]
        return np.array(input_features)



# Then continue with your pipeline