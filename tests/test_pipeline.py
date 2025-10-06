from src.tauxcharge.pipeline import preprocessing_pipe # type: ignore
import pandas as pd
from src.tauxcharge.preprocessing.sequence_generator import SequenceGenerator
from src.tauxcharge.preprocessing.splitter import TrainTestSplitter
from sklearn.base import BaseEstimator, RegressorMixin
from src.tauxcharge.elastic.elasticsearch import ElasticFetcher
from src.tauxcharge.model.lstm_regressor import KerasLSTMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#from collections import Iterable

import numpy as np

def test_pipeline():
    df = ElasticFetcher().fetch_all()
    df = preprocessing_pipe.fit_transform(df)
    #df_train,df_test = TrainTestSplitter().fit_transform(df)
    lstm_model = KerasLSTMRegressor().load_model_lstm()
    print("fin chargement du model \n debut sequensage")
    x_train = SequenceGenerator(look_back=288).fit_transform(df)
    #x_test = SequenceGenerator(look_back=288).fit_transform(df_test)
    print("debut prediction sur les donnee")
    predictions_train = lstm_model.predict([x_train['X_seq'], x_train['name_seq']]).flatten()
    print("fin prediction sur les donnee")
    
    print(f"Predictions on train set: {predictions_train[:5]}")
    predictions_test = lstm_model.predict([x_train['X_seq'], x_train['name_seq']]).flatten()
    y_true = x_train['y_seq']
    mae = mean_absolute_error(y_true, predictions_train)
    mse = mean_squared_error(y_true, predictions_train)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, predictions_train)
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    assert True






