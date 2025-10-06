from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass
from datetime import timedelta, datetime
import pickle
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
from tensorflow.keras.models import Model, load_model as keras_load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation,
    Dropout, MaxPooling1D, LSTM, Dense, Masking
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


@dataclass
class ForecastResult:
    pred_scaled: float
    pred: float
    predicted_for: datetime


class CNNLSTMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        look_back: int = 2016,
        horizon: int = 1440,
        feature_dim: Optional[int] = None,
        lstm_units: int = 128,
        conv_channels: int = 64,
        dropout_rate: float = 0.2,
        use_masking: bool = False,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 20,
        patience: int = 5,
        val_split: float = 0.1,
        verbose: int = 1,
        target_col_idx: int = 0
    ):
        self.look_back = look_back
        self.horizon = horizon
        self.feature_dim = feature_dim
        self.lstm_units = lstm_units
        self.conv_channels = conv_channels
        self.dropout_rate = dropout_rate
        self.use_masking = use_masking
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_split = val_split
        self.verbose = verbose
        self.target_col_idx = target_col_idx
        self.model_: Optional[Model] = None
        self.feature_dim_: Optional[int] = None

    def build_model(self, time_steps: int, feature_dim: int) -> Model:
        inp_x = Input(shape=(time_steps, feature_dim), name="time_series_input")
        x = inp_x

        if self.use_masking:
            x = Masking(mask_value=0.0, name="mask")(x)

        x = Conv1D(self.conv_channels, 7, padding="causal", name="conv1")(x)
        x = BatchNormalization(name="bn1")(x)
        x = Activation("relu", name="relu1")(x)
        x = Dropout(self.dropout_rate, name="drop1")(x)

        x = Conv1D(self.conv_channels, 3, padding="causal", name="conv2")(x)
        x = BatchNormalization(name="bn2")(x)
        x = Activation("relu", name="relu2")(x)
        x = Dropout(self.dropout_rate, name="drop2")(x)

        x = MaxPooling1D(pool_size=2, name="pool2")(x)

        x = Conv1D(self.conv_channels, 3, dilation_rate=2, padding="causal", name="conv3_dil2")(x)
        x = BatchNormalization(name="bn3")(x)
        x = Activation("relu", name="relu3")(x)
        x = Dropout(self.dropout_rate, name="drop3")(x)

        x = LSTM(self.lstm_units, return_sequences=False, name="lstm")(x)
        x = Dropout(self.dropout_rate, name="drop_lstm")(x)

        x = Dense(128, activation='relu', name="dense_pre")(x)
        x = Dropout(0.3, name="drop_pre")(x)
        x = Dense(256, activation="relu", name="dense1")(x)
        x = Dropout(self.dropout_rate, name="drop_dense1")(x)

        out = Dense(self.horizon, name="forecast")(x)
        model = Model(inputs=inp_x, outputs=out, name="CNN_LSTM_PerSeries")
        model.compile(optimizer=Adam(self.learning_rate), loss="mse", metrics=["mae"])
        return model

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X); y = np.asarray(y)
        assert X.ndim == 3, "X doit être (B, look_back, feature_dim)"
        assert X.shape[1] == self.look_back, f"look_back attendu={self.look_back}, obtenu={X.shape[1]}"

        self.feature_dim_ = X.shape[2] if self.feature_dim is None else self.feature_dim

        if y.ndim == 1:
            if self.horizon == 1:
                y = y.reshape(-1, 1)
            else:
                raise ValueError("y est 1D mais horizon>1. Fournir y shape (B, horizon).")
        elif y.ndim == 2 and y.shape[1] != self.horizon:
            raise ValueError(f"y doit avoir shape (B, {self.horizon}).")

        self.model_ = self.build_model(self.look_back, self.feature_dim_)
        cb = EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        self.model_.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.val_split,
            callbacks=[cb],
            verbose=self.verbose
        )
        return self

    def predict(self, X: np.ndarray, verbose: Optional[int] = None) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Modèle non chargé/entraîné. Appelle fit() ou load_model()/load_model_lstm() d'abord.")
        if verbose is None:
            verbose = self.verbose
        return self.model_.predict(X, verbose=verbose)

    # --------- Sauvegarde / Chargement (Keras natif) ----------
    def load_model(self, path="src/tauxcharge/model/all_models_thies.pkl"):
       
        # Charger l'objet sauvegardé
        with open(path, "rb") as f:
            all_models = pickle.load(f)
            print("Modèle chargé avec succès.")
        self.model_ = all_models
        return self.model_

    # recupere le nom des models disponibe 
    def get_models_name(self):
        all_model = self.model_
        keys_list = list(all_model.keys())
        liste = pd.DataFrame({'models_name':keys_list})
        return liste

    # --------- Forecast block (direct / récursif) ----------
    def forecast_block(
        self,
        regressor,
        X_seq: np.ndarray,
        scaler,
        start_time: datetime,
        steps: int = 288*5,
        interval_minutes: int = 5,
        mode: str = "direct",
        exog_builder: Optional[Callable[[np.ndarray, datetime], np.ndarray]] = None,
    ) :
        
        X = np.asarray(X_seq).copy()
        if self.feature_dim_ is None:
            self.feature_dim_ = X.shape[2]
        assert X.shape == (1, self.look_back, self.feature_dim_), \
            f"X_seq doit être (1, {self.look_back}, {self.feature_dim_})"

        results: List[ForecastResult] = []
        t_cur = start_time

        if mode == "direct":
            yhat_scaled = np.asarray(regressor.predict(X, verbose=0)).ravel()[:steps]
            for y_s in yhat_scaled:
                y_s = float(y_s)
                pred_val = float(scaler.inverse_transform([[y_s]])[0][0])

                results.append(ForecastResult(y_s, pred_val, t_cur))

                last_row = X[0, -1, :].copy()
                if exog_builder is not None:
                    last_row = exog_builder(last_row, t_cur)
                last_row[self.target_col_idx] = y_s

                X[:, :-1, :] = X[:, 1:, :]
                X[:, -1, :] = last_row
                t_cur += timedelta(minutes=interval_minutes)
            return results, X

        elif mode == "recursive":
            for _ in range(steps):
                yhat_scaled = np.asarray(self.predict(X, verbose=0)).ravel()
                y1 = float(yhat_scaled[0])
                pred_val = float(scaler.inverse_transform([[y1]])[0][0])
                results.append(ForecastResult(y1, pred_val, t_cur))

                last_row = X[0, -1, :].copy()
                if exog_builder is not None:
                    last_row = exog_builder(last_row, t_cur)
                last_row[self.target_col_idx] = y1

                X[:, :-1, :] = X[:, 1:, :]
                X[:, -1, :] = last_row
                t_cur += timedelta(minutes=interval_minutes)
            return results, X

        else:
            raise ValueError("mode doit être 'direct' ou 'recursive'")
