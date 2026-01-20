import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RandomForestBaseline:
    """
    Random Forest Regressor.

    MŰKÖDÉS:
    Sok (pl. 100 db) 'döntési fát' épít. Mindegyik fa megpróbálja megjósolni az árat
    a múltbeli adatok alapján, majd a végén átlagoljuk a fák döntését.

    ERŐSSÉGE:
    Nagyon robusztus, nem zavarják annyira a kiugró értékek, és képes
    nem-lineáris (bonyolult) összefüggéseket is megtanulni.
    """

    def __init__(self, n_estimators=100, lookback_window=5):
        self.n_estimators = n_estimators  # Hány fát építsen az erdőben
        self.lookback_window = lookback_window
        # n_jobs=-1: Minden processzormagot használjon (gyorsabb)
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        self.model_name = "Random Forest"

    def create_features(self, data: pd.Series):
        # Ugyanaz a 'csúszóablak' módszer, mint a Linear Regression-nél
        X, y = [], []
        for i in range(len(data) - self.lookback_window):
            X.append(data.iloc[i:(i + self.lookback_window)].values)
            y.append(data.iloc[i + self.lookback_window])
        return np.array(X), np.array(y)

    def run(self, train_data: pd.Series, test_data: pd.Series):
        print(f"--- {self.model_name} Futtatása... ---")

        full_data = pd.concat([train_data, test_data])
        X_full, y_full = self.create_features(full_data)

        train_size_adjusted = len(train_data) - self.lookback_window

        X_train = X_full[:train_size_adjusted]
        y_train = y_full[:train_size_adjusted]
        X_test = X_full[train_size_adjusted:]
        y_test = y_full[train_size_adjusted:]

        # TANÍTÁS
        self.model.fit(X_train, y_train)

        # JÓSLÁS
        predictions = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        return {
            'predictions': predictions,
            'rmse': rmse,
            'mae': mae,
            'actual': y_test
        }