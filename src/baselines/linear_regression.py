import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error


class LinearRegressionBaseline:
    """
    Lineáris Regresszió Baseline.

    MŰKÖDÉS:
    Megnézi az elmúlt X napot (lookback_window), és megpróbál egy matematikai
    összefüggést (egyenest) találni a múltbeli árak és a holnapi ár között.

    """

    def __init__(self, lookback_window=5):
        # Hány napot nézzünk vissza a múltba?
        self.lookback_window = lookback_window
        # Maga a Scikit-learn modell
        self.model = LinearRegression()
        self.model_name = "Linear Regression"

    def create_features(self, data: pd.Series):
        """
        Ez a függvény alakítja át az idősort tanítható adathalmazzá (X és y).
        Példa (ha window=3):
        [100, 101, 102, 103] -> X=[100, 101, 102], y=103
        """
        X, y = [], []
        # Végigmegyünk az adatokon és 'ablakokat' vágunk ki
        for i in range(len(data) - self.lookback_window):
            # A bemenet (X) az ablakban lévő árak
            X.append(data.iloc[i:(i + self.lookback_window)].values)
            # A kimenet (y) a következő napi ár
            y.append(data.iloc[i + self.lookback_window])
        return np.array(X), np.array(y)

    def run(self, train_data: pd.Series, test_data: pd.Series):
        print(f"--- {self.model_name} Futtatása... ---")

        # Összefűzzük az adatokat, hogy a feature generálásnál ne vesszenek el
        # a teszt időszak első napjaihoz tartozó előzmények.
        full_data = pd.concat([train_data, test_data])
        X_full, y_full = self.create_features(full_data)

        # Kiszámoljuk, hol vágjuk el újra az adatot
        # A lookback miatt a train adat eleje 'elveszik', ezt korrigálni kell
        train_size_adjusted = len(train_data) - self.lookback_window

        # Szétválasztás Train (Tanító) és Test (Teszt) halmazra
        X_train = X_full[:train_size_adjusted]
        y_train = y_full[:train_size_adjusted]
        X_test = X_full[train_size_adjusted:]
        y_test = y_full[train_size_adjusted:]

        # A MODELL TANÍTÁSA
        # Itt számolja ki az egyenes egyenletét a modell
        self.model.fit(X_train, y_train)

        # JÓSLÁS
        # A teszt adatokra (X_test) kérünk becslést
        predictions = self.model.predict(X_test)

        # Hiba számítása
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        return {
            'predictions': predictions,
            'rmse': rmse,
            'mae': mae,
            'actual': y_test
        }