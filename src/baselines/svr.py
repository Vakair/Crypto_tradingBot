import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


class SVRBaseline:
    """
    Support Vector Regression (SVR).

    MŰKÖDÉS:
    Hasonló a lineáris regresszióhoz, de megenged egy bizonyos hibahatárt (margin),
    és képes görbéket is illeszteni (kernel='rbf').

    FONTOS:
    Az SVR nagyon rosszul működik, ha az adatok nincsenek skálázva!
    A Bitcoin ára 60.000, míg a változás csak 100-200. Ezt a modell nem érti,
    ezért először mindent átalakítunk (StandardScaler) kicsi számokká.
    """

    def __init__(self, kernel='rbf', C=100, gamma=0.1, lookback_window=5):
        self.lookback_window = lookback_window
        # Paraméterek:
        # kernel='rbf': Görbült összefüggések kezelése
        # C=100: Mennyire büntesse a hibát
        self.model = SVR(kernel=kernel, C=C, gamma=gamma)
        self.model_name = "SVR"

        #két külön skálázó: bemenetnek (X), kimenetnek (y)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def create_features(self, data: pd.Series):
        X, y = [], []
        for i in range(len(data) - self.lookback_window):
            X.append(data.iloc[i:(i + self.lookback_window)].values)
            y.append(data.iloc[i + self.lookback_window])
        # reshape (-1, 1) kell, mert a skálázó 2D tömböt vár
        return np.array(X), np.array(y).reshape(-1, 1)

    def run(self, train_data: pd.Series, test_data: pd.Series):
        print(f"--- {self.model_name} Futtatása (Skálázással)... ---")

        full_data = pd.concat([train_data, test_data])
        X_full, y_full = self.create_features(full_data)

        train_size_adjusted = len(train_data) - self.lookback_window

        X_train = X_full[:train_size_adjusted]
        y_train = y_full[:train_size_adjusted]
        X_test = X_full[train_size_adjusted:]
        y_test = y_full[train_size_adjusted:]

        #SKÁLÁZÁS (Nagyon fontos lépés!)
        # Csak a tanító adaton illesztjük (fit), hogy ne "lássunk bele" a tesztbe
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        #A teszt adatot a tanító alapján skálázzuk (transform)
        X_test_scaled = self.scaler_X.transform(X_test)

        #TANÍTÁS (A skálázott adatokon)
        self.model.fit(X_train_scaled, y_train_scaled.ravel())

        #JÓSLÁS (Még mindig skálázott értékeket kapunk)
        preds_scaled = self.model.predict(X_test_scaled)

        #VISSZASKÁLÁZÁS (Inverse Transform)
        #Visszaalakítjuk a kapott kicsi számokat valódi dollár árfolyamra
        predictions = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        y_test_original = y_test.flatten()

        rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
        mae = mean_absolute_error(y_test_original, predictions)

        return {
            'predictions': predictions,
            'rmse': rmse,
            'mae': mae,
            'actual': y_test_original
        }