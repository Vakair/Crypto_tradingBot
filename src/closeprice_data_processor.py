import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    def __init__(self, data_path, target_col='close', window_size=60, test_split=0.1):
        self.data_path = data_path
        self.target_col = target_col
        self.window_size = window_size
        self.test_split = test_split

        # Skálázók
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))

        self.feature_cols = ['close', 'SMA_10', 'SMA_50', 'RSI', 'MACD', 'Volatility']

    def load_and_process(self):
        print(f"--- Adatok betöltése és feldolgozása... ---")
        df = pd.read_csv(self.data_path, index_col='date', parse_dates=True)
        df = df.sort_index()

        data = df[self.feature_cols].dropna()

        # Kiszámoljuk, hol vágjuk el az adatot időben
        split_idx = int(len(data) * (1 - self.test_split))

        # Tanító halmaz a skálázó betanításához
        train_df = data.iloc[:split_idx]

        # A teljes adatot konvertáljuk numpy array-re
        values = data.values
        target = data[[self.target_col]].values

        #SCALER FIT !
        # Megtanuljuk a min/max értékeket a múltból
        self.scaler_features.fit(train_df.values)
        self.scaler_target.fit(train_df[[self.target_col]].values)

        #TRANSFORM A TELJES ADATON
        # A jövőbeli adatokat a múltbeli skálázóval alakítjuk át
        scaled_features = self.scaler_features.transform(values)
        scaled_target = self.scaler_target.transform(target)

        #WINDOWING
        X, y = [], []
        for i in range(self.window_size, len(scaled_features)):
            X.append(scaled_features[i - self.window_size:i])
            y.append(scaled_target[i])

        X, y = np.array(X), np.array(y)

        #VÉGSŐ SZÉTVÁGÁS (Train/Test)
        # Mivel a windowing rövidíti az elejét, újra kell számolni a split pontot a tömbökben
        # A split_idx az eredeti adatra vonatkozott.
        # A windowing miatt az első 'window_size' adat elveszett az X-ből.
        train_size = split_idx - self.window_size

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Dátumok a vizualizációhoz (a teszt részhez)
        test_dates = data.index[split_idx:]

        print(f"   Train méret: {X_train.shape}")
        print(f"   Test méret:  {X_test.shape}")

        return X_train, y_train, X_test, y_test, test_dates

    def inverse_transform_predictions(self, predictions):
        return self.scaler_target.inverse_transform(predictions)

    def inverse_transform_actuals(self, actuals):
        return self.scaler_target.inverse_transform(actuals)