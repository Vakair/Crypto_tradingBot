import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



class FeatureEngineer:
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame):
        df = df.copy()

        # Célváltozó (Return)
        df['target_return'] = df['close'].pct_change()

        # Trend Indikátorok
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()  # Hosszú távú trend

        #Távolság a trendtől
        #(Ha 1.1, akkor 10%-kal az átlag felett vagyunk.)
        df['Dist_SMA10'] = df['close'] / df['SMA_10']
        df['Dist_SMA50'] = df['close'] / df['SMA_50']

        #RSI (Jól működött, maradjon)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        #Momentum (3 napos változás)
        df['Momentum_3D'] = df['close'].pct_change(periods=3)

        df.dropna(inplace=True)
        return df


class DataProcessor:
    def __init__(self, data_path, target_col='target_return', window_size=14, test_split=0.1):
        self.data_path = data_path
        self.target_col = target_col
        self.window_size = window_size
        self.test_split = test_split

        self.scaler_features = MinMaxScaler(feature_range=(0, 1))

        #
        # Kivettük a sima SMA-t, helyette a 'Dist' (távolság) van, ami jobb a gépnek
        self.feature_cols = ['target_return', 'Dist_SMA10', 'Dist_SMA50', 'RSI', 'Momentum_3D']

    def load_and_process(self):
        print(f"--- Adatok betöltése (SMART FEATURES)... ---")
        df = pd.read_csv(self.data_path, index_col='date', parse_dates=True)
        df = df.sort_index()

        # Feature Engineeringet itt hívjuk meg, ha még nincs a fájlban
        # De mivel a main.py-ban generáljuk, feltételezzük, hogy a CSV-ben benne vannak.


        # Biztosítás: újraszámoljuk, ha a CSV régi lenne
        df = FeatureEngineer.add_technical_indicators(df)

        data = df[self.feature_cols].dropna()

        # SPLIT
        split_idx = int(len(data) * (1 - self.test_split))
        train_df = data.iloc[:split_idx]

        # TARGET: Classification (UP=1, DOWN=0)
        target_class = (data[self.target_col] > 0).astype(int).values

        values = data.values
        self.scaler_features.fit(train_df.values)
        scaled_features = self.scaler_features.transform(values)

        # Windowing
        X, y = [], []
        for i in range(self.window_size, len(scaled_features)):
            X.append(scaled_features[i - self.window_size:i])
            y.append(target_class[i])

        X, y = np.array(X), np.array(y)

        train_size = split_idx - self.window_size
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        test_dates = data.index[split_idx:]

        print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
        return X_train, y_train, X_test, y_test, test_dates