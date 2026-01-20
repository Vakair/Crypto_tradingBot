import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- KONFIGURÁCIÓ ---
SYMBOL_YF = "BTC-USD"
START_DATE = "2020-01-01"  # 5 évnyi adat
END_DATE = "2024-12-31"
TIMEFRAME = "1h"  # ÓRÁS ADATOK!
WINDOW_SIZE = 14
MODEL_SAVE_PATH = 'models/gru_model_hourly.keras'
SCALER_SAVE_PATH = 'models/scaler_hourly.pkl'


def download_hourly_data():
    print(f" Órás adatok letöltése Yahoo Finance-ről ({TIMEFRAME})...")
    # A yfinance 1h adatokból max 730 napot (2 évet) ad ingyen, de az nekünk elég!
    df = yf.download(SYMBOL_YF, period="2y", interval=TIMEFRAME, progress=False)

    # Adattisztítás
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close': 'close'})
    df = df[['close']].copy()

    print(f" Letöltve: {len(df)} órás gyertya")
    return df


def feature_engineering(df):
    print(" Feature Engineering...")
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['target_return'] = df['close'].pct_change()
    df['Dist_SMA10'] = df['close'] / df['SMA_10']
    df['Dist_SMA50'] = df['close'] / df['SMA_50']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Momentum_3D'] = df['close'].pct_change(periods=3)

    # Target: 1 ha a köv. órában nő, 0 ha csökken
    df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df


def build_model(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    if not os.path.exists('models'): os.makedirs('models')

    #Adat
    df = download_hourly_data()
    if df.empty:
        print("!!! Hiba: Üres adat. !!!")
        return

    # Features
    df = feature_engineering(df)

    # Skálázás
    features = ['target_return', 'Dist_SMA10', 'Dist_SMA50', 'RSI', 'Momentum_3D']
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df[features])

    # Windowing
    X, y = [], []
    for i in range(WINDOW_SIZE, len(data_scaled)):
        X.append(data_scaled[i - WINDOW_SIZE:i])
        y.append(df['Target'].iloc[i])

    X, y = np.array(X), np.array(y)

    #Tanítás
    print(f" Modell tanítása {len(X)} mintán...")
    model = build_model((X.shape[1], X.shape[2]))
    early_stop = EarlyStopping(monitor='loss', patience=5)
    # model.fit(X, y, epochs=10, batch_size=32, verbose=1, callbacks=[early_stop])
    model.fit(X, y, epochs=50, batch_size=64, verbose=1, callbacks=[early_stop])

    #Mentés
    model.save(MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"\nDaytrade modell mentve: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()