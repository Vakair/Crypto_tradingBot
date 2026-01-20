import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model


from src.backtester import Backtester

# --- KONFIGURÁCIÓ ---
SYMBOL_YF = "BTC-USD"

# IDŐSZAK: 2025. Január 1-től MÁIG.
# Ez garantálja, hogy a modell sosem látta ezeket az adatokat tanulás közben.
START_DATE = "2025-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')

TIMEFRAME = "1h"
WINDOW_SIZE = 14
MODEL_PATH = 'models/gru_model_hourly.keras'
SCALER_PATH = 'models/scaler_hourly.pkl'

# Stratégia beállítások
LONG_THRESHOLD = 0.52  # 52% felett LONG
SHORT_THRESHOLD = 0.48  # 48% alatt SHORT


def download_hourly_data():
    print(f"2025-ös (Éles Teszt) adatok letöltése...")
    try:
        df = yf.download(SYMBOL_YF, start=START_DATE, interval=TIMEFRAME, progress=False)

        # Yahoo Finance MultiIndex javítása (ha van)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={'Close': 'close'})
        # Ellenőrzés, hogy van-e adat
        if df.empty:
            print("!!! Hiba: Üres adat érkezett! !!!")
            return pd.DataFrame()

        df = df[['close']].copy()
        print(f"Letöltve: {len(df)} órás gyertya (2025-től máig)")
        return df
    except Exception as e:
        print(f"!!! Hiba a letöltésnél: {e} !!!")
        return pd.DataFrame()


def feature_engineering(df):
    print("Indikátorok számítása...")
    df = df.copy()
    # Ugyanazok a feature-ök, mint a tanításnál!
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

    df.dropna(inplace=True)
    return df


def apply_strategy(probs):
    """
    Átalakítja a 0-1 közötti valószínűséget jelekké:
    1 (Long), -1 (Short), 0 (Cash)
    """
    signals = []
    for p in probs:
        if p > LONG_THRESHOLD:
            signals.append(1)
        elif p < SHORT_THRESHOLD:
            signals.append(-1)
        else:
            signals.append(0)
    return np.array(signals)


def main():
    print(f"\nDAYTRADE (1h) BACKTEST INDÍTÁSA...")
    print(f"   Modell: {MODEL_PATH}")
    print(f"   Időszak: {START_DATE} -> {END_DATE} (Out-of-Sample)")
    print("-" * 50)

    if not os.path.exists(MODEL_PATH):
        print("HIBA: Nincs meg az órás modell! Futtasd a train_daytrade_model.py-t!")
        return

    # Betöltés
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    #Adat
    df = download_hourly_data()
    if df.empty: return

    # Features
    df = feature_engineering(df)

    #Adatok előkészítése a modellnek (Skálázás + Ablakozás)
    features = ['target_return', 'Dist_SMA10', 'Dist_SMA50', 'RSI', 'Momentum_3D']
    data_values = df[features].values
    data_scaled = scaler.transform(data_values)

    X_test = []
    for i in range(WINDOW_SIZE, len(data_scaled)):
        X_test.append(data_scaled[i - WINDOW_SIZE:i])
    X_test = np.array(X_test)

    # 5. Adatok előkészítése a Backtesternek
    # A Backtester a 'returns' tömböt várja.
    # Fontos: A model a T időpontban dönt a T+1 mozgásról.
    # A Backtester a 'returns' tömböt úgy indexeli, hogy returns[i+1].
    # Ezért a legegyszerűbb, ha átadjuk a teljes 'pct_change' oszlopot,
    # és a jeleket (signals) hozzáigazítjuk.

    # Kivágjuk azokat a hozamokat és dátumokat, amikhez van X bemenetünk
    # A df indexeiből levágjuk az első WINDOW_SIZE elemet
    aligned_returns = df['close'].pct_change().iloc[WINDOW_SIZE:].values
    aligned_dates = df.index[WINDOW_SIZE:]

    # Biztonsági vágás: X_test és Returns hossza egyezzen
    min_len = min(len(X_test), len(aligned_returns))
    X_test = X_test[:min_len]
    aligned_returns = aligned_returns[:min_len]
    aligned_dates = aligned_dates[:min_len]

    print(f"Tesztelt gyertyák száma: {len(X_test)}")

    #Jóslás
    print("Modell elemzése folyamatban...")
    probs = model.predict(X_test, verbose=0).flatten()

    #Jelek generálása
    signals = apply_strategy(probs)

    #BACKTEST FUTTATÁSA
    #src/backtester.py
    # stop_loss_pct=0.01 -> 1% stop loss
    backtester = Backtester(initial_capital=1000, transaction_fee=0.001, stop_loss_pct=0.01)

    equity_curve, trade_count = backtester.run_strategy(aligned_returns, signals)
    sharpe, max_dd = backtester.calculate_metrics(equity_curve)

    #Eredmények Kiírása
    final_balance = equity_curve[-1]
    profit_pct = ((final_balance - 1000) / 1000) * 100

    print("\nPÉNZÜGYI JELENTÉS (2025 DAYTRADE):")
    print("=" * 40)
    print(f"Kezdő tőke:   $1,000.00")
    print(f"Végső tőke:   ${final_balance:,.2f}")
    print(f"Profit:       {profit_pct:+.2f}%")
    print(f"Kötések:      {trade_count} db")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print("=" * 40)

    # Buy & Hold összehasonlítás
    bh_signals = np.ones(len(aligned_returns))  # Mindig Long
    bh_equity, _ = backtester.run_strategy(aligned_returns, bh_signals)

    #Grafikon (plot_equity_curves metódussal)
    results = {
        'Daytrade Bot (1h)': {'equity': equity_curve, 'trades': trade_count},
        'Buy & Hold (Ref)': {'equity': bh_equity, 'trades': 1}
    }


    print("Grafikon rajzolása...")
    # A plot_equity_curves alapból elmenti, de a fájlnév a backtesterben van fixűlva.
    backtester.plot_equity_curves(aligned_dates, results, title="Daytrade (Hourly) vs Buy & Hold - 2025")


if __name__ == "__main__":
    main()