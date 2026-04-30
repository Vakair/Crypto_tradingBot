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
START_DATE = "2025-01-01"
END_DATE = datetime.now().strftime('%Y-%m-%d')

TIMEFRAME = "1h"
WINDOW_SIZE = 14
MODEL_PATH = 'models/gru_model_hourly.keras'
SCALER_PATH = 'models/scaler_hourly.pkl'

# Stratégia beállítások - CSAK LONG (Memóriával)
LONG_THRESHOLD = 0.51  # 51% felett LONG
EXIT_THRESHOLD = 0.49  # 49% alatt CASH


def download_hourly_data():
    print(f"2025-ös (Éles Teszt) adatok letöltése...")
    try:
        df = yf.download(SYMBOL_YF, start=START_DATE, interval=TIMEFRAME, progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={'Close': 'close'})
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


def apply_long_only_strategy(probs):
    """
    Állapotmegőrző CSAK LONG stratégia.
    Ha 0.49 és 0.51 között van, tartja az eddigi pozíciót.
    """
    signals = []
    current_pos = 0
    for p in probs:
        if p > LONG_THRESHOLD:
            current_pos = 1      # LONG-ba lép
        elif p < EXIT_THRESHOLD:
            current_pos = 0      # CASH-be lép (nincs short)
        signals.append(current_pos)
    return np.array(signals)


def main():
    print(f"\nDAYTRADE (1h) BACKTEST INDÍTÁSA - CSAK LONG VERZIÓ...")
    print(f"   Modell: {MODEL_PATH}")
    print(f"   Időszak: {START_DATE} -> {END_DATE} (Out-of-Sample)")
    print("-" * 50)

    if not os.path.exists(MODEL_PATH):
        print("HIBA: Nincs meg az órás modell! Futtasd a train_daytrade_model.py-t!")
        return

    # Betöltés
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Adat
    df = download_hourly_data()
    if df.empty: return

    # Features
    df = feature_engineering(df)

    # Adatok előkészítése a modellnek
    features = ['target_return', 'Dist_SMA10', 'Dist_SMA50', 'RSI', 'Momentum_3D']
    data_values = df[features].values
    data_scaled = scaler.transform(data_values)

    X_test = []
    for i in range(WINDOW_SIZE, len(data_scaled)):
        X_test.append(data_scaled[i - WINDOW_SIZE:i])
    X_test = np.array(X_test)

    aligned_returns = df['close'].pct_change().iloc[WINDOW_SIZE:].values
    aligned_dates = df.index[WINDOW_SIZE:]

    min_len = min(len(X_test), len(aligned_returns))
    X_test = X_test[:min_len]
    aligned_returns = aligned_returns[:min_len]
    aligned_dates = aligned_dates[:min_len]

    print(f"Tesztelt gyertyák száma: {len(X_test)}")

    # Jóslás
    print("Modell elemzése folyamatban...")
    probs = model.predict(X_test, verbose=0).flatten()

    # Jelek generálása (CSAK LONG)
    signals = apply_long_only_strategy(probs)

    # BACKTEST FUTTATÁSA - A Bot kap egy 3%-os stop-losst
    backtester = Backtester(initial_capital=1000, transaction_fee=0.001, stop_loss_pct=0.01)

    equity_curve, trade_count = backtester.run_strategy(aligned_returns, signals)
    sharpe, max_dd = backtester.calculate_metrics(equity_curve)

    # Eredmények Kiírása
    final_balance = equity_curve[-1]
    profit_pct = ((final_balance - 1000) / 1000) * 100

    print("\nPÉNZÜGYI JELENTÉS (2025 DAYTRADE - LONG ONLY):")
    print("=" * 40)
    print(f"Kezdő tőke:   $1,000.00")
    print(f"Végső tőke:   ${final_balance:,.2f}")
    print(f"Profit:       {profit_pct:+.2f}%")
    print(f"Kötések:      {trade_count} db")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print("=" * 40)

    # Buy & Hold összehasonlítás - KÜLÖN Backtester, 100%-os Stop-lossal (tehát kikapcsolva)
    bh_backtester = Backtester(initial_capital=1000, transaction_fee=0.001, stop_loss_pct=1.00)
    bh_signals = np.ones(len(aligned_returns))  # Mindig Long
    bh_equity, _ = bh_backtester.run_strategy(aligned_returns, bh_signals)

    # Grafikon
    results = {
        'Daytrade Bot (1h) LONG ONLY': {'equity': equity_curve, 'trades': trade_count},
        'Buy & Hold (Ref)': {'equity': bh_equity, 'trades': 1}
    }

    print("Grafikon rajzolása...")
    backtester.plot_equity_curves(aligned_dates, results, title="Daytrade (Hourly) LONG ONLY vs Buy & Hold - 2025")

if __name__ == "__main__":
    main()