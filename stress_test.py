import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
from tensorflow.keras.models import load_model
from src.backtester import Backtester

warnings.filterwarnings("ignore")

# --- KONFIGURÁCIÓ ---
START_DATE = '2022-01-01'
END_DATE = '2022-07-01'
DATA_PATH = 'data/top_100_cryptos_with_correct_network.csv'
TARGET_SYMBOL = 'BTCUSDT'
WINDOW_SIZE = 14

# --- LONG / SHORT KÜSZÖBÖK ---
LONG_THRESHOLD = 0.55  # Ha > 55% -> LONG
SHORT_THRESHOLD = 0.45  # Ha < 45% -> SHORT (Eddig ez csak Cash volt)


def prepare_stress_data(df, start_date, end_date):
    print("   Feature Engineering és Dátum szűrés...")
    df['target_return'] = df['close'].pct_change()
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['Dist_SMA10'] = df['close'] / df['SMA_10']
    df['Dist_SMA50'] = df['close'] / df['SMA_50']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Momentum_3D'] = df['close'].pct_change(periods=3)

    df.dropna(inplace=True)
    mask = (df.index >= start_date) & (df.index <= end_date)
    df_slice = df.loc[mask].copy()

    feature_cols = ['target_return', 'Dist_SMA10', 'Dist_SMA50', 'RSI', 'Momentum_3D']
    data_values = df_slice[feature_cols].values
    actual_returns = df_slice['close'].pct_change().fillna(0).values[WINDOW_SIZE:]
    dates = df_slice.index[WINDOW_SIZE:]

    return data_values, actual_returns, dates


def apply_long_short_strategy(probs):
    """
    Most már 3 kimenet van: 1 (Long), -1 (Short), 0 (Cash)
    """
    signals = []
    current_pos = 0
    for p in probs:
        if p > LONG_THRESHOLD:
            current_pos = 1  # LONG
        elif p < SHORT_THRESHOLD:
            current_pos = -1  # SHORT (Esésre fogadunk!)
        # Ha 0.45 és 0.55 között van, tartjuk az előzőt, VAGY kiszállunk Cash-be.
        # Most legyen Cash a biztonság kedvéért, ha bizonytalan.
        else:
            current_pos = 0
        signals.append(current_pos)
    return np.array(signals)


def main():
    print(f"BEAR MARKET STRESS TESZT (LONG/SHORT) ({START_DATE} - {END_DATE})...\n")

    if not os.path.exists('models/gru_model.keras'): return

    print("   Modell betöltése...")
    model = load_model('models/gru_model.keras')
    scaler = joblib.load('models/scaler_features.pkl')

    df = pd.read_csv(DATA_PATH)
    df = df[df['symbol'] == TARGET_SYMBOL].copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)

    raw_data, actual_returns, test_dates = prepare_stress_data(df, START_DATE, END_DATE)
    data_scaled = scaler.transform(raw_data)

    X_test = []
    for i in range(WINDOW_SIZE, len(data_scaled)):
        X_test.append(data_scaled[i - WINDOW_SIZE:i])
    X_test = np.array(X_test)

    min_len = min(len(X_test), len(actual_returns))
    X_test = X_test[:min_len]
    actual_returns = actual_returns[:min_len]
    test_dates = test_dates[:min_len]

    print("   A GRU elemzi a piaci összeomlást...")
    probs = model.predict(X_test, verbose=0).flatten()

    # Stratégiák
    preds_buyhold = np.ones(len(actual_returns))

    # LONG ONLY (A régi stratégiánk - csak védekezik)
    preds_long_only = np.where(probs > 0.55, 1, 0)

    # LONG / SHORT (Az új profi stratégia - támad is)
    preds_long_short = apply_long_short_strategy(probs)

    print("\n EREDMÉNYEK A VÁLSÁG ALATT:")

    # Backtesterek
    bt_market = Backtester(initial_capital=10000, stop_loss_pct=1.0)  # Stop-loss nélkül
    bt_bot = Backtester(initial_capital=10000, stop_loss_pct=0.05)  # Stop-loss-szal

    results = {}

    # Buy & Hold
    eq_bh, tr_bh = bt_market.run_strategy(actual_returns, preds_buyhold)
    sharpe_bh, dd_bh = bt_market.calculate_metrics(eq_bh)
    results['Buy & Hold'] = {'equity': eq_bh, 'trades': tr_bh}

    # Long Only (Védekező)
    eq_lo, tr_lo = bt_bot.run_strategy(actual_returns, preds_long_only)
    sharpe_lo, dd_lo = bt_bot.calculate_metrics(eq_lo)
    results['GRU (Long Only)'] = {'equity': eq_lo, 'trades': tr_lo}

    # Long/Short (Támadó)
    eq_ls, tr_ls = bt_bot.run_strategy(actual_returns, preds_long_short)
    sharpe_ls, dd_ls = bt_bot.calculate_metrics(eq_ls)
    results['GRU (Long/Short)'] = {'equity': eq_ls, 'trades': tr_ls}

    # Kiíratás
    print("-" * 75)
    print(f"{'Modell':<25} | {'Profit %':<10} | {'Max DD %':<10} | {'Kötés':<6}")
    print("-" * 75)
    print(f"{'Buy & Hold':<25} | {(eq_bh[-1] - 10000) / 100:.2f}%     | {dd_bh:>8.2f}%  | {tr_bh:>6}")
    print(f"{'GRU (Long Only)':<25} | {(eq_lo[-1] - 10000) / 100:.2f}%     | {dd_lo:>8.2f}%  | {tr_lo:>6}")
    print(f"{'GRU (Long/Short) 🚀':<25} | {(eq_ls[-1] - 10000) / 100:.2f}%     | {dd_ls:>8.2f}%  | {tr_ls:>6}")
    print("-" * 75)

    bt_bot.plot_equity_curves(test_dates, results, title="Stressz Teszt: Long Only vs Long/Short")


if __name__ == "__main__":
    main()