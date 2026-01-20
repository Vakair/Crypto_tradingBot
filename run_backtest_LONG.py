import pandas as pd
import numpy as np
import warnings
from src.data_processor import DataProcessor
from src.backtester_LONG import Backtester
from src.baselines.naive import NaiveBaseline
from src.baselines.random_forest import RandomForestBaseline
from src.baselines.linear_regression import LinearRegressionBaseline
from src.baselines.svr import SVRBaseline
from gru_runner import build_gru_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings("ignore")

# --- KONFIGURÁCIÓ ---
DATA_PATH = 'data/top_100_cryptos_with_correct_network.csv'
TARGET_SYMBOL = 'BTCUSDT'
WINDOW_SIZE = 14
TEST_SPLIT = 0.1

# Stratégia Thresholdok
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45


def apply_smart_strategy(probs):
    signals = []
    current_pos = 0
    for p in probs:
        if p > BUY_THRESHOLD:
            current_pos = 1
        elif p < SELL_THRESHOLD:
            current_pos = 0
        signals.append(current_pos)
    return np.array(signals)


def main():
    print("BACKTEST...\n")

    #ADATOK
    print("   Adatok betöltése...")
    processor = DataProcessor(DATA_PATH, window_size=WINDOW_SIZE, test_split=TEST_SPLIT)
    X_train, y_train, X_test, y_test, test_dates = processor.load_and_process()

    df_full = pd.read_csv(DATA_PATH)
    df_btc = df_full[df_full['symbol'] == TARGET_SYMBOL]
    mask = df_btc['date'].isin(test_dates.astype(str))
    actual_returns = df_btc.loc[mask, 'close'].pct_change().fillna(0).values

    min_len = min(len(y_test), len(actual_returns))
    y_test = y_test[:min_len]
    actual_returns = actual_returns[:min_len]
    X_test = X_test[:min_len]
    test_dates = test_dates[:min_len]

    print(f"\nTeszt időszak: {min_len} nap")

    # MODELLEK
    predictions_dict = {}

    predictions_dict['Buy & Hold'] = np.ones(min_len)
    predictions_dict['Naive Baseline'] = np.roll(y_test.flatten(), 1)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print("\n--- Baseline modellek futtatása... ---")
    lr = LinearRegressionBaseline(lookback_window=WINDOW_SIZE)
    lr.model.fit(X_train_flat, y_train.flatten())
    predictions_dict['Linear Reg'] = (lr.model.predict(X_test_flat) > 0.5).astype(int)

    rf = RandomForestBaseline(lookback_window=WINDOW_SIZE)
    rf.model.fit(X_train_flat[-50000:], y_train.flatten()[-50000:])
    predictions_dict['Random Forest'] = (rf.model.predict(X_test_flat) > 0.5).astype(int)

    svr = SVRBaseline(lookback_window=WINDOW_SIZE)
    svr.model.fit(X_train_flat[-10000:], y_train.flatten()[-10000:])
    predictions_dict['SVR'] = (svr.model.predict(X_test_flat) > 0.5).astype(int)

    #GRU
    print("\n--- GRU MODELL TANÍTÁSA... ---")
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
              callbacks=[early_stop, reduce_lr], verbose=1)

    gru_probs = model.predict(X_test).flatten()
    predictions_dict['Saját GRU (Smart)'] = apply_smart_strategy(gru_probs)
    predictions_dict['Saját GRU (Sima)'] = (gru_probs > 0.5).astype(int)

    #KIÉRTÉKELÉS (PROFI TÁBLÁZAT)
    print("\n PÉNZÜGYI & KOCKÁZATI ELEMZÉS (Start: $10,000)...")
    backtester = Backtester(initial_capital=10000, transaction_fee=0.001)

    results_for_plot = {}
    sorted_keys = sorted(predictions_dict.keys())

    # Fejléc formázása
    header = f"{'Modell':<20} | {'Profit %':<10} | {'Sharpe':<8} | {'Max DD %':<10} | {'Kötés':<6}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for name in sorted_keys:
        preds = predictions_dict[name]
        equity_curve, trade_count = backtester.run_strategy(actual_returns, preds)

        # Metrikák számolása
        sharpe, max_dd = backtester.calculate_metrics(equity_curve)

        final_value = equity_curve[-1]
        profit_pct = ((final_value - 10000) / 10000) * 100

        results_for_plot[name] = {'equity': equity_curve, 'trades': trade_count}

        # Adatok kiírása
        print(f"{name:<20} | {profit_pct:>8.2f}%  | {sharpe:>8.2f} | {max_dd:>8.2f}%  | {trade_count:>6}")

    backtester.plot_equity_curves(test_dates, results_for_plot,
                                  title=f"Stratégiák Teljesítménye ({TARGET_SYMBOL})")


if __name__ == "__main__":
    main()