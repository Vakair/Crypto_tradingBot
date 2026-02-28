import pandas as pd
import numpy as np
import warnings
from src.data_processor import DataProcessor
from src.backtester import Backtester
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

# --- ÚJ STRATÉGIA PARAMÉTEREK ---
LONG_THRESHOLD = 0.55  # Ha > 55% biztos -> LONG
SHORT_THRESHOLD = 0.35  # Ha < 45% biztos -> SHORT


def apply_hedge_fund_strategy(probs):
    """
    Ez a függvény generálja a -1 (Short) jeleket is!
    """
    signals = []
    for p in probs:
        if p > LONG_THRESHOLD:
            signals.append(1)  # Long
        elif p < SHORT_THRESHOLD:
            signals.append(-1)  # Short
        else:
            signals.append(0)  # Cash (Bizonytalan)
    return np.array(signals)


def main():
    print("FINAL BACKTEST (LONG/SHORT KÉPESSÉGGEL)...\n")

    #ADATOK
    print("Adatok betöltése...")
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

    #BASELINE MODELLEK
    predictions_dict = {}
    predictions_dict['Buy & Hold'] = np.ones(min_len)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print("\n--- Baseline modellek futtatása... ---")
    lr = LinearRegressionBaseline(lookback_window=WINDOW_SIZE)
    lr.model.fit(X_train_flat, y_train.flatten())
    predictions_dict['Linear Reg'] = (lr.model.predict(X_test_flat) > 0.5).astype(int)

    # GRU MODELL
    print("\n--- GRU MODELL TANÍTÁSA... ---")
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
              callbacks=[early_stop, reduce_lr], verbose=1)

    gru_probs = model.predict(X_test).flatten()

    # --- STRATÉGIÁK ---
    #Sima GRU (Csak Vesz / Elad)
    predictions_dict['Saját GRU (Long Only)'] = (gru_probs > 0.5).astype(int)

    #Hedge Fund GRU (Vesz / Shortol / Cash)
    predictions_dict['Saját GRU (Long/Short)'] = apply_hedge_fund_strategy(gru_probs)

    #KIÉRTÉKELÉS
    print("\nPÉNZÜGYI ELEMZÉS (Start: $10,000)...")
    # Itt most bekapcsoljuk a Stop-Loss-t (5%) mindenkinek
    backtester = Backtester(initial_capital=10000, transaction_fee=0.001, stop_loss_pct=0.05)

    results_for_plot = {}
    sorted_keys = sorted(predictions_dict.keys())

    print("-" * 65)
    print(f"{'Modell':<25} | {'Profit %':<10} | {'Sharpe':<8} | {'Max DD':<10}")
    print("-" * 65)

    for name in sorted_keys:
        preds = predictions_dict[name]
        equity_curve, trade_count = backtester.run_strategy(actual_returns, preds)
        sharpe, max_dd = backtester.calculate_metrics(equity_curve)

        final_value = equity_curve[-1]
        profit_pct = ((final_value - 10000) / 10000) * 100

        results_for_plot[name] = {'equity': equity_curve, 'trades': trade_count}

        print(f"{name:<25} | {profit_pct:>8.2f}%  | {sharpe:>8.2f} | {max_dd:>8.2f}%")

    backtester.plot_equity_curves(test_dates, results_for_plot,
                                  title=f"Stratégiák Versenye ({TARGET_SYMBOL})")


if __name__ == "__main__":
    main()