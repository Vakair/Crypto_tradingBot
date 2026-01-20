import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

from src.baselines.naive import NaiveBaseline
from src.baselines.linear_regression import LinearRegressionBaseline
from src.baselines.arima import ArimaBaseline
from src.baselines.random_forest import RandomForestBaseline
from src.baselines.svr import SVRBaseline

# --- KONFIGURÁCIÓ ---
DATA_PATH = 'data/top_100_cryptos_with_correct_network.csv'
ENHANCED_DATA_PATH = 'data/enhanced_data_returns.csv'  # Új fájlnév a hozamoknak
TARGET_SYMBOL = 'BTCUSDT'
TEST_DAYS = 60


class FeatureEngineer:
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame):
        df = df.copy()

        # --- A LÉNYEG: HOZAM SZÁMÍTÁSA ---
        df['target_return'] = df['close'].pct_change()

        # Egyéb indikátorok maradhatnak, return a legfontosabb
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['RSI'] = 100 - (100 / (1 + (
                    df['close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / -df['close'].diff().where(
                lambda x: x < 0, 0).rolling(14).mean())))

        # Log Return (stabilitás miatt ezt is kiszámoljuk)
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        df['Volatility'] = df['Log_Return'].rolling(window=20).std()

        df.dropna(inplace=True)
        return df


def load_data_returns():
    print(f"\nAdatok betöltése és HOZAM számítása...")
    if not os.path.exists(DATA_PATH): raise FileNotFoundError("Nincs meg a CSV!")

    df = pd.read_csv(DATA_PATH)
    df = df[df['symbol'] == TARGET_SYMBOL].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    # Indikátorok és HOZAM (Return) generálása
    df = FeatureEngineer.add_technical_indicators(df)

    df.to_csv(ENHANCED_DATA_PATH)
    print(f"Hozam adatok elmentve: {ENHANCED_DATA_PATH}")

    # A Baseline-oknak most a 'target_return' oszlopot adjuk oda!
    series = df['target_return'].asfreq('D', method='ffill')
    return series


def calculate_directional_accuracy_returns(actuals, predictions):
    """
    Irányhelyesség HOZAMOKRA.
    Sokkal egyszerűbb: Ha a jóslat > 0 (emelkedés) és a valóság > 0, akkor találat.
    """
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    min_len = min(len(actuals), len(predictions))

    actuals = actuals[-min_len:]
    predictions = predictions[-min_len:]

    # Csak az előjelet nézzük (Sign)
    # +1 ha emelkedés, -1 ha csökkenés
    hits = np.sign(actuals) == np.sign(predictions)
    return np.mean(hits) * 100


def main():
    os.makedirs('results', exist_ok=True)
    try:
        series = load_data_returns()
    except Exception as e:
        print(f"KRITIKUS HIBA: {e}")
        return

    train_data = series.iloc[:-TEST_DAYS]
    test_data = series.iloc[-TEST_DAYS:]

    # A modellek paramétereit kicsit igazítjuk a kicsi számokhoz
    models = [
        NaiveBaseline(),
        LinearRegressionBaseline(lookback_window=5),
        # Arima d=0, mert a hozam már stacioner (nem kell deriválni)
        ArimaBaseline(order=(5, 0, 0)),
        RandomForestBaseline(lookback_window=5),
        SVRBaseline(lookback_window=5)
    ]

    plt.figure(figsize=(16, 9))
    plt.plot(test_data.index, test_data.values, label='VALÓS HOZAM', color='black', alpha=0.5)

    print("\nBaseline Modellek (HOZAMRA)...")

    for model in models:
        try:
            res = model.run(train_data, test_data)
            preds = res['predictions']
            actuals = res['actual']

            # Grafikon igazítás
            plot_index = test_data.index[-len(preds):]

            # Irányhelyesség
            acc = calculate_directional_accuracy_returns(actuals, preds)

            # RMSE itt lehet nagyon kicsi lesz!
            print(f"{model.model_name:<20} -> RMSE: {res['rmse']:.4f} | Irány: {acc:.2f}%")

            plt.plot(plot_index, preds, label=f"{model.model_name} (Acc: {acc:.0f}%)")

        except Exception as e:
            print(f"!!! HIBA: {model.model_name}: {e} !!!")

    plt.title(f'{TARGET_SYMBOL} Napi Hozam Előrejelzés (Returns)', fontsize=16)
    plt.legend()
    plt.savefig('results/baseline_returns.png')
    plt.show()


if __name__ == "__main__":
    main()