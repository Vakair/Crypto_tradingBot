import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings


warnings.filterwarnings("ignore")

# --- MODELLEK IMPORTÁLÁSA ---
from src.baselines.naive import NaiveBaseline
from src.baselines.linear_regression import LinearRegressionBaseline
from src.baselines.arima import ArimaBaseline
from src.baselines.random_forest import RandomForestBaseline
from src.baselines.svr import SVRBaseline

# --- KONFIGURÁCIÓ ---
DATA_PATH = 'data/top_100_cryptos_with_correct_network.csv'
ENHANCED_DATA_PATH = 'data/enhanced_data.csv'
TARGET_SYMBOL = 'BTCUSDT'
TARGET_COL = 'close'
TEST_DAYS = 60



class FeatureEngineer:
    """
    Ez az osztály felel azért, hogy a nyers árfolyamadatokból (Close)
    olyan technikai indikátorokat gyártson, amik segítenek a modellnek.
    """

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame):
        df = df.copy()

        #Simple Moving Average (SMA)
        df['SMA_10'] = df['close'].rolling(window=10).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()

        #RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        #MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        #Volatilitás (Log Return alapján)
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        df['Volatility'] = df['Log_Return'].rolling(window=20).std()

        #Takarítás: A kezdeti NaN értékek eldobása
        df.dropna(inplace=True)

        return df


def load_data():
    """Adatok betöltése, feature engineering és szűrése."""
    print(f"\nAdatok betöltése innen: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("\n Nem találja a CSV fájlt! \n")

    df = pd.read_csv(DATA_PATH)

    # Szűrés a konkrét coinra
    print(f"   Szűrés erre: {TARGET_SYMBOL}")
    df = df[df['symbol'] == TARGET_SYMBOL].copy()

    # Dátum formátum
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df.set_index('date', inplace=True)

    print(f"   Nyers adatok: {len(df)} sor.")

    # ---  Feature Engineering ---
    print("   Technikai indikátorok generálása...")
    df = FeatureEngineer.add_technical_indicators(df)

    df.to_csv(ENHANCED_DATA_PATH)
    print(f"  Bővített adatok elmentve ide: {ENHANCED_DATA_PATH}")

    # A baseline-oknak csak a 'close' árfolyam kell Series-ként
    # Biztosítjuk, hogy ne legyenek lyukak (ffill)
    series = df[TARGET_COL].asfreq('D', method='ffill')
    return series


def calculate_directional_accuracy(actuals, predictions):
    """
    Kiszámolja, hogy milyen %-ban találta el a modell az irányt (fel/le).
    """
    actuals = np.array(actuals)
    predictions = np.array(predictions)

    # Mivel a baseline-ok néha rövidebb jóslatot adnak vissza a lookback miatt,
    # igazítanunk kell a hosszokat.
    min_len = min(len(actuals), len(predictions))
    actuals = actuals[-min_len:]
    predictions = predictions[-min_len:]

    # Irány számítása: (Mai ár - Tegnapi ár)
    # A jóslat irányát úgy nézzük: (Jósolt Mai ár - Tényeges Tegnapi ár)
    if len(actuals) < 2:
        return 0.0

    actual_direction = actuals[1:] - actuals[:-1]
    predicted_direction = predictions[1:] - actuals[:-1]

    # Ha az előjel megegyezik (+ és + VAGY - és -), akkor jó az irány
    hits = np.sign(actual_direction) == np.sign(predicted_direction)
    return np.mean(hits) * 100


def main():
    os.makedirs('results', exist_ok=True)

    try:
        series = load_data()
    except Exception as e:
        print(f"KRITIKUS HIBA: {e}")
        return

    # Train-Test szétvágás
    train_data = series.iloc[:-TEST_DAYS]
    test_data = series.iloc[-TEST_DAYS:]

    # Modellek
    models = [
        NaiveBaseline(),
        LinearRegressionBaseline(lookback_window=5),
        ArimaBaseline(order=(5, 1, 0)),
        RandomForestBaseline(lookback_window=5),
        SVRBaseline(lookback_window=5)
    ]

    results = {}

    # Grafikon
    plt.figure(figsize=(16, 9))
    plt.plot(test_data.index, test_data.values, label='VALÓS ÁR (Actual)', color='black', linewidth=3, zorder=10)

    print("\nBaseline Modellek futtatása...")

    for model in models:
        try:
            res = model.run(train_data, test_data)
            results[model.model_name] = res

            preds = res['predictions']
            actuals = res['actual']  # a visszakapott adatot használjuk

            # Grafikon igazítás
            plot_index = test_data.index
            if len(preds) < len(test_data):
                diff = len(test_data) - len(preds)
                plot_index = test_data.index[diff:]

            # --- Irányhelyesség számítása ---
            acc = calculate_directional_accuracy(actuals, preds)

            plt.plot(plot_index, preds, label=f"{model.model_name} (RMSE: {res['rmse']:.1f}, Acc: {acc:.1f}%)")

            print(f" {model.model_name:<20} -> RMSE: {res['rmse']:.2f} | Irányhelyesség: {acc:.2f}%")

        except Exception as e:
            print(f" HIBA ennél: {model.model_name}: {e}")

    # Mentés
    plt.title(f'{TARGET_SYMBOL} Baseline Összehasonlítás (RMSE és Irányhelyesség)', fontsize=16)
    plt.xlabel('Dátum')
    plt.ylabel(f'Árfolyam ({TARGET_COL} USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = 'results/closeprice_baseline_comparison.png'
    plt.savefig(save_path)
    print(f"\nKész! Grafikon elmentve ide: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()