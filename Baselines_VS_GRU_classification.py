import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMA

# Import baselines
from src.baselines.linear_regression import LinearRegressionBaseline
from src.baselines.random_forest import RandomForestBaseline
from src.baselines.svr import SVRBaseline
from src.data_processor import DataProcessor


#gyorsitott arima
class FastArimaBaseline:
    def __init__(self, order=(5, 0, 0)):
        self.order = order
        self.model_name = "ARIMA (Fast)"

    def run(self, train_data, test_data):
        # gyorsitott ml
        train_vals = train_data.values if hasattr(train_data, 'values') else train_data
        test_vals = test_data.values if hasattr(test_data, 'values') else test_data

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train_vals, order=self.order)
            fitted_model = model.fit()

            full_vals = np.concatenate([train_vals, test_vals])
            applied_model = fitted_model.apply(full_vals)
            predictions = applied_model.predict(start=len(train_vals), end=len(full_vals) - 1)

        return {
            'predictions': predictions,
            'actual': test_vals
        }


DATA_PATH = 'data/top_100_cryptos_with_correct_network.csv'
MODEL_PATH = 'models/gru_model.keras'
WINDOW_SIZE = 14
TEST_SPLIT = 0.1


def main():
    # gru -> osszes adat ez az elonye
    processor = DataProcessor(DATA_PATH, window_size=WINDOW_SIZE, test_split=TEST_SPLIT)
    X_train, y_train_binary, X_test, y_test_binary, test_dates = processor.load_and_process()

    #baseline adatok
    df = pd.read_csv(DATA_PATH)
    df = df[df['symbol'] == 'BTCUSDT'].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    df['target_return'] = df['close'].pct_change()
    series = df['target_return'].dropna()

    # 90% Train - 10% Test
    split_idx = int(len(series) * (1 - TEST_SPLIT))
    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]

    baselines = [
        FastArimaBaseline(order=(5, 0, 0)),
        LinearRegressionBaseline(lookback_window=WINDOW_SIZE),
        RandomForestBaseline(lookback_window=WINDOW_SIZE),
        SVRBaseline(lookback_window=WINDOW_SIZE)
    ]

    results = {}

   # basline modellek
    for model in baselines:

        try:
            res = model.run(train_series, test_series)

            #  hozam > 0 = UP (1), DOWN (0)
            preds_continuous = res['predictions']
            actuals_continuous = res['actual']

            preds_binary = (np.array(preds_continuous) > 0).astype(int)
            actuals_binary = (np.array(actuals_continuous) > 0).astype(int)

            acc = accuracy_score(actuals_binary, preds_binary)
            results[model.model_name] = acc * 100
        except Exception as e:
            print(f"Hiba {model.model_name} esetén: {e}")


    # GRU
    if os.path.exists(MODEL_PATH):
        gru_model = load_model(MODEL_PATH)
        gru_probs = gru_model.predict(X_test, verbose=0).flatten()

        gru_binary = (gru_probs > 0.5).astype(int)
        gru_acc = accuracy_score(y_test_binary, gru_binary)
        results['GRU (Deep Learning)'] = gru_acc * 100

    else:
        print("HIBA")

    # kiiratas, vizualizacio
    print("\n VÉGEREDMÉNY")
    for name, acc in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f" - {name:<30}: {acc:.2f}%")

    # Bar chart rajzolása
    plt.figure(figsize=(10, 6))

    colors = ['#ff4c4c' if 'GRU' in name else '#888888' for name in results.keys()]
    bars = plt.bar(results.keys(), results.values(), color=colors)
    plt.axhline(50, color='black', linestyle='--', alpha=0.5, label='Pénzfeldobás (50%)')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.1f}%", ha='center', va='bottom',
                 fontweight='bold')

    plt.title('Modellek Pontossága Iránybecslésben (Classification Accuracy)')
    plt.ylabel('Pontosság (%)')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(results.values()) + 10)
    plt.legend()
    plt.tight_layout()

    plt.savefig('results/classifier_comparison.png')
    print("\nGrafikon mentve: classifier_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()