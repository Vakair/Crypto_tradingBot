import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout

# --- KONFIGURÁCIÓ ---
DATA_PATH = 'data/top_100_cryptos_with_correct_network.csv'
TARGET_SYMBOL = 'BTCUSDT'
WINDOW_SIZE = 60
EPOCHS = 20
BATCH_SIZE = 32


def load_and_prepare_data():
    df = pd.read_csv(DATA_PATH)
    df = df[df['symbol'] == TARGET_SYMBOL].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    data = df[['close']].dropna().values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(WINDOW_SIZE, len(scaled_data)):
        X.append(scaled_data[i - WINDOW_SIZE:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    split = int(len(X) * 0.9)
    return X[:split], y[:split], X[split:], y[split:], scaler, df.index[split + WINDOW_SIZE:]


def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
    elif model_type == 'GRU':
        model.add(GRU(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=False))

    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def main():
    X_train, y_train, X_test, y_test, scaler, test_dates = load_and_prepare_data()
    input_shape = (X_train.shape[1], 1)

    results = {}

    # modell tanitas
    for m_type in ['LSTM', 'GRU']:

        model = build_model(m_type, input_shape)

        start_time = time.time()
        history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test),
                            verbose=1)
        train_time = time.time() - start_time

        preds_scaled = model.predict(X_test)
        preds = scaler.inverse_transform(preds_scaled)
        actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

        rmse = np.sqrt(np.mean((preds - actuals) ** 2))

        results[m_type] = {
            'time': train_time,
            'rmse': rmse,
            'preds': preds.flatten(),
            'actuals': actuals.flatten(),
            'loss_history': history.history['loss']
        }

        print(f" {m_type}Idő: {train_time:.2f} mp | RMSE: {rmse:.2f} USD")

    # vizualizacio
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(results['LSTM']['loss_history'], label='LSTM Loss', color='blue')
    plt.plot(results['GRU']['loss_history'], label='GRU Loss', color='red')
    plt.title('Tanulási Sebesség Összehasonlítása')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Hiba')
    plt.legend()

    plt.subplot(1, 2, 2)
    plot_len = min(100, len(test_dates))  # utolsó 100 nap
    plt.plot(test_dates[-plot_len:], results['LSTM']['actuals'][-plot_len:], label='Valós Ár', color='black',
             linewidth=2)
    plt.plot(test_dates[-plot_len:], results['LSTM']['preds'][-plot_len:],
             label=f"LSTM (RMSE: {results['LSTM']['rmse']:.0f})", color='blue', alpha=0.7)
    plt.plot(test_dates[-plot_len:], results['GRU']['preds'][-plot_len:],
             label=f"GRU (RMSE: {results['GRU']['rmse']:.0f})", color='red', alpha=0.7)
    plt.title('Árbecslés: LSTM vs GRU (Utolsó 100 nap)')
    plt.xticks(rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/lstm_vs_gru_price.png')
    print("\nGrafikon mentve: lstm_vs_gru_price.png")
    plt.show()


if __name__ == "__main__":
    main()