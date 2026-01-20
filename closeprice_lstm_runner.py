import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as pd_tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from src.closeprice_data_processor import DataProcessor

# --- KONFIGURÁCIÓ ---
DATA_PATH = 'data/enhanced_data.csv'
EPOCHS = 50
BATCH_SIZE = 32
WINDOW_SIZE = 60
DROPOUT_RATE = 0.2


def build_lstm_model(input_shape):
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(DROPOUT_RATE))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(DROPOUT_RATE))

    model.add(Dense(25))
    model.add(Dense(1))


    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')

    return model


def main():
    processor = DataProcessor(DATA_PATH, window_size=WINDOW_SIZE, test_split=0.1)

    X_train, y_train, X_test, y_test, test_dates = processor.load_and_process()

    print("\n--- LSTM Modell építése... ---")
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))

    print("\n--- Tanítás indítása... ---")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )

    print("\n--- Kiértékelés... ---")
    predictions_scaled = model.predict(X_test)

    predictions = processor.inverse_transform_predictions(predictions_scaled)
    y_test_actual = processor.inverse_transform_actuals(y_test)

    rmse = np.sqrt(np.mean((predictions - y_test_actual) ** 2))


    #Valós változás (Holnapi ár - Mai ár)
    actual_change = y_test_actual[1:] - y_test_actual[:-1]

    #Modell által jósolt változás (Jósolt holnapi ár - Mai VALÓS ár)
    predicted_change = predictions[1:] - y_test_actual[:-1]

    # 3. Egyezik az előjel?
    hits = np.sign(actual_change) == np.sign(predicted_change)
    accuracy = np.mean(hits) * 100

    print(f"\n LSTM EREDMÉNYEK (Fixált):")
    print(f"   RMSE Hiba: {rmse:.2f} USD")
    print(f"   Irányhelyesség: {accuracy:.2f}%")

    # Vizualizáció
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Modell Tanulási Görbe (Huber Loss)')
    plt.legend()

    plt.subplot(2, 1, 2)
    # Biztosítjuk, hogy a hosszak egyezzenek a rajzolásnál
    plot_len = min(len(test_dates), len(y_test_actual))

    plt.plot(test_dates[:plot_len], y_test_actual[:plot_len], label='Valós Ár', color='black')
    plt.plot(test_dates[:plot_len], predictions[:plot_len], label='LSTM Becslés', color='blue')
    plt.title(f'Bitcoin Árfolyam - RMSE: {rmse:.0f}, Acc: {accuracy:.1f}%')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/closeprice_lstm_result_fixed.png')
    plt.show()


if __name__ == "__main__":
    main()