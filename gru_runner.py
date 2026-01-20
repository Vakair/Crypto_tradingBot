import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as pd_tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization  # GRU import!
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.data_processor import DataProcessor

# --- KONFIGURÁCIÓ  ---
DATA_PATH = 'data/top_100_cryptos_with_correct_network.csv'
EPOCHS = 100
BATCH_SIZE = 32
WINDOW_SIZE = 14
DROPOUT_RATE = 0.3


def build_gru_model(input_shape):
    model = Sequential()

    #GRU
    model.add(GRU(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(DROPOUT_RATE))
    model.add(BatchNormalization())

    #GRU Réteg
    model.add(GRU(units=32, return_sequences=False))
    model.add(Dropout(DROPOUT_RATE))
    model.add(BatchNormalization())

    # Output (Sigmoid = 0-1 valószínűség)
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def main():
    processor = DataProcessor('data/top_100_cryptos_with_correct_network.csv',
                              window_size=WINDOW_SIZE, test_split=0.1)

    X_train, y_train, X_test, y_test, test_dates = processor.load_and_process()

    model = build_gru_model((X_train.shape[1], X_train.shape[2]))

    #EarlyStop: Ha nem javul, álljon le.
    #ReduceLROnPlateau: Ha elakad, csökkentse a tanulási sebességet (finomhangolás)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_data=(X_test, y_test),
                        callbacks=[early_stop, reduce_lr],
                        verbose=1)

    # Kiértékelés
    y_pred_prob = model.predict(X_test)
    predictions = (y_pred_prob > 0.5).astype(int).flatten()
    actuals = y_test.flatten()

    accuracy = np.mean(predictions == actuals) * 100

    print(f"\n GRU (Classification) EREDMÉNYEK:")
    print(f"   Pontosság: {accuracy:.2f}%")


    # Vizualizáció
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Test Acc')
    plt.axhline(55, color='red', linestyle='--', label='Baseline (55%)')
    plt.title(f'GRU Modell Pontosság (Cél: >55%)')
    plt.legend()
    plt.savefig('results/gru_classification.png')
    plt.show()


if __name__ == "__main__":
    main()