import os
import joblib  # A skálázók mentéséhez kell (pip install joblib ha nincs)
from src.data_processor import DataProcessor
from gru_runner import build_gru_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Mappa a modelleknek
os.makedirs('models', exist_ok=True)

DATA_PATH = 'data/top_100_cryptos_with_correct_network.csv'
WINDOW_SIZE = 14
TEST_SPLIT = 0.1


def main():
    print(" MODELL VÉGLEGESÍTÉSE ÉS MENTÉSE...\n")

    #Adatok betöltése
    processor = DataProcessor(DATA_PATH, window_size=WINDOW_SIZE, test_split=TEST_SPLIT)
    X_train, y_train, X_test, y_test, _ = processor.load_and_process()

    #  Modell Tanítása
    print("   Tanítás indítása...")
    model = build_gru_model((X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Most verbose=1, hogy lássuk a végét
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[early_stop, reduce_lr], verbose=1)

    #MENTÉS
    print("\n Mentés folyamatban...")

    #A Keras modell mentése
    model.save('models/gru_model.keras')
    print("Modell elmentve: models/gru_model.keras")

    #Skálázók mentése (Hogy a jövőbeli adatokat is ugyanígy tudjuk skálázni)
    joblib.dump(processor.scaler_features, 'models/scaler_features.pkl')
    # joblib.dump(processor.scaler_target, 'models/scaler_target.pkl') # Ez most nem kell, mert osztályozunk
    print("    Skálázó elmentve: models/scaler_features.pkl")




if __name__ == "__main__":
    main()