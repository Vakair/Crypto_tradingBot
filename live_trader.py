import ccxt
import pandas as pd
import numpy as np
import joblib
import os
import time
import sys
import json
import csv
from datetime import datetime
from dotenv import load_dotenv
from tensorflow.keras.models import load_model

# --- KONFIGURÁCIÓ ---
SYMBOL = 'BTC/USDT'
TIMEFRAME = '1d'
WINDOW_SIZE = 14
LONG_THRESHOLD = 0.55  # Vétel (
SHORT_THRESHOLD = 0.45  # Eladás (
STOP_LOSS_PCT = 0.05

# Fájlok
MODEL_PATH = 'models/gru_model.keras'
SCALER_PATH = 'models/scaler_features.pkl'
LOG_FILE = 'trade_history.csv'
STATE_FILE = 'bot_state.json'

# --- API  ---
load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
secret_key = os.getenv('BINANCE_SECRET_KEY')

if not api_key or not secret_key:
    print(" HIBA: Nincsenek kulcsok")
    sys.exit()

#ADAT SZOLGÁLTATÓ (MAINNET - Csak olvasás, publikus adat)
data_exchange = ccxt.binance({
    'enableRateLimit': True,
})

# KERESKEDŐ (TESTNET)
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})
exchange.set_sandbox_mode(True)


# --- MEMÓRIA ÉS NAPLÓZÁS ---
def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Date', 'Action', 'Price', 'Amount', 'Reason', 'Balance_USDT'])


def log_trade(action, price, amount, reason, balance):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, action, price, amount, reason, balance])
    print(f"   📝 Napló frissítve: {action} @ {price}")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"in_position": False, "entry_price": 0.0}


def save_state(in_position, entry_price):
    state = {"in_position": in_position, "entry_price": entry_price}
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)


# --- STANDARD FÜGGVÉNYEK ---

def calculate_features(df):
    df = df.copy()
    df['SMA_10'] = df['close'].rolling(10).mean()
    df['SMA_50'] = df['close'].rolling(50).mean()
    df['target_return'] = df['close'].pct_change()
    df['Dist_SMA10'] = df['close'] / df['SMA_10']
    df['Dist_SMA50'] = df['close'] / df['SMA_50']

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Momentum_3D'] = df['close'].pct_change(periods=3)
    df.dropna(inplace=True)
    return df


def get_market_data():
    try:
        #data_exchange (Mainnet) használata az adatokhoz!
        # Így lesz 1000 napnyi adatunk, működni fog az SMA_50.
        bars = data_exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=1000)

        if not bars: return None
        df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Adatletöltési hiba (Mainnet): {e}")
        return None


def prepare_data_for_model(df, scaler):
    feature_cols = ['target_return', 'Dist_SMA10', 'Dist_SMA50', 'RSI', 'Momentum_3D']
    last_data = df.iloc[-WINDOW_SIZE:].copy()

    if len(last_data) < WINDOW_SIZE:
        print(f"!!! Kevés adat: !!!{len(last_data)}")
        return None

    data_values = last_data[feature_cols].values
    data_scaled = scaler.transform(data_values)
    return data_scaled.reshape(1, WINDOW_SIZE, len(feature_cols))


# --- KERESKEDÉS VÉGREHAJTÁS ---

def execute_trade(action, current_price, reason="Model"):
    try:
        # Itt az 'exchange' (Testnet) használata a kereskedéshez!
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        btc_balance = balance['BTC']['free']

        state = load_state()

        if action == "BUY":
            risk_amount = 1000
            if usdt_balance > 20:
                amount_to_spend = min(usdt_balance * 0.98, risk_amount)
                amount_btc = amount_to_spend / current_price

                print(f"   🟢 VÉTEL: {amount_btc:.5f} BTC (Ok: {reason})")
                order = exchange.create_market_buy_order(SYMBOL, amount_btc)

                save_state(in_position=True, entry_price=order['average'])
                log_trade("BUY", order['average'], amount_btc, reason, usdt_balance)
            else:
                print("!!! Nincs elég USDT. !!!")

        elif action == "SELL":
            if btc_balance > 0.0005:
                print(f"   🔴 ELADÁS: {btc_balance:.5f} BTC (Ok: {reason})")
                order = exchange.create_market_sell_order(SYMBOL, btc_balance)

                save_state(in_position=False, entry_price=0.0)
                log_trade("SELL", order['average'], btc_balance, reason, usdt_balance + (btc_balance * current_price))
            else:
                if state['in_position']:
                    save_state(in_position=False, entry_price=0.0)
                    print(" !!! Szinkronizációs hiba javítva (State reset). !!!")
                else:
                    print(" !!! Nincs eladható BTC. !!!")

    except Exception as e:
        print(f"!!! Kereskedési hiba: {e} !!!")


# --- FŐ CIKLUS ---
def main():
    print(f"GRU BOT (Mainnet Adatok + Testnet Trade)...")
    init_log_file()

    if not os.path.exists(MODEL_PATH): return
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print("Rendszer éles. Figyelés...")

    while True:
        try:
            print("\n" + "=" * 50)
            print(f" {datetime.now().strftime('%H:%M:%S')} | Elemzés (Timeframe: {TIMEFRAME})...")

            # Adat (Mainnetről)
            df = get_market_data()
            if df is None:
                time.sleep(10)
                continue
            current_price = df.iloc[-1]['close']

            # Stop-Loss
            state = load_state()
            if state['in_position']:
                entry_price = state['entry_price']
                if current_price < entry_price * (1 - STOP_LOSS_PCT):
                    print(f" VÉSZFÉK (STOP-LOSS)! Beszálló: {entry_price}, Most: {current_price}")
                    execute_trade("SELL", current_price, reason="Stop-Loss")
                    time.sleep(60)
                    continue

                    # Modell
            df_features = calculate_features(df)
            X_input = prepare_data_for_model(df_features, scaler)

            if X_input is None:
                print("!!! Kevés adat az előkészítés után. !!!")
                time.sleep(10)
                continue

            prediction = model.predict(X_input, verbose=0)[0][0]
            print(f"Modell: {prediction:.4f} ({prediction * 100:.1f}%) | Ár: ${current_price:,.2f}")
            if state['in_position']:
                print(f"Pozícióban: IGEN (Vettünk: ${state['entry_price']:,.2f})")

            # Döntés
            action = "HOLD"
            reason = "Model"

            if prediction > LONG_THRESHOLD:
                if not state['in_position']:
                    action = "BUY"
                    print(f"VÉTELI JEL (> {LONG_THRESHOLD})")
                else:
                    print(" Hold (Már van pozíció)")

            elif prediction < SHORT_THRESHOLD:
                if state['in_position']:
                    action = "SELL"
                    print(f" ELADÁSI JEL (< {SHORT_THRESHOLD})")
                else:
                    print(" Hold (Nincs mit eladni)")
            else:
                print(f" Bizonytalan ({SHORT_THRESHOLD}-{LONG_THRESHOLD}) -> Tartás")

            #   Végrehajtás
            if action != "HOLD":
                execute_trade(action, current_price, reason)

            print("=" * 50)
            print("💤 Várakozás 60 másodpercig...")
            time.sleep(60)

        except KeyboardInterrupt:
            print("\n!! Leállítva. !!")
            break
        except Exception as e:
            print(f"\n!! Hiba: {e} !!")
            time.sleep(10)


if __name__ == "__main__":
    main()