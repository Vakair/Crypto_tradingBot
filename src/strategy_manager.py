import ccxt
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
import os
import statsmodels.api as sm
from datetime import datetime
from tensorflow.keras.models import load_model
from dotenv import load_dotenv


class StrategyManager:
    def __init__(self):
        # API Kapcsolat (Binance Testnet)
        load_dotenv()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.secret_key = os.getenv('BINANCE_SECRET_KEY')

        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.exchange.set_sandbox_mode(True)  # TESTNET!

        #Modellek betöltése
        print("Modellek betöltése...")
        self.daily_model = load_model('models/gru_model.keras')
        self.daily_scaler = joblib.load('models/scaler_features.pkl')

        self.hourly_model = load_model('models/gru_model_hourly.keras')
        self.hourly_scaler = joblib.load('models/scaler_hourly.pkl')

        # Állapotfájl
        self.state_file = 'bot_state_multi.json'
        self.state = self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)

        # Ha NEM létezik
        default_state = {
            "swing": {"in_position": False, "entry_price": 0.0, "amount": 0.0},
            "daytrade": {"in_position": False, "entry_price": 0.0, "amount": 0.0},
            "pair": {"in_position": False, "type": None}
        }

        # ÉS AZONNAL EL IS MENTJÜK!
        print(f"Új állapotfájl létrehozása: {self.state_file}")
        with open(self.state_file, 'w') as f:
            json.dump(default_state, f)

        return default_state

    def save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)

    def get_data(self, tickers, period, interval):
        try:
            # Yahoo-ról szedjük az elemzéshez
            if isinstance(tickers, str): tickers = [tickers]
            df = yf.download(tickers, period=period, interval=interval, progress=False)

            # Formázás
            if len(tickers) == 1:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
                df = df.rename(columns={'Close': 'close'})
                return df[['close']].copy()
            else:
                return df['Close'].dropna()
        except:
            return pd.DataFrame()

    # --- KERESKEDÉSI MOTOR (EXECUTION) ---

    def place_order(self, symbol, side, amount_usdt=None, amount_coin=None):
        """Univerzális order küldő"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']

            if side == 'buy':
                # Ha USDT-ben adjuk meg, átszámoljuk coinra
                if amount_usdt:
                    amount = amount_usdt / price
                else:
                    amount = amount_coin

                print(f"   🟢 ORDER: VÉTEL {amount:.5f} {symbol} @ ${price}")
                order = self.exchange.create_market_buy_order(symbol, amount)
                return order, price, order['amount']

            elif side == 'sell':
                # Eladásnál mindent eladunk, ami van
                if amount_coin:
                    amount = amount_coin
                else:
                    balance = self.exchange.fetch_balance()
                    base_currency = symbol.split('/')[0]
                    amount = balance[base_currency]['free']

                if amount > 0:
                    print(f"   🔴 ORDER: ELADÁS {amount:.5f} {symbol} @ ${price}")
                    order = self.exchange.create_market_sell_order(symbol, amount)
                    return order, price, amount
                else:
                    print("!!!!!! Nincs mit eladni. !!!!!!!!")
                    return None, price, 0

        except Exception as e:
            print(f"!!!!! API Hiba: {e} !!!!!!!!")
            return None, 0, 0

    # ==========================
    # SWING TRADE LOGIKA (napi adatok)
    # ==========================
    def run_swing_strategy(self):
        print("\n--- SWING TRADE (Daily) ---")
        df = self.get_data("BTC-USD", "100d", "1d")
        if df.empty: return "Nincs adat"

        # Feature Engineering (Standard)
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

        # Előkészítés a modellnek
        last_data = df.iloc[-14:][['target_return', 'Dist_SMA10', 'Dist_SMA50', 'RSI', 'Momentum_3D']].values
        scaled = self.daily_scaler.transform(last_data).reshape(1, 14, 5)
        prob = self.daily_model.predict(scaled, verbose=0)[0][0]

        current_price = df['close'].iloc[-1]
        print(f"   Ár: ${current_price:.2f} | Modell: {prob:.4f}")

        # Döntés és Végrehajtás
        status = "HOLD"

        # VÉTEL
        if prob > 0.55 and not self.state['swing']['in_position']:
            print("VÉTEL!")
            order, price, amt = self.place_order('BTC/USDT', 'buy', amount_usdt=500)  # 500 dollárért
            if order:
                self.state['swing'] = {"in_position": True, "entry_price": price, "amount": amt}
                self.save_state()
                status = "BOUGHT"

        # ELADÁS
        elif prob < 0.45 and self.state['swing']['in_position']:
            print("ELADÁS!")
            order, price, amt = self.place_order('BTC/USDT', 'sell', amount_coin=self.state['swing']['amount'])
            if order:
                self.state['swing'] = {"in_position": False, "entry_price": 0.0, "amount": 0.0}
                self.save_state()
                status = "SOLD"

        # STOP LOSS (Swingnél 10%)
        elif self.state['swing']['in_position']:
            entry = self.state['swing']['entry_price']
            if current_price < entry * 0.90:
                print("STOP LOSS!")
                self.place_order('BTC/USDT', 'sell', amount_coin=self.state['swing']['amount'])
                self.state['swing'] = {"in_position": False, "entry_price": 0.0, "amount": 0.0}
                self.save_state()
                status = "STOP LOSS"

        return f"{status} (Prob: {prob:.2f})"

    # ==========================
    #  DAYTRADE LOGIKA
    # ==========================
    def run_daytrade_strategy(self):
        print("\n--- 🏎️ DAYTRADE (Hourly) ---")
        df = self.get_data("BTC-USD", "5d", "1h")
        if df.empty: return "Nincs adat"

        # Feature Engineering (V1 Simple)
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

        last_data = df.iloc[-14:][['target_return', 'Dist_SMA10', 'Dist_SMA50', 'RSI', 'Momentum_3D']].values
        scaled = self.hourly_scaler.transform(last_data).reshape(1, 14, 5)
        prob = self.hourly_model.predict(scaled, verbose=0)[0][0]

        current_price = df['close'].iloc[-1]
        print(f"   Ár: ${current_price:.2f} | Modell: {prob:.4f}")

        status = "HOLD"

        # VÉTEL (Kisebb pénzzel)
        if prob > 0.52 and not self.state['daytrade']['in_position']:
            print("DAYTRADE VÉTEL!")
            order, price, amt = self.place_order('BTC/USDT', 'buy', amount_usdt=300)
            if order:
                self.state['daytrade'] = {"in_position": True, "entry_price": price, "amount": amt}
                self.save_state()
                status = "BOUGHT"

        # ELADÁS
        elif prob < 0.48 and self.state['daytrade']['in_position']:
            print("DAYTRADE ELADÁS!")
            order, price, amt = self.place_order('BTC/USDT', 'sell', amount_coin=self.state['daytrade']['amount'])
            if order:
                self.state['daytrade'] = {"in_position": False, "entry_price": 0.0, "amount": 0.0}
                self.save_state()
                status = "SOLD"

        # SZIGORÚ STOP LOSS (3%)
        elif self.state['daytrade']['in_position']:
            entry = self.state['daytrade']['entry_price']
            if current_price < entry * 0.97:
                print("DAYTRADE STOP!")
                self.place_order('BTC/USDT', 'sell', amount_coin=self.state['daytrade']['amount'])
                self.state['daytrade'] = {"in_position": False, "entry_price": 0.0, "amount": 0.0}
                self.save_state()
                status = "STOP LOSS"

        return f"{status} (Prob: {prob:.2f})"

    # ==========================
    #PAIR TRADING
    # ==========================
    def run_pair_strategy(self):
        print("\n--- 👯 PAIR TRADING (ETH-SOL) ---")
        t1, t2 = "ETH-USD", "SOL-USD"
        df = self.get_data([t1, t2], "25d", "1h")
        if len(df) < 168: return "Nincs elég adat"

        # Statisztika (Rolling)
        y = df[t1].values
        x = df[t2].values

        # OLS Regresszió az elmúlt 168 órán
        y_win = y[-168:]
        x_win = x[-168:]
        x_win_c = sm.add_constant(x_win)
        try:
            model = sm.OLS(y_win, x_win_c).fit()
            beta = model.params[1]
        except:
            beta = 1.0

        current_spread = y[-1] - (beta * x[-1])
        spread_history = y_win - (beta * x_win)
        z = (current_spread - np.mean(spread_history)) / np.std(spread_history)

        print(f"   Z-Score: {z:.2f} | Hedge Ratio: {beta:.2f}")

        status = "HOLD"

        # BELÉPŐ: LONG SPREAD (ETH olcsó / SOL drága) -> Veszünk ETH-t
        # Megjegyzés: Spot-on nem tudunk SOL-t shortolni, csak ha van.
        # Egyszerűsítve: Csak az ETH lábat kötjük meg.
        if z < -2.0 and not self.state['pair']['in_position']:
            print("LONG SPREAD (Buy ETH)")
            order, price, amt = self.place_order('ETH/USDT', 'buy', amount_usdt=200)
            if order:
                self.state['pair'] = {"in_position": True, "type": "long_spread", "entry_z": z}
                self.save_state()
                status = "ENTERED LONG SPREAD"

        # KILÉPŐ: Ha Z visszatér 0 közelébe
        elif abs(z) < 0.5 and self.state['pair']['in_position']:
            print("EXIT PAIR (Profit Take)")
            # Mivel csak ETH-t vettünk, azt adjuk el
            self.place_order('ETH/USDT', 'sell')
            self.state['pair'] = {"in_position": False, "type": None}
            self.save_state()
            status = "CLOSED POSITIONS"

        return f"{status} (Z: {z:.2f})"