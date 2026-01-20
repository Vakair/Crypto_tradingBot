import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# --- KONFIGURÁCIÓ ---
PAIRS = [
    ('BTC-USD', 'SOL-USD'),
    ('ETH-USD', 'SOL-USD'),
    ('BTC-USD', 'ETH-USD')
]

TIMEFRAME = "1h"
PERIOD = "500d"  # Elmúlt 500 nap
WINDOW_SIZE = 168  # 1 hetes csúszóablak


def download_data(ticker1, ticker2):
    print(f"\n Adatok letöltése: {ticker1} vs {ticker2}...")
    try:
        df = yf.download([ticker1, ticker2], period=PERIOD, interval=TIMEFRAME, progress=False)['Close']
        df = df.dropna()
        if df.empty:
            print("!!! Üres adat. !!!")
            return None
        return df
    except Exception as e:
        print(f"!!! Hiba: {e} !!!")
        return None


def calculate_rolling_stats(df, t1, t2, window):
    """
    Hedge Ratio és Z-Score számítása csúszóablakkal.
    """
    z_scores = []

    # Adatok numpy tömbként a sebességért
    y = df[t1].values
    x = df[t2].values

    print(f"Statisztikák számolása (Rolling OLS)...")

    spreads = []

    for i in range(len(df)):
        if i < window:
            z_scores.append(0)  # Kezdetben nincs jel
            spreads.append(0)
            continue

        # Ablak kiválasztása
        y_window = y[i - window:i]
        x_window = x[i - window:i]

        # OLS Regresszió a Hedge Ratiohoz
        x_window_c = sm.add_constant(x_window)
        try:
            model = sm.OLS(y_window, x_window_c).fit()
            beta = model.params[1]  # Hedge Ratio
        except:
            beta = 1.0  # Fallback

        # Aktuális Spread
        spread = y[i] - (beta * x[i])
        spreads.append(spread)

        # Z-Score a spread alapján
        spread_window = np.array(spreads[i - window:i])

        # Biztonsági ellenőrzés (0-val osztás ellen)
        std_dev = np.std(spread_window)
        if std_dev == 0:
            z = 0
        else:
            z = (spread - np.mean(spread_window)) / std_dev

        # Extrém értékek vágása (hogy ne legyen végtelen)
        z = np.clip(z, -5, 5)
        z_scores.append(z)

    return np.array(z_scores)


def run_pair_strategy(ticker1, ticker2):
    #Adat
    df = download_data(ticker1, ticker2)
    if df is None: return

    # Z-Score
    df['z_score'] = calculate_rolling_stats(df, ticker1, ticker2, WINDOW_SIZE)

    #Stratégia (PnL számítás)
    positions_t1 = []
    positions_t2 = []
    current_pos = 0

    entry_threshold = 2.0
    exit_threshold = 0.5

    for z in df['z_score']:
        # SHORT SPREAD: Short T1, Long T2
        if z > entry_threshold:
            current_pos = -1
            # LONG SPREAD: Long T1, Short T2
        elif z < -entry_threshold:
            current_pos = 1
        # EXIT
        elif abs(z) < exit_threshold:
            current_pos = 0

        #Súlyozás (50-50%)
        # Nem a Hedge Ratióval szorzunk, hanem fix tőkeallokációval
        if current_pos == 1:
            positions_t1.append(0.5)  # Tőke 50%-a Long T1
            positions_t2.append(-0.5)  # Tőke 50%-a Short T2
        elif current_pos == -1:
            positions_t1.append(-0.5)  # Tőke 50%-a Short T1
            positions_t2.append(0.5)  # Tőke 50%-a Long T2
        else:
            positions_t1.append(0)
            positions_t2.append(0)

    df['pos_t1'] = positions_t1
    df['pos_t2'] = positions_t2

    #Profit Számítás (Reális)
    ret_t1 = df[ticker1].pct_change().fillna(0)
    ret_t2 = df[ticker2].pct_change().fillna(0)

    # A stratégia hozama = (Súly1 * Hozam1) + (Súly2 * Hozam2)
    strat_ret = (df['pos_t1'].shift(1) * ret_t1) + (df['pos_t2'].shift(1) * ret_t2)

    # Költségek (0.1% a teljes pozícióváltásra)
    trades = (df['pos_t1'].diff().abs() + df['pos_t2'].diff().abs()) / 2
    strat_ret = strat_ret - (trades * 0.001)

    # Kumulatív hozam
    cumulative = (1 + strat_ret).cumprod()

    #redmény
    total_profit = (cumulative.iloc[-1] - 1) * 100

    # Sharpe (évesítve)
    std_dev = strat_ret.std()
    if std_dev == 0:
        sharpe = 0
    else:
        sharpe = (strat_ret.mean() / std_dev) * np.sqrt(24 * 365)

    print(f"      EREDMÉNY ({ticker1}-{ticker2}):")
    print(f"      Profit: {total_profit:.2f}%")
    print(f"      Sharpe: {sharpe:.2f}")

    #Grafikon
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative, label='Pair Trading Strategy', color='purple')

    # Referencia (Buy & Hold Ticker 1)
    bh_cum = (1 + ret_t1).cumprod()
    plt.plot(bh_cum, label=f'Buy & Hold {ticker1}', alpha=0.3, color='gray')

    plt.title(f'Pair Trading: {ticker1} vs {ticker2} (Profit: {total_profit:.2f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"results/pair_trading_{ticker1}_{ticker2}.png"
    plt.savefig(filename)
    print(f"    Grafikon mentve: {filename}")
    plt.close()


def main():
    if not os.path.exists('results'): os.makedirs('results')
    print("PAIR TRADING BACKTEST (Fixed Logic)...")

    for p in PAIRS:
        run_pair_strategy(p[0], p[1])

    print("\n Kész.")


if __name__ == "__main__":
    main()