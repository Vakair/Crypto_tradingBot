import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Backtester:
    def __init__(self, initial_capital=10000, transaction_fee=0.001, stop_loss_pct=0.05):
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.stop_loss_pct = stop_loss_pct

    def run_strategy(self, returns, signals):
        """
        Look-Ahead Safe Backtesting.
        """
        capital = [self.initial_capital]
        in_position = False
        trade_count = 0

        # Look-ahead safe loop (utolsó előtti napig)
        for i in range(len(returns) - 1):
            current_signal = signals[i]
            next_return = returns[i + 1]  # A döntés a holnapi hozamra vonatkozik!

            # --- STOP LOSS LOGIKA ---
            force_sell = False
            if in_position and next_return < -self.stop_loss_pct:
                force_sell = True

            #1 Kereskedési döntés
            # VÉTEL
            if current_signal == 1 and not in_position:
                capital[-1] *= (1 - self.transaction_fee)
                in_position = True
                trade_count += 1

            # ELADÁS
            elif (current_signal == 0 or force_sell) and in_position:
                capital[-1] *= (1 - self.transaction_fee)
                in_position = False
                trade_count += 1

            #2 Pénzmozgás (Holnap)
            if in_position:
                new_balance = capital[-1] * (1 + next_return)
                capital.append(new_balance)
            else:
                capital.append(capital[-1])

        return np.array(capital), trade_count

    def calculate_metrics(self, equity_curve):
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        if returns.std() == 0:
            sharpe_ratio = 0
        else:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365)
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        return sharpe_ratio, max_drawdown

    def plot_equity_curves(self, dates, results_dict, title="Backtest Eredmények"):
        plt.figure(figsize=(12, 8))

        for name, data in results_dict.items():
            if isinstance(data, dict) and 'equity' in data:
                equity = data['equity']
                trade_count = data['trades']
            else:
                equity = data
                trade_count = 0

            #Egyszerűen annyi dátumot veszünk, ahány adatpontunk van
            #Így garantáltan egyezni fog a méret (x és y)
            plot_dates = dates[:len(equity)]

            final_val = equity[-1]
            label = f"{name} (${final_val:,.0f} | {trade_count} kötés)"

            if "GRU" in name:
                plt.plot(plot_dates, equity, label=label, linewidth=3, color='red')
            elif "Buy & Hold" in name:
                plt.plot(plot_dates, equity, label=label, linewidth=2, color='black', linestyle='--')
            else:
                plt.plot(plot_dates, equity, label=label, alpha=0.6)

        plt.title(title, fontsize=16)
        plt.xlabel("Dátum")
        plt.ylabel("Portfólió Érték (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/final_profit_chart.png')
        print(f"\nGrafikon elmentve: results/final_profit_chart.png")
        plt.show()