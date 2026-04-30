import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Backtester:
    def __init__(self, initial_capital=10000, transaction_fee=0.001, stop_loss_pct=0.05):
        self.initial_capital = initial_capital
        self.transaction_fee = transaction_fee
        self.stop_loss_pct = stop_loss_pct

    def run_strategy(self, returns, signals):
        capital = [self.initial_capital]
        current_position = 0  # 0: Cash, 1: Long, -1: Short
        trade_count = 0

        for i in range(len(returns) - 1):
            signal = signals[i]
            next_return = returns[i + 1]  # Hozam T és T+1 között

            # 1. LÉPÉS: Kereskedési döntés a modell JELZÉSE alapján (Mielőtt a piac mozog)
            if signal != current_position:
                # Ha volt nyitott pozíció, azt lezárjuk (költség)
                if current_position != 0:
                    capital[-1] *= (1 - self.transaction_fee)

                # Ha nyitunk újat
                if signal != 0:
                    capital[-1] *= (1 - self.transaction_fee)
                    trade_count += 1

                current_position = signal

            # 2. LÉPÉS: Pénzmozgás és Stop-Loss (A piac megmozdul)
            if current_position == 1:  # LONG
                if next_return < -self.stop_loss_pct:
                    # Kiestünk a stop-loss miatt! A VESZTESÉGET LEVONJUK!
                    capital.append(capital[-1] * (1 - self.stop_loss_pct))
                    current_position = 0  # Kényszerzárás a következő körre
                    capital[-1] *= (1 - self.transaction_fee)  # Tőzsdei díj a zárásért
                else:
                    capital.append(capital[-1] * (1 + next_return))

            elif current_position == -1:  # SHORT
                if next_return > self.stop_loss_pct:
                    # Short stop-loss! (Nekünk a növekedés a rossz)
                    capital.append(capital[-1] * (1 - self.stop_loss_pct))
                    current_position = 0  # Kényszerzárás
                    capital[-1] *= (1 - self.transaction_fee)
                else:
                    capital.append(capital[-1] * (1 - next_return))

            else:  # CASH
                # Készpénzben ülünk, nincs változás
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
            if isinstance(data, dict):
                equity = data['equity']
                trade_count = data['trades']
            else:
                equity = data
                trade_count = 0

            plot_dates = dates[:len(equity)]
            final_val = equity[-1]
            label = f"{name} (${final_val:,.0f} | {trade_count} kötés)"

            if "Long/Short" in name:
                plt.plot(plot_dates, equity, label=label, linewidth=3, color='purple')
            elif "Buy & Hold" in name:
                plt.plot(plot_dates, equity, label=label, linewidth=2, color='black', linestyle='--')
            else:
                plt.plot(plot_dates, equity, label=label, alpha=0.6)

        plt.title(title, fontsize=16)
        plt.xlabel("Dátum")
        plt.ylabel("Portfólió Érték (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/final_profit_chart_LONG_SHORT.png')
        print(f"\nGrafikon elmentve: results/final_profit_chart_LONG/SHORT.png")
        plt.show()