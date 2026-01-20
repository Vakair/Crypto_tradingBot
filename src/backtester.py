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
        Long/Short Backtesting.
        signals:
            1 = LONG
            -1 = SHORT
            0 = CASH
        """
        capital = [self.initial_capital]
        current_position = 0  # 0: Cash, 1: Long, -1: Short
        trade_count = 0

        # Look-ahead safe loop
        for i in range(len(returns) - 1):
            signal = signals[i]
            next_return = returns[i + 1]  # Holnapi hozam

            # --- STOP LOSS LOGIKA ---
            force_close = False
            # Longnál: ha esik az ár
            if current_position == 1 and next_return < -self.stop_loss_pct:
                force_close = True
            # Shortnál: ha NŐ az ár (az nekünk rossz)
            elif current_position == -1 and next_return > self.stop_loss_pct:
                force_close = True

            #1 Kereskedési döntés (Váltás)
            # Ha STOP-LOSS van, vagy a jel más, mint a mostani pozíció
            if force_close or (signal != current_position):

                # Ha volt nyitott pozíció, azt le kell zárni (költség)
                if current_position != 0:
                    capital[-1] *= (1 - self.transaction_fee)

                # Ha nyitunk újat (és nem csak stop-loss miatti kényszerzárás van)
                if not force_close and signal != 0:
                    capital[-1] *= (1 - self.transaction_fee)
                    trade_count += 1

                # Ha force close volt, akkor készpénzbe megyünk, amúgy a jel szerint
                current_position = 0 if force_close else signal

            #2 Pénzmozgás (Holnap)
            if current_position == 1:  # LONG
                capital.append(capital[-1] * (1 + next_return))
            elif current_position == -1:  # SHORT (Inverz hozam)
                capital.append(capital[-1] * (1 - next_return))
            else:  # CASH
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
        plt.savefig('results/final_profit_chart_LONG/SHORT.png')
        print(f"\nGrafikon elmentve: results/final_profit_chart_LONG/SHORT.png")
        plt.show()