Vagjunkbele/
├── .venv/
│
├── data/
│   ├── enhanced_data.csv
│   ├── enhanced_data_returns.csv
│   ├── top_100_cryptos_with_correct_network.csv
│   └── readme.md
│
├── models/
│   ├── gru_model.keras
│   ├── gru_model_hourly.keras
│   ├── scaler_features.pkl
│   └── scaler_hourly.pkl
│
├── results/
│   ├── baseline_returns.png
│   ├── closeprice_baseline_comparison.png
│   ├── closeprice_lstm_result_fixed.png
│   ├── final_profit_chart.png
│   ├── gru_classification.png
│   ├── lstm_classification.png
│   ├── lstm_returns_final.png
│   ├── pair_heatmap.png
│   ├── pair_trading_BTC-USD_ETH-USD.png
│   ├── pair_trading_BTC-USD_SOL-USD.png
│   └── pair_trading_ETH-USD_SOL-USD.png
│
├── src/
│   ├── baselines/
│   │   ├── arima.py
│   │   ├── linear_regression.py
│   │   ├── naive.py
│   │   ├── random_forest.py
│   │   ├── svr.py
│   │   └── readme.md
│   │
│   ├── backtester.py
│   ├── backtester_LONG.py
│   ├── closeprice_data_processor.py
│   ├── data_processor.py
│   ├── strategy_manager.py
│
├── .env
├── app.py
├── bot_state.json
├── bot_state_multi.json
├── closeprice_lstm_runner.py
├── closeprice_main.py
├── debug_data.py
├── find_pairs.py
├── gru_runner.py
├── live_trader.py
├── main.py
├── pair_trading.md
├── README.md
├── run_backtest.py
├── run_backtest_hourly.py
├── run_backtest_LONG.py
├── run_pair_trading.py
├── run_superbot.py
├── save_model.py
├── strategies.json
├── stress_test.py
├── test_connection.py
├── trade_history.csv
└── train_daytrade_model.py
