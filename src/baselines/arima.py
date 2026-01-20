import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings


warnings.filterwarnings("ignore")


class ArimaBaseline:
    """
    ARIMA (AutoRegressive Integrated Moving Average).

    MŰKÖDÉS:

    1. AR (p=5): AutoRegresszió - a múltbeli árak hatása a jövőre.
    2. I (d=1): Integrálás - a trend kivonása (hogy stabil legyen az átlag).
    3. MA (q=0): Mozgóátlag - a múltbeli hibák korrigálása (most 0).


    """

    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model_name = "ARIMA"
        self.lookback_window = 0

    def run(self, train_data: pd.Series, test_data: pd.Series):
        print(f"--- {self.model_name} Futtatása")

        # A 'history' lista tárolja az összes ismert adatot.
        # Kezdetben csak a tanító adatokat tartalmazza.
        history = [x for x in train_data]
        predictions = []

        # WALK-FORWARD VALIDÁCIÓ:
        # Végigmegyünk a teszt napokon egyesével.
        for t in range(len(test_data)):
            #Létrehozzuk az ARIMA modellt az eddig ismert adatokon (history)
            model = ARIMA(history, order=self.order)

            #Illesztés (Fit) - ez a legidőigényesebb lépés
            model_fit = model.fit()

            #Jóslás 1 lépéssel előre (Forecast)
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)

            #A valós adatot (obs) hozzáadjuk a history-hoz,
            # hogy a következő napon már ezt is felhasználja a modell.
            obs = test_data.iloc[t]
            history.append(obs)

        # Hiba számítása
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        mae = mean_absolute_error(test_data, predictions)

        return {
            'predictions': predictions,
            'rmse': rmse,
            'mae': mae,
            'actual': test_data.values
        }