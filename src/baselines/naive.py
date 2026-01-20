import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class NaiveBaseline:
    """
    Naive Forecast (Vagy Persistence Model).

    MŰKÖDÉS:
    Ez a modell nem 'számol' semmit. Egyszerűen azt mondja:
    "A holnapi árfolyam pontosan ugyanannyi lesz, mint a mai."

    MIÉRT KELL EZ?
    Ez a 'nulla pont'.
    """

    def __init__(self):
        # A modell neve, ami megjelenik a grafikonon
        self.model_name = "Naive Forecast"

    def run(self, train_data: pd.Series, test_data: pd.Series):
        print(f"--- {self.model_name} Futtatása... ---")

        #Előkészület: Kell az utolsó ismert adat a tanító halmazból
        #Ez lesz az első jóslatunk a teszt időszak első napjára.
        history = train_data.iloc[-1]

        predictions = []  # Ide gyűjtjük a jóslatokat

        # Ciklus a teszt adatokon (pl. 60 napon át)
        for i in range(len(test_data)):
            # A JÓSLÁS: A jóslat (yhat) egyszerűen a legutolsó ismert ár (history)
            yhat = history
            predictions.append(yhat)

            #A következő naphoz a 'history' már a mostani tényleges ár lesz.
            # Tehát mindig egyet lépünk előre: a mai valós adat lesz a holnapi jóslat alapja.
            history = test_data.iloc[i]

        #Hiba számítása (RMSE és MAE)
        # RMSE: Mennyit tévedtünk átlagosan (négyzetes hiba gyöke - kiemeli a nagy tévedéseket)
        rmse = np.sqrt(mean_squared_error(test_data, predictions))
        #MAE: Átlagosan hány dollárt tévedtünk
        mae = mean_absolute_error(test_data, predictions)

        # Visszaadjuk az eredményeket egy szótárban (dictionary)
        return {
            'predictions': predictions,
            'rmse': rmse,
            'mae': mae,
            'actual': test_data.values
        }