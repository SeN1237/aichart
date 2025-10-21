import pandas as pd
import numpy as np

def get_forecast(series: pd.Series) -> float:
    """
    Prosta prognoza ceny na 6 miesięcy.
    Argument: pandas.Series (historia cen).
    Zwraca: float (prognozowana cena).
    """

    try:
        s = series.dropna()

        if s.empty:
            return np.nan

        last_price = s.iloc[-1].item()


        # tu możesz dać dowolny prosty model — np. średnia + losowa zmiana
        forecast_price = last_price * (1 + np.random.uniform(-0.15, 0.15))

        return float(forecast_price)

    except Exception as e:
        print(f"[BŁĄD get_forecast] {e}")
        return np.nan

