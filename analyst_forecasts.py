import requests
import pandas as pd

API_KEY = "4lEINIKcOtY7dQ1K1lCb0tbsvzDld4uU"  # <-- wstaw tutaj swój klucz API FMP

def get_forecasts(tickers):
    """
    Pobiera podsumowanie cen docelowych dla listy tickerów z FMP.
    Zwraca DataFrame z kolumnami: Ticker, Średnia prognoza, Prognoza maks, Prognoza min, Liczba analityków
    """
    all_data = []
    
    for ticker in tickers:
        url = f"https://financialmodelingprep.com/stable/price-target-summary?symbol={ticker}&apikey={API_KEY}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            all_data.append({
                "Ticker": ticker,
                "Średnia prognoza": data.get("priceTargetAverage"),
                "Prognoza maks": data.get("priceTargetHigh"),
                "Prognoza min": data.get("priceTargetLow"),
                "Liczba analityków": data.get("numberOfAnalysts")
            })
            
        except requests.exceptions.HTTPError as e:
            print(f"Błąd przy pobieraniu {ticker}: {e}")
            all_data.append({
                "Ticker": ticker,
                "Średnia prognoza": None,
                "Prognoza maks": None,
                "Prognoza min": None,
                "Liczba analityków": None
            })
    
    df = pd.DataFrame(all_data)
    return df

