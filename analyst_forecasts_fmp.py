import yfinance as yf
import pandas as pd

def get_forecasts(tickers):
    """
    Pobiera rekomendacje analityków z Yahoo Finance dla listy tickerów.
    Zwraca DataFrame z kolumnami: Ticker, Konsensus, Kupuj, Trzymaj, Sprzedaj
    """
    all_data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            rec = stock.recommendations
            if rec is None:
                all_data.append({
                    "Ticker": ticker,
                    "Konsensus": None,
                    "Kupuj": None,
                    "Trzymaj": None,
                    "Sprzedaj": None
                })
                continue

            # filtrowanie ostatnich rekomendacji (ostatni rok)
            rec = rec[rec.index.year >= (pd.Timestamp.now().year - 1)]
            counts = rec['To Grade'].value_counts()
            buy = counts.get('Buy', 0)
            hold = counts.get('Hold', 0)
            sell = counts.get('Sell', 0)
            total = buy + hold + sell
            consensus = (buy - sell)/total*100 if total > 0 else None

            all_data.append({
                "Ticker": ticker,
                "Konsensus": consensus,
                "Kupuj": buy,
                "Trzymaj": hold,
                "Sprzedaj": sell
            })

        except Exception as e:
            print(f"Błąd przy pobieraniu {ticker}: {e}")
            all_data.append({
                "Ticker": ticker,
                "Konsensus": None,
                "Kupuj": None,
                "Trzymaj": None,
                "Sprzedaj": None
            })

    return pd.DataFrame(all_data)


