# simple_analyst_forecasts.py
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from candlestick_chart import get_candlestick_figure
import candlestick_chart as cc
from forecast_utils import get_forecast


TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "INTC", "AMD"
]


def _close_series(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    if "Close" in df.columns:
        s = df["Close"]
    elif "Adj Close" in df.columns:
        s = df["Adj Close"]
    else:
        return None
    s = s.dropna()
    return s if not s.empty else None

def _forecast_one(ticker: str):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
        s = _close_series(df)
        if s is None or len(s) < 10:
            return None

        last = s.iloc[-1].item()

        # 6 miesięcy ≈ 126 sesji
        if len(s) > 126:
            past = s.iloc[-126].item()
            sixm_ret = last / past - 1.0
        else:
            rets = s.pct_change().dropna()
            if rets.empty:
                return None
            mean = rets.mean()
            horizon = min(126, len(s))
            sixm_ret = (1.0 + mean) ** horizon - 1.0

        pred = round(last * (1.0 + sixm_ret), 2)
        return {
            "Ticker": ticker,
            "Cena": last,
            "Prognoza 6M": pred,
            "Zmiana 6M (%)": sixm_ret * 100.0,
        }
    except Exception as e:
        print(f"Błąd prognozy {ticker}: {e}")
        return None

def get_forecasts(tickers, limit=20):
    """
    Zwraca DataFrame z prognozami dla podanych tickerów.
    Dane: Yahoo Finance + prosta prognoza (forecast_utils).
    Kolumny: Ticker, Cena, Prognoza 6M, Zmiana 6M (%).
    """

    rows = []

    for ticker in tickers[:limit]:
        try:
            # pobranie ostatnich notowań
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)

            if df is None or df.empty:
                continue

            # wybór kolumny Close / Adj Close
            if "Close" in df.columns:
                col = "Close"
            elif "Adj Close" in df.columns:
                col = "Adj Close"
            else:
                continue

            s = df[col].dropna()
            if s.empty:
                continue

            last_price = s.iloc[-1].item()

            forecast_price = get_forecast(s)  # <- zakładam, że zwraca liczbę
            forecast_price = float(forecast_price) if forecast_price is not None else np.nan

            change_pct = (forecast_price / last_price - 1.0) * 100.0 if last_price > 0 and not np.isnan(forecast_price) else np.nan

            rows.append({
                "Ticker": ticker,
                "Cena": last_price,
                "Prognoza 6M": forecast_price,
                "Zmiana 6M (%)": change_pct,
            })

        except Exception as e:
            print(f"[BŁĄD] {ticker}: {e}")
            continue

    df_out = pd.DataFrame(rows)
    return df_out

    
    # 🔧 konwersja do float, żeby sortowanie działało
    df["Zmiana 6M (%)"] = pd.to_numeric(df["Zmiana 6M (%)"], errors="coerce")
    return df.sort_values("Zmiana 6M (%)", ascending=False).head(limit)

    
# np. poniżej sekcji z danymi:
st.subheader("📈 Wykres świecowy")
ticker = st.selectbox("Wybierz spółkę:", TICKERS)
fig = get_candlestick_figure(ticker)
if fig is None:
    st.info(f"Brak danych świecowych dla: {ticker} (lub niepełne OHLC).")
else:
    # wyświetl wykres plotly w streamlit
    st.plotly_chart(fig, use_container_width=True)

