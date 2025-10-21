import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

def get_candlestick_data_wig20(ticker: str, period: str = "6mo", interval: str = "1d"):
    """
    Pobiera dane OHLC dla wybranego tickera WIG20 i generuje wykres świecowy Plotly.

    Zwraca tuple: (DataFrame z danymi OHLC, Figure Plotly)
    """
    try:
        # Pobieranie danych
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        
        if df.empty:
            print(f"[BŁĄD] Pobieranie danych OHLC dla {ticker}: brak danych")
            return None, None
        
        # Debug
        print(f"[DEBUG] Dane dla {ticker}: {df.shape[0]} wierszy, kolumny: {list(df.columns)}")
        
        # Tworzenie wykresu świecowego
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=ticker
        )])
        fig.update_layout(
            title=f"Wykres świecowy: {ticker}",
            xaxis_title="Data",
            yaxis_title="Cena",
            xaxis_rangeslider_visible=False
        )
        return df, fig

    except Exception as e:
        print(f"[BŁĄD] get_candlestick_data_wig20({ticker}): {e}")
        return None, None

