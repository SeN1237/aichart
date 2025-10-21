import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

def get_candlestick_data_crypto(ticker: str, period="6mo", interval="1d"):
    """
    Pobiera dane OHLC z Yahoo Finance i tworzy wykres świecowy.
    Obsługuje multi-index kolumny (np. ('Open', 'BTC-USD')).
    """
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False  # wymuszamy OHLC
        )
    except Exception as e:
        print(f"[BŁĄD] Pobieranie danych dla {ticker}: {e}")
        return None, go.Figure()

    # Debug – sprawdzamy strukturę
    print(f"[DEBUG] Dane dla {ticker}: {df.shape} wierszy, kolumny: {df.columns.tolist()}")

    # Jeśli kolumny są MultiIndex → spłaszczamy
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]  # bierzemy tylko pierwszy poziom
        print(f"[INFO] Spłaszczono kolumny: {df.columns.tolist()}")

    # Walidacja danych
    required_cols = ["Open", "High", "Low", "Close"]
    if df.empty or not all(col in df.columns for col in required_cols):
        print(f"[BŁĄD] Brak wymaganych kolumn OHLC dla {ticker}: {df.columns.tolist()}")
        return df, go.Figure()

    # Tworzenie wykresu świecowego
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=ticker
    )])
    fig.update_layout(
        title=f"Candlestick dla {ticker}",
        xaxis_title="Data",
        yaxis_title="Cena (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )

    return df, fig


