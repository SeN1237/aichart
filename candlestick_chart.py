import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

def get_candlestick_figure(ticker: str, period: str = "6mo", interval: str = "1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            print(f"Brak danych OHLC dla {ticker}")
            return None

        # Obsługa MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = ["Open", "High", "Low", "Close"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"Błąd pobierania danych świecowych dla {ticker}: brak kolumn {missing}")
            return None

        df = df.dropna(subset=required_cols)
        if df.empty:
            print(f"Puste dane OHLC dla {ticker}")
            return None

        fig = go.Figure(
            data=[go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name=ticker
            )]
        )
        fig.update_layout(
            title=f"Wykres świecowy: {ticker}",
            xaxis_title="Data",
            yaxis_title="Cena",
            xaxis_rangeslider_visible=False
        )
        return fig

    except Exception as e:
        print(f"Błąd pobierania danych świecowych dla {ticker}: {e}")
        return None


