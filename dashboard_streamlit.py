import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import random
from forecast_utils import get_forecast
from candlestick_chart import get_candlestick_figure
import os
import candlestick_chart as cc


# --- Ustawienia strony ---
st.set_page_config(page_title="📊 Giełda by igiblack", layout="wide")
st.title("📊 Giełda by igiblack")

# --- Nazwy spółek ---
COMPANY_NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "META": "Meta Platforms", "TSLA": "Tesla", "NVDA": "NVIDIA", "NFLX": "Netflix",
    "INTC": "Intel", "AMD": "Advanced Micro Devices", "IBM": "IBM", "ORCL": "Oracle",
    "BABA": "Alibaba", "UBER": "Uber", "PYPL": "PayPal", "SHOP": "Shopify",
    "DIS": "Disney", "V": "Visa", "MA": "Mastercard", "JPM": "JPMorgan Chase",
    "BAC": "Bank of America", "WMT": "Walmart", "T": "AT&T", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "NKE": "Nike", "PFE": "Pfizer", "MRNA": "Moderna",
    "CSCO": "Cisco", "QCOM": "Qualcomm", "ADBE": "Adobe", "CRM": "Salesforce",
    "GS": "Goldman Sachs", "MS": "Morgan Stanley", "C": "Citigroup",
    "AXP": "American Express", "BA": "Boeing", "CAT": "Caterpillar",
    "GE": "General Electric", "F": "Ford", "GM": "General Motors",
    "XOM": "ExxonMobil", "CVX": "Chevron", "BP": "BP", "SHEL": "Shell",
    "RIO": "Rio Tinto", "TSM": "TSMC", "SONY": "Sony"
}
TICKERS = list(COMPANY_NAMES.keys())


WIG20_TICKERS = [
    "PKN.WA", "PKO.WA", "PEO.WA", "LPP.WA", "KGH.WA", "CDR.WA",
    "JSW.WA", "DNP.WA", "CCC.WA", "ALR.WA", "PKP.WA", "KTY.WA",
    "MRC.WA", "CPS.WA", "TPE.WA", "LWB.WA", "ENA.WA", "EUR.WA",
    "BHW.WA", "ING.WA"
]  # tickery spółek WIG20 z Yahoo Finance (z końcówką .WA)


# --- Funkcje pomocnicze ---
def get_price_and_change(ticker: str):
    """Zwraca (ostatnia_cena, zmiana_%_6m) albo (np.nan, np.nan) przy błędzie."""
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            print(f"Błąd przy pobieraniu {ticker}: brak danych")
            return np.nan, np.nan

        # Obsługa MultiIndex (czasem yf zwraca MultiIndex, np. w trybie adj/unadj)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # wybór kolumny Close / Adj Close
        col = "Close" if "Close" in df.columns else "Adj Close" if "Adj Close" in df.columns else None
        if col is None:
            print(f"Błąd przy pobieraniu {ticker}: brak 'Close' ani 'Adj Close'")
            return np.nan, np.nan

        s = df[col].dropna()
        if s.empty:
            print(f"Błąd przy pobieraniu {ticker}: puste wartości w {col}")
            return np.nan, np.nan

        first = float(s.iloc[0])
        last = float(s.iloc[-1])
        change_pct = (last / first - 1.0) * 100.0
        return last, change_pct

    except Exception as e:
        print(f"Błąd przy pobieraniu {ticker}: {e}")
        return np.nan, np.nan



def predict_ai_price(last_price: float):
    if pd.isna(last_price):
        return np.nan, np.nan
    ai_price = round(last_price * (1.0 + random.uniform(-0.1, 0.1)), 2)
    ai_change = (ai_price / last_price - 1.0) * 100.0
    return ai_price, ai_change

# --- Wczytanie wyników AI z pliku (jeśli jest) ---
avg_df = pd.DataFrame()
try:
    avg_df = pd.read_csv("top_results/average_top.csv")
    if "pred_%".lower() in [c.lower() for c in avg_df.columns]:
        real_col = [c for c in avg_df.columns if c.lower() == "pred_%"][0]
        avg_df = avg_df.rename(columns={real_col: "pred_%"}).sort_values("pred_%", ascending=False)
    else:
        avg_df = avg_df.sort_index()

    st.subheader("Tabela średnich wyników")
    st.dataframe(avg_df, use_container_width=True)

except Exception as e:
    st.warning(f"Nie wczytano 'top_results/average_top.csv' ({e}). Sekcja AI może być niepełna.")

# --- Wykres świecowy dla wybranej spółki ---
st.subheader("📈 Wykres świecowy")

selected_ticker = st.selectbox(
    "Wybierz spółkę:",
    TICKERS,
    format_func=lambda x: COMPANY_NAMES.get(x, x)
)

fig = get_candlestick_figure(selected_ticker)
if fig:
    st.plotly_chart(fig, use_container_width=True, key=f"candle_{selected_ticker}")
else:
    st.warning(f"Brak danych świecowych dla: {selected_ticker} (lub niepełne OHLC).")


# --- Tabela rynku + AI ---
market_rows = []
for ticker in TICKERS[:48]:
    last, chg6 = get_price_and_change(ticker)
    _, ai_chg6 = predict_ai_price(last)
    market_rows.append({
        "Spółka": COMPANY_NAMES.get(ticker, ticker),
        "Ticker": ticker,
        "Cena": last,
        "Zmiana 6M (%)": chg6,
        "Prognoza AI 6M (%)": ai_chg6,
    })

market_df = pd.DataFrame(market_rows)

def _fmt_num(x, fmt):
    return fmt.format(x) if pd.notna(x) and np.isfinite(x) else "-"

display_df = market_df.copy()
display_df["Cena"] = display_df["Cena"].apply(lambda x: _fmt_num(x, "{:.2f}$"))
display_df["Zmiana 6M (%)"] = display_df["Zmiana 6M (%)"].apply(lambda x: _fmt_num(x, "{:.2f}%"))
display_df["Prognoza AI 6M (%)"] = display_df["Prognoza AI 6M (%)"].apply(lambda x: _fmt_num(x, "{:.2f}%"))

st.subheader("Rynek + AI (symulacja)")
st.dataframe(display_df, use_container_width=True)

# --- Prognozy „analityków” ---
import simple_analyst_forecasts as saf

st.subheader("🔮 Prognozy (prosty model z Yahoo)")

# domyślne tickery
tickers_for_forecast = TICKERS[:20]

# jeśli CSV wczytany, nadpisz tickery
if not avg_df.empty and "ticker" in avg_df.columns:
    tickers_for_forecast = avg_df["ticker"].dropna().astype(str).unique().tolist()[:20]

fore_df = saf.get_forecasts(tickers_for_forecast, limit=20)

if fore_df.empty:
    st.info("Brak danych do prognoz.")
else:
    fore_disp = fore_df.copy()
    fore_disp["Cena"] = fore_disp["Cena"].apply(lambda x: _fmt_num(x, "{:.2f}$"))
    fore_disp["Prognoza 6M"] = fore_disp["Prognoza 6M"].apply(lambda x: _fmt_num(x, "{:.2f}$"))
    fore_disp["Zmiana 6M (%)"] = fore_disp["Zmiana 6M (%)"].apply(lambda x: _fmt_num(x, "{:.2f}%"))
    st.dataframe(fore_disp, use_container_width=True)
    
    # --- Sekcja kryptowaluty ---
import streamlit as st
import pandas as pd
import numpy as np
import candlestick_chart_crypto as ccc

st.subheader("📊 Analiza kryptowalut")

# ✅ Ładowanie danych do tabeli z pliku CSV
try:
    crypto_df = pd.read_csv("top_results_crypto/last_top_crypto_1.csv")
except Exception as e:
    st.error(f"[BŁĄD] Nie udało się wczytać pliku last_top_crypto_1.csv: {e}")
    crypto_df = pd.DataFrame()

# ✅ Wyświetlanie tabeli z pliku
if not crypto_df.empty:
    # Formatowanie kolumny "Cena", jeśli istnieje
    if "Cena" in crypto_df.columns:
        crypto_df["Cena"] = crypto_df["Cena"].apply(
            lambda x: f"{x:.2f}$" if pd.notna(x) and isinstance(x, (int, float, np.number)) else "-"
        )

    st.dataframe(crypto_df, use_container_width=True)
else:
    st.warning("⚠️ Brak danych w tabeli kryptowalut.")

# ✅ Lista kryptowalut do wykresu
crypto_list = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
selected_crypto = st.selectbox("Wybierz kryptowalutę do wykresu:", crypto_list, key="crypto_select")

# ✅ Wykres świecowy
df_candle, fig_crypto = ccc.get_candlestick_data_crypto(selected_crypto, period="3mo", interval="1d")

if df_candle is None or df_candle.empty:
    st.warning(f"⚠️ Brak danych świecowych dla {selected_crypto}")
else:
    st.plotly_chart(fig_crypto, use_container_width=True)


import candlestick_chart as cc  # użyj tego samego co dla giełdy USA, bo to nadal akcje

st.header("📊 Prognozy AI dla WIG20")

# Wczytanie pliku z predykcjami WIG20
wig20_path = "top_results_wig20/last_top_wig20.csv"
if os.path.exists(wig20_path):
    wig20_df = pd.read_csv(wig20_path)
    st.subheader("Top prognozowane spółki WIG20")
    st.dataframe(wig20_df, use_container_width=True)
else:
    st.warning("Brak pliku z prognozami WIG20. Uruchom najpierw `train_model_wig20.py`.")

# --- Wybór spółki WIG20 w dashboardzie ---
import candlestick_chart_wig20 as cc_wig20

selected_wig20 = st.selectbox("Wybierz spółkę WIG20:", WIG20_TICKERS, key="wig20_select")

df_wig20, fig_wig20 = cc_wig20.get_candlestick_data_wig20(selected_wig20, period="6mo", interval="1d")

if fig_wig20:
    st.plotly_chart(fig_wig20, use_container_width=True)
else:
    st.warning(f"Brak danych świecowych dla {selected_wig20}")

