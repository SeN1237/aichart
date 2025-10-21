import streamlit as st
import pandas as pd

# --- Tytuł ---
st.set_page_config(page_title="Symulacje AI Trading", layout="wide")
st.title("📊 Średnie wyniki z 10 symulacji AI Trading")

# --- Wczytanie CSV ---
try:
    df = pd.read_csv("average_top.csv")
except FileNotFoundError:
    st.error("❌ Plik averaged_results.csv nie został znaleziony. Uruchom najpierw simulate_average.py")
    st.stop()

# --- Sortowanie malejąco ---
df = df.sort_values("pred_%", ascending=False)

# --- Tabela ---
st.subheader("Tabela średnich wyników")
st.dataframe(df, use_container_width=True)

# --- Wykres ---
st.subheader("Wykres TOP 20")
top20 = df.head(20)
st.bar_chart(top20.set_index("ticker")["pred_%"])

