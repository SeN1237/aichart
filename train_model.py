import pandas as pd
from xgboost import XGBRegressor
from features_prices import build_price_features
from features_news import build_news_features
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

SIMULATION_NUMBER = int(os.environ.get("SIMULATION_NUMBER", 1))


# --- KONFIGURACJA ---
TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA","NVDA","NFLX","ADBE","INTC",
   "PYPL","CRM","ORCL","CSCO","IBM","INTU","PEP","TXN","QCOM","AVGO",
    "SBUX","BABA","AMD","UBER","SHOP","TWLO","SPOT","HON","V","BKNG",
    "MDLZ","DOCU","ISRG","ADI","MU","CRWD","NVDA","ZM","ROKU","AMAT",
    "OKTA","TSM","GILD","SNOW","GOOG","PLTR","IBM","MSFT","AAPL","NFLX"
] # możesz dodać więcej
HORIZON = 21
TOP_K = 50
TCOST = 0.0015
INITIAL_CAPITAL = 10000  # $ na start

# --- Budowanie cech ---
print("Budowanie cech cenowych...")
price_feats = build_price_features(TICKERS)

print("Budowanie cech z newsów...")
news_feats = build_news_features(TICKERS, days=7)

# Upewniamy się, że obie ramki mają kolumnę 'date'
if 'date' not in price_feats.columns:
    price_feats = price_feats.rename_axis('date').reset_index()
if 'date' not in news_feats.columns:
    news_feats = news_feats.rename_axis('date').reset_index()

# Konwersja dat na datetime
price_feats['date'] = pd.to_datetime(price_feats['date'])
news_feats['date'] = pd.to_datetime(news_feats['date'])

# Debug
print("Price feats cols:", price_feats.dtypes)
print("News feats cols:", news_feats.dtypes)

# Łączenie cen i newsów
features = (
    price_feats
    .merge(news_feats, on=['ticker', 'date'], how='left')
    .set_index('date')
    .sort_index()
)

# Tworzymy target
features['target'] = features.groupby('ticker')['ret_21'].shift(-HORIZON)


features['sentiment'] = features['sentiment'].fillna(0)

# --- Trening modelu ---
# Usuwamy wiersze z brakującym targetem
features = features.dropna(subset=['target'])

Xcols = [c for c in features.columns if c not in ['target','ticker']]
X = features[Xcols]
y = features['target']

model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
print("Trening modelu AI...")
model.fit(X, y)

# --- Prognozy ---
preds = pd.Series(model.predict(X), index=X.index)
snapshot = features[['ticker']].copy()
snapshot['pred'] = preds.values
last_snap = snapshot.loc[snapshot.index.max()]
top = last_snap.sort_values('pred', ascending=False).head(TOP_K)

# --- Dodaj prognozę % ---
top['pred_%'] = top['pred']*100

# --- Symulacja equity ---
equity = [INITIAL_CAPITAL]
last_hold = pd.Series(dtype=float)

for date in sorted(features.index.unique()):
    frame = features.loc[date]
    ret_map = frame.set_index('ticker')['ret_1']
    
    # top 10 na dany dzień
    daily_top = frame.copy()
    daily_top['pred'] = model.predict(frame[Xcols])
    daily_top = daily_top.sort_values('pred', ascending=False).head(TOP_K)
    daily_top['weight'] = 1.0 / len(daily_top)
    
    daily_top = daily_top[~daily_top.index.duplicated(keep='first')]
    
    tickers_union = ret_map.index.union(daily_top.index).unique()
    ret_map = ret_map.reindex(tickers_union).fillna(0)
    weights = daily_top['weight'].reindex(tickers_union).fillna(0)
    
    
    # koszt transakcji przy rebalansie
    turnover = (last_hold.reindex(tickers_union).fillna(0) - weights).abs().sum()
    r = (ret_map * weights).sum() - TCOST * turnover
    
    equity.append(equity[-1] * (1 + r))
    last_hold = weights.copy()

equity_series = pd.Series(equity[1:], index=sorted(features.index.unique()))



# --- Wyświetlenie wyników ---
print("Top akcje do kupienia (ostatni dzień):")
print(top[['ticker','pred_%']])

# --- Wyświetlenie wyników i zapis każdej symulacji ---
import os

SIMULATION_NUMBER = 1  # <- ta zmienna będzie zmieniana w pętli, jeśli robisz wielokrotne symulacje

# Tworzymy folder, jeśli nie istnieje
if not os.path.exists("top_results"):
    os.makedirs("top_results")

top_file = f"top_results/last_top_{SIMULATION_NUMBER}.csv"
top.to_csv(top_file, index=False)
print(f"Zapisano plik: {top_file}")

print("Top akcje do kupienia (ostatni dzień):")
print(top[['ticker','pred_%']])



total_return = (equity_series[-1]/INITIAL_CAPITAL - 1)*100
print(f"Przykładowy zysk: ${equity_series[-1]:.2f} ({total_return:.2f}%) od {equity_series.index[0].date()} do {equity_series.index[-1].date()}")

# --- Wykres equity ---
equity_series.plot(title="Symulacja portfela (top 10 akcji)")
plt.xlabel("Data")
plt.ylabel("Kapitał ($)")
plt.show()

import os

# --- ZAPIS TOP K DO PLIKU (dla wielu symulacji) ---
sim_number = os.getenv("SIMULATION_NUMBER", "1")  # numer symulacji z env
os.makedirs("top_results", exist_ok=True)
top.to_csv(f"top_results/last_top_{sim_number}.csv", index=False)
print(f"Wyniki zapisane do: top_results/last_top_{sim_number}.csv")


