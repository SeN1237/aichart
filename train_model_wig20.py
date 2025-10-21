import pandas as pd
from xgboost import XGBRegressor
from features_prices import build_price_features
from features_news import build_news_features
from datetime import datetime
import matplotlib.pyplot as plt
import os

# --- KONFIGURACJA ---
TICKERS_WIG20 = [
    "PKN.WA", "PKO.WA", "PEO.WA", "LPP.WA", "KGH.WA", "CDR.WA",
    "JSW.WA", "DNP.WA", "CCC.WA", "ALR.WA", "PKP.WA", "KTY.WA",
    "MRC.WA", "CPS.WA", "TPE.WA", "LWB.WA", "ENA.WA", "EUR.WA",
    "BHW.WA", "ING.WA"
]  # tickery spółek WIG20 z Yahoo Finance (z końcówką .WA)

HORIZON = 21
TOP_K = 10
TCOST = 0.0015
INITIAL_CAPITAL = 10000

print(">>> Budowanie cech cenowych (WIG20)...")
price_feats = build_price_features(TICKERS_WIG20)

print(">>> Budowanie cech z newsów (WIG20)...")
news_feats = build_news_features(TICKERS_WIG20, days=7)

# Resetujemy indeksy
if 'date' not in price_feats.columns:
    price_feats = price_feats.rename_axis('date').reset_index()
if 'date' not in news_feats.columns:
    news_feats = news_feats.rename_axis('date').reset_index()

price_feats['date'] = pd.to_datetime(price_feats['date'])
news_feats['date'] = pd.to_datetime(news_feats['date'])

# Łączenie
features = (
    price_feats
    .merge(news_feats, on=['ticker', 'date'], how='left')
    .set_index('date')
    .sort_index()
)

# Target = ret_21 przesunięte o HORIZON
features['target'] = features.groupby('ticker')['ret_21'].shift(-HORIZON)
features['sentiment'] = features['sentiment'].fillna(0)
features = features.dropna(subset=['target'])

# Dane treningowe
Xcols = [c for c in features.columns if c not in ['target', 'ticker']]
X = features[Xcols]
y = features['target']

print(">>> Trening modelu AI (WIG20)...")
model = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
model.fit(X, y)

# Prognozy
preds = pd.Series(model.predict(X), index=X.index)
snapshot = features[['ticker']].copy()
snapshot['pred'] = preds.values
last_snap = snapshot.loc[snapshot.index.max()]
top = last_snap.sort_values('pred', ascending=False).head(TOP_K)
top['pred_%'] = top['pred'] * 100

# Symulacja equity
equity = [INITIAL_CAPITAL]
last_hold = pd.Series(dtype=float)

for date in sorted(features.index.unique()):
    frame = features.loc[date]
    ret_map = frame.set_index('ticker')['ret_1']

    daily_top = frame.copy()
    daily_top['pred'] = model.predict(frame[Xcols])
    daily_top = daily_top.sort_values('pred', ascending=False).head(TOP_K)
    daily_top['weight'] = 1.0 / len(daily_top)

    tickers_union = pd.Index(ret_map.index.tolist() + daily_top.index.tolist()).unique()

# upewniamy się, że nie ma duplikatów
    ret_map = ret_map.groupby(ret_map.index).first().reindex(tickers_union).fillna(0)
    weights = daily_top['weight'].groupby(daily_top.index).first().reindex(tickers_union).fillna(0)


    turnover = (last_hold.reindex(tickers_union).fillna(0) - weights).abs().sum()
    r = (ret_map * weights).sum() - TCOST * turnover

    equity.append(equity[-1] * (1 + r))
    last_hold = weights.copy()

equity_series = pd.Series(equity[1:], index=sorted(features.index.unique()))

# --- ZAPIS ---
os.makedirs("top_results_wig20", exist_ok=True)
top_file = "top_results_wig20/last_top_wig20.csv"
top.to_csv(top_file, index=False)
print(f"[OK] Zapisano plik: {top_file}")

# --- Podsumowanie ---
print("Top spółki WIG20 do kupienia (ostatni dzień):")
print(top[['ticker', 'pred_%']])

total_return = (equity_series[-1]/INITIAL_CAPITAL - 1)*100
print(f"Przykładowy zysk: {total_return:.2f}% od {equity_series.index[0].date()} do {equity_series.index[-1].date()}")

equity_series.plot(title="Symulacja portfela (WIG20)")
plt.xlabel("Data")
plt.ylabel("Kapitał (PLN)")
plt.show()

