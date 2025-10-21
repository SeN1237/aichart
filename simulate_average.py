import pandas as pd
import glob

# --- Znajdź wszystkie CSV z wynikami ---
csv_files = glob.glob("last_top_*.csv")  # upewnij się, że ścieżka jest poprawna
if not csv_files:
    raise FileNotFoundError("Nie znaleziono plików last_top_*.csv. Sprawdź ścieżkę i nazwy plików.")

# --- Wczytaj wszystkie CSV do jednej listy DataFrame ---
dfs = [pd.read_csv(f) for f in csv_files]

# --- Połącz wszystkie DataFrame w jeden ---
all_runs = pd.concat(dfs, axis=0)

# --- Oblicz średnią prognozę dla każdego tickera ---
average_preds = all_runs.groupby("ticker")["pred_%"].mean().reset_index()
average_preds = average_preds.sort_values("pred_%", ascending=False)

# --- Zapisz do CSV ---
average_preds.to_csv("average_top.csv", index=False)

print("Średnie prognozy zapisane do average_top.csv:")
print(average_preds.head(50))  # pokaż top 50


