import os
import pandas as pd
import subprocess
import time

NUM_SIMULATIONS = 10
dfs = []

for sim in range(1, NUM_SIMULATIONS + 1):
    print(f"=== Symulacja {sim}/{NUM_SIMULATIONS} ===")
    # Uruchamiamy train_model.py i przekazujemy numer symulacji
    subprocess.run(["python", "train_model.py"], check=True, env={**os.environ, "SIMULATION_NUMBER": str(sim)})
    
    # Wczytujemy zapisany plik CSV
    top_file = f"top_results/last_top_{sim}.csv"
    df = pd.read_csv(top_file)
    dfs.append(df)
    
    time.sleep(60)  # pauza 1 minuta między symulacjami

# Łączymy wszystkie wyniki
all_runs = pd.concat(dfs)
avg_df = all_runs.groupby("ticker")["pred_%"].mean().reset_index()
avg_df = avg_df.sort_values("pred_%", ascending=False)

# Zapisujemy do pliku, aby dashboard mógł go wczytać
avg_df.to_csv("top_results/average_top.csv", index=False)
print("Zapisano plik średnich: top_results/average_top.csv")

