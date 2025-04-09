import yfinance as yf
import pandas as pd
from envs.base_market_env import BaseMarketEnv
from agents.dqn_agent import DQNAgent
import os
import torch

print(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# 1. Télécharger les données
print("📥 Téléchargement des données...")
# df_full = yf.download("AAPL", period="36mo", interval="1d").dropna().reset_index()
# os.makedirs("data", exist_ok=True)
# df_full.to_csv("data/AAPL.csv", index=False)  # pour le dashboard
#
# # 2. Extraire uniquement les colonnes numériques pour l'environnement
# df = df_full[["Close", "Volume"]]
# 2. Charger et préparer les données de commodité
df = pd.read_excel("data/comodity egg.xlsx")
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.rename(columns={"Kota Semarang": "Close"})
df["Volume"] = 1000

# 3. Initialiser l’environnement avec uniquement les colonnes numériques
env = BaseMarketEnv(df[["Close", "Volume"]], initial_cash=1000.0, trading_cost=0.001, history_length=5)

# 4. Créer et entraîner l'agent DQN
agent = DQNAgent(env, batch_size=32)
agent.train(episodes=100)

print("✅ Entraînement terminé.")
