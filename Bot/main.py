import yfinance as yf
import pandas as pd
from envs.base_market_env import BaseMarketEnv
from agents.dqn_agent import DQNAgent
import os
import torch

print(
    "🧠 Appareil utilisé :",
    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

# Charger les données locales (commodité)
print("📊 Chargement du dataset...")
df = pd.read_excel("data/comodity egg.xlsx")
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.rename(columns={"Kota Semarang": "Close"})
df["Volume"] = 1000

# Réduire aux colonnes utiles
df = df[["Close"]]

# Initialiser l’environnement avec 60 jours d’historique
env = BaseMarketEnv(df, initial_cash=1000.0, trading_cost=0.001, history_length=60)

# Initialiser et entraîner l'agent DQN
agent = DQNAgent(env, batch_size=32)
agent.train(episodes=200)

print("✅ Entraînement terminé.")
