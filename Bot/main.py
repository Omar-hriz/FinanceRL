import pandas as pd
from envs.base_market_env import BaseMarketEnv
from agents.dqn_agent import DQNAgent
import os
import torch

print("🧠 Appareil utilisé :", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 📊 Chargement du dataset
df = pd.read_excel("data/comodity egg.xlsx")
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.rename(columns={"Kota Semarang": "Close"})
df["Volume"] = 1000
df = df[["Close"]]

# 🏗️ Initialisation de l'environnement
env = BaseMarketEnv(df, initial_cash=1000.0, trading_cost=0.001, history_length=60)

# 🤖 Entraînement de l'agent
agent = DQNAgent(env, batch_size=32)
agent.train(episodes=2)

# 💾 Sauvegarde du modèle après entraînement
os.makedirs("models", exist_ok=True)
torch.save(agent.model.state_dict(), "models/dqn_model.pth")
print("📁 Modèle exporté avec succès : models/dqn_model.pth")

print("✅ Entraînement terminé.")
