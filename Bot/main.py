import yfinance as yf
import pandas as pd
from envs.base_market_env import BaseMarketEnv
from agents.hold_buy_agent import HoldBuyAgent
import os

# 1. Télécharger les données
df_full = yf.download("AAPL", period="6mo", interval="1d").dropna().reset_index()
df_full.to_csv("data/AAPL.csv", index=False)  # pour le dashboard

# 2. Extraire uniquement les colonnes numériques pour l'environnement
df = df_full[["Close", "Volume"]]

# 3. Initialiser l'environnement
env = BaseMarketEnv(df, initial_cash=1000.0, trading_cost=0.001)

# 4. Créer et entraîner l'agent
agent = HoldBuyAgent(env)
agent.train()
