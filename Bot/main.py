import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

from envs.base_market_env import BaseMarketEnv
from agents.hold_buy_agent import HoldBuyAgent
from agents.random_agent import RandomAgent
from agents.buy_and_sell_agent import BuyAndSellAgent
from agents.dqn_agent import DQNAgent

# 1. Télécharger les données
df_full = yf.download("AAPL", period="6mo", interval="1d").dropna().reset_index()
os.makedirs("data", exist_ok=True)
df_full.to_csv("data/AAPL.csv", index=False)

# 2. Extraire uniquement les colonnes numériques utiles
df = df_full[["Close", "Volume"]]

# 3. Initialiser l’environnement
env_base = BaseMarketEnv(df, initial_cash=1000.0, trading_cost=0.001)

# 4. Définir les agents
agents = {
    "Hold & Buy": HoldBuyAgent,
    "Random": RandomAgent,
    "Buy & Sell Alterné": BuyAndSellAgent,
    "DQN": DQNAgent,
}

results = {}

# 5. Entraîner et exécuter chaque agent
for name, AgentClass in agents.items():
    print(f"\n▶ Agent : {name}")
    # Recréer un environnement propre pour chaque agent
    env = BaseMarketEnv(df, initial_cash=1000.0, trading_cost=0.001)

    # Créer l'agent
    if name == "DQN":
        agent = AgentClass(env, history_length=1)
        agent.train(episodes=10)
    else:
        agent = AgentClass(env)
        agent.train()

    # Exécuter l'agent
    total_reward, portfolio_values = agent.run()
    print(f"{name} - Reward: {total_reward:.2f} | Final Value: {portfolio_values[-1]:.2f}")
    results[name] = portfolio_values

# 6. Afficher les courbes
plt.figure(figsize=(12, 6))
for name, values in results.items():
    plt.plot(values, label=name)
plt.title("Évolution de la valeur du portefeuille selon l'agent")
plt.xlabel("Jour")
plt.ylabel("Valeur du portefeuille ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
