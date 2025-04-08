import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt

from envs.base_market_env import BaseMarketEnv
from agents.hold_buy_agent import HoldBuyAgent
from agents.random_agent import RandomAgent
from agents.buy_and_sell_agent import BuyAndSellAgent
from agents.dqn_agent import DQNAgent

# 1. Télécharger les données
df_full = yf.download("AAPL", period="6mo", interval="1d").dropna().reset_index()
os.makedirs("data", exist_ok=True)
df_full.to_csv("data/AAPL.csv", index=False)

# 2. Extraire uniquement les colonnes nécessaires
df = df_full[["Close", "Volume"]]

# 3. Définir une fonction pour tester un agent
def test_agent(agent_class, label, train=False):
    env = BaseMarketEnv(df, initial_cash=1000.0, trading_cost=0.001)
    if agent_class.__name__ == "DQNAgent":
        agent = agent_class(env, history_length=10)
        if train:
            agent.train(episodes=20)
    else:
        agent = agent_class(env)
        if train:
            agent.train()
    total_reward, portfolio_values = agent.run()
    print(f"{label} - Reward: {total_reward:.2f} - Final Portfolio: {portfolio_values[-1]:.2f}")
    return label, portfolio_values

# 4. Tester chaque agent
results = []
results.append(test_agent(HoldBuyAgent, "Hold & Buy"))
results.append(test_agent(RandomAgent, "Random"))
results.append(test_agent(BuyAndSellAgent, "Buy & Sell"))
results.append(test_agent(DQNAgent, "DQN (Deep RL)", train=True))

# 5. Visualisation
plt.figure(figsize=(12, 6))
for label, portfolio in results:
    plt.plot(portfolio, label=label)

plt.title("Comparaison des agents de trading")
plt.xlabel("Jours")
plt.ylabel("Valeur du portefeuille ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
