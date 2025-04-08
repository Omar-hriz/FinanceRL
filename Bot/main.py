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
df_full = yf.download("GOOGL", period="12mo", interval="1d").dropna().reset_index()
os.makedirs("data", exist_ok=True)
df_full.to_csv("data/GOOGL.csv", index=False)

# 2. Extraire les colonnes utiles
df = df_full[["Close", "Volume"]]

# 3. Initialiser environnement de base
env_base = BaseMarketEnv(df, initial_cash=1000.0, trading_cost=0.001)

# 4. Définir les agents à tester
agents = {
    "Hold & Buy": HoldBuyAgent,
    "Random": RandomAgent,
    "Buy & Sell Alterné": BuyAndSellAgent,
    "DQN": DQNAgent,
}

results = {}
histories = {}

# 5. Entraîner et exécuter chaque agent
for name, AgentClass in agents.items():
    print(f"\n▶ Agent : {name}")
    env = BaseMarketEnv(df, initial_cash=1000.0, trading_cost=0.001)

    if name == "DQN":
        agent = AgentClass(env, history_length=10)
        agent.train(episodes=10)
    else:
        agent = AgentClass(env)
        agent.train()

    # Exécution
    if hasattr(agent, "run") and "history" in agent.run.__code__.co_varnames:
        total_reward, portfolio_values, history = agent.run()
        histories[name] = history
    else:
        total_reward, portfolio_values = agent.run()
        history = [("", "HOLD")] * len(portfolio_values)  # fallback neutre
        histories[name] = history

    print(f"{name} - Reward: {total_reward:.2f} | Final Value: {portfolio_values[-1]:.2f}")
    results[name] = portfolio_values

# 6. Courbes de performance globale
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

# 7. Fonction pour afficher les points BUY/SELL
import altair as alt
import pandas as pd

alt.data_transformers.disable_max_rows()

def plot_altair_portfolio(portfolio_values, history, dates=None):
    df = pd.DataFrame({
        "jour": range(len(portfolio_values)),
        "valeur": portfolio_values,
        "action": ["HOLD"] + [x[1].upper() for x in history]  # première valeur neutre
    })

    if dates is not None:
        df["date"] = pd.to_datetime(dates[-len(df):])
        x_axis = alt.X("date:T", title="Date")
    else:
        x_axis = alt.X("jour:Q", title="Jour")

    line = alt.Chart(df).mark_line().encode(
        x=x_axis,
        y=alt.Y("valeur:Q", title="Valeur du portefeuille"),
        tooltip=["jour", "valeur", "action"]
    )

    points = alt.Chart(df).transform_filter(
        alt.datum.action != "HOLD"
    ).mark_point(size=60).encode(
        x=x_axis,
        y="valeur:Q",
        color=alt.Color("action:N", scale=alt.Scale(domain=["BUY", "SELL"], range=["green", "red"])),
        shape=alt.Shape("action:N", scale=alt.Scale(domain=["BUY", "SELL"], range=["triangle", "triangle-down"])),
        tooltip=["jour", "action", "valeur"]
    )

    return (line + points).properties(
        title="Valeur du portefeuille et actions",
        width=1000,
        height=400
    ).interactive()

# 8. Exemple : afficher pour l’agent DQN (ou un autre)
if "DQN" in results:
    chart = plot_altair_portfolio(results["DQN"], histories["DQN"], dates=df_full["Date"])
    chart.show()


