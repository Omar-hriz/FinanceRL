import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ✅ Désactive CUDA si problème

import torch
import pandas as pd
import numpy as np
from envs.base_market_env import BaseMarketEnv
from agents.dqn_agent import QNetwork
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# ⚙️ Paramètres
history_length = 60
initial_cash = 1000.0
trading_cost = 0.001
model_path = "models/dqn_model.pth"
result_path = "data/results_test.json"
device = torch.device("cpu")

# 📊 Chargement des données
df = pd.read_excel("data/comodity egg.xlsx")
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.rename(columns={"Kota Semarang": "Close"})
df["Volume"] = 1000
df = df[["Close", "Tanggal", "Volume"]]

# 🧪 Filtrage sur 2022
df_test = df[(df["Tanggal"].dt.year == 2022)].reset_index(drop=True)

# 🌍 Environnement
env = BaseMarketEnv(df_test, initial_cash=initial_cash, trading_cost=trading_cost, history_length=history_length)

# 🧠 Modèle
input_dim = 68
output_dim = 3
model = QNetwork(input_dim=input_dim, output_dim=output_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 🔁 Simulation
state_history = []
state = env.reset()
for _ in range(history_length):
    state_history.append(state)
state_seq = np.array(state_history)

done = False
portfolio_values = []
actions_log = []

while not done:
    state_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()

    next_state, reward, done, info = env.step(action)
    state_seq = np.concatenate([state_seq[1:], [next_state]])
    portfolio_values.append(info["portfolio"])
    actions_log.append({
        "step": len(portfolio_values) - 1,
        "portfolio": info["portfolio"],
        "cash": info["cash"],
        "position": info["position"],
        "action": action
    })

# 📈 Calcul des métriques
final_value = portfolio_values[-1]
total_return = final_value - initial_cash
roi = (final_value / initial_cash - 1) * 100
returns = np.diff(np.array(portfolio_values))
volatility = float(np.std(returns))
sharpe_ratio = float(np.mean(returns) / (volatility + 1e-8))

# ✅ Résumé
results = {
    "final_portfolio": round(final_value, 2),
    "total_return": round(total_return, 2),
    "roi": round(roi, 2),
    "volatility": round(volatility, 4),
    "sharpe_ratio": round(sharpe_ratio, 4),
    "portfolio_evolution": portfolio_values,
    "log": actions_log
}

# 💾 Sauvegarde en JSON
os.makedirs("data", exist_ok=True)
with open(result_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"📁 Résultats exportés dans {result_path}")

# 🖼️ Affichage
print("✅ Test terminé")
print(f"📈 Portefeuille final : {final_value:.2f} €")
print(f"💸 Gain / Perte : {total_return:.2f} €")
print(f"📊 ROI : {roi:.2f} %")
print(f"📉 Volatilité : {volatility:.4f}")
print(f"⚖️ Sharpe Ratio : {sharpe_ratio:.4f}")

plt.plot(portfolio_values)
plt.title("📊 Évolution du portefeuille - Test")
plt.xlabel("Step")
plt.ylabel("Valeur (€)")
plt.grid(True)
plt.tight_layout()
plt.show()
