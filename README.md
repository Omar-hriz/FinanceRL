#  BaseMarketEnv 
Cet environnement est basé sur l’article *"Deep Reinforcement Learning for Trading"* (Zhang et al., 2020), et sert de socle pour entraîner un agent à trader un actif financier à l’aide du renforcement.

---

##  Fonctionnalités principales

- Compatible `gymnasium`
- Gère un portefeuille composé de :
  - 💵 cash
  - 📈 position (quantité d'actif détenue)
- Évalue dynamiquement la valeur du portefeuille et génère une récompense basée sur les profits nets

---

##  Paramètres d'initialisation

```python
BaseMarketEnv(df, initial_cash=1000.0, trading_cost=0.001)
```

- `df` : DataFrame avec au moins une colonne `'Close'`
- `initial_cash` : Cash initial (par défaut 1000)
- `trading_cost` : Frais de transaction appliqué à chaque action (hors hold)

---

##  Structure MDP

### 🔍 Observation (`observation_space`)
Vecteur contenant :
- Données du marché à l’instant `t` (`df.iloc[current_step]`)
- Quantité d’actif détenue
- Ratio `cash / portefeuille`

### 🎮 Action (`action_space`)
- `0` : Short → vendre tout
- `1` : Hold → ne rien faire
- `2` : Long → acheter tout avec le cash

### 💰 Récompense (`_compute_reward`)
```python
reward = (portfolio_value - initial_cash) / initial_cash
reward -= trading_cost * abs(action - 1)
```
Pas de frais si l’agent choisit **Hold** (`action = 1`).

---

##  Fonctionnement

1. `reset()` : réinitialise l'environnement
2. `step(action)` :
   - applique l'action
   - met à jour l'état du portefeuille
   - avance à la ligne suivante des données
   - retourne : `observation, reward, done, info`

---

##  Info retournée

Chaque step retourne un dictionnaire :

```python
{
  'portfolio': valeur totale (cash + actifs),
  'cash': liquidité disponible,
  'position': nombre d’unités d’actif détenues
}
```

---

##  Conditions requises

- `df` doit contenir une colonne `'Close'`
- Les autres colonnes (Volume, RSI, etc.) peuvent enrichir l’observation

---

##  Exemple d’utilisation

```python
import pandas as pd
from envs.base_market_env import BaseMarketEnv

df = pd.read_csv("data/AAPL.csv")
env = BaseMarketEnv(df)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print(reward, info)
```
