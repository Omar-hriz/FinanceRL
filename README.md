#  BaseMarketEnv 
Cet environnement est basé sur l’article *"Deep Reinforcement Learning for Trading"* (Zhang et al., 2020), et sert de socle pour entraîner un agent à trader un actif financier à l’aide du renforcement.

---

##  Fonctionnalités principales

- Compatible avec l’interface `gymnasium`
- Gestion complète d’un portefeuille comprenant :
  - du cash
  - une position (quantité d’actif détenue)
- Calcul de la valeur totale du portefeuille à chaque étape
- Récompense basée sur le rendement ajusté par la volatilité
- Coût de transaction appliqué lors d’un changement de position
- Observation enrichie par des indicateurs financiers et des prix normalisés
- Comparaison avec un agent random

---

##  Paramètres d'initialisation

```python
BaseMarketEnv(df, initial_cash=1000.0, trading_cost=0.001)
```

- `df` : DataFrame avec au moins une colonne `'Close'`
- `initial_cash` : Cash initial (par défaut 1000)
- `trading_cost` : Frais de transaction appliqué à chaque action (hors hold)
- `history_length` : Longueur de l’historique utilisé pour l’observation  

---

##  Structure MDP

### Observation (`observation_space`)
Vecteur de taille history_length + 8 comprenant :

- Les history_length derniers prix normalisés par la moyenne mobile
- Les indicateurs suivants à l’instant t :
- Rendements sur 21, 42, 63, 252 jours
- MACD
- RSI
- Position détenue
- Ratio `cash / portefeuille`

### Action (`action_space`)
- `0` : Short → vendre tout
- `1` : Hold → ne rien faire
- `2` : Long → acheter tout avec le cash

### Récompense (`_compute_reward`)
La récompense est calculée comme le rendement actualisé et normalisé par la volatilité, avec pénalité liée au coût de transaction :  
```python
reward = position * scaled_return - (trading_cost * abs(action - 1))
```
La récompense est bornée entre -10 et +10.

---

##  Fonctionnement

1. `reset()` : réinitialise l'environnement
2. `step(action)` :
   - applique l'action
   - met à jour la position et le cash
   - avance à l'instant suivante
   - retourne : `observation, reward, done, info`

---

##  Info retournée

Chaque step retourne un dictionnaire :

```python
{
  'portfolio': valeur totale (cash + positions),
  'cash': liquidité disponible,
  'position': nombre d’unités d’actif détenues
}
```

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

## Entraînement avec un agent DQN

L’agent DQN implémente un réseau LSTM pour exploiter les dépendances temporelles dans l’état. Le fichier main.py montre un exemple complet :  
```python
from agents.dqn_agent import DQNAgent

agent = DQNAgent(env, batch_size=32)
agent.train(episodes=300)
```

Après l’entraînement, le modèle est automatiquement sauvegardé dans `models/dqn_model.pth` et les logs dans `logs.json`. Afin de pouvoir les afficher avec streamlit

## Visualisation des performances
Exécuter la commande suivante, pour visualiser les résultats sur une interfaces graphiques  : 
```bash
streamlit run metrics_dashboard
```

