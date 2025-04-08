import pandas as pd
from envs.base_market_env import BaseMarketEnv

# Exemple avec des données fictives
df = pd.DataFrame({
    'Close': [100, 102, 101, 105, 108],
    'Volume': [1000, 1100, 1050, 1200, 1150]
})

env = BaseMarketEnv(df)
obs = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # test aléatoire
    obs, reward, done, info = env.step(action)
    print(f"Step: {env.current_step}, Action: {action}, Reward: {reward:.4f}, Portfolio: {info['portfolio']:.2f}")
