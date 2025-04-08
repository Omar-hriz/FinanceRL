import json
import os
from tqdm import trange


class HoldBuyAgent:
    def __init__(self, env):
        self.env = env
        self.logs = {
            "rewards": [],
            "entropy": [],
            "log": [],
            "state": {}
        }

    def train(self):
        obs = self.env.reset()
        done = False
        total_reward = 0
        step_count = 0
        bought = False

        while not done:
            if not bought:
                action = 2  # BUY
                bought = True
            else:
                action = 1  # HOLD

            obs, reward, done, info = self.env.step(action)
            self.logs["rewards"].append(reward)
            self.logs["entropy"].append(action)
            self.logs["log"].append({"step": step_count, "action": action, "reward": reward, **info})
            total_reward += reward
            self.logs["state"] = info
            step_count += 1

        print(f"✅ Entraînement terminé | Total Reward: {total_reward:.2f} | Final Portfolio: {info['portfolio']:.2f} €")
        self.save_logs()

    def save_logs(self, filename="logs.json"):
        with open(filename, "w") as f:
            json.dump(self.logs, f)
        print(f"📁 Logs sauvegardés dans {filename}")
