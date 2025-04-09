import random
import json
import os

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.actions = ["buy", "sell", "hold"]

    def train(self):
        # Pas d'entraînement pour un agent aléatoire
        pass

    def run(self, log_file="logs_random.json"):
        state = self.env.reset()
        done = False
        portfolio = []
        rewards = []
        actions_taken = []
        history = []

        while not done:
            action = random.choice(self.actions)
            state, reward, done, _ = self.env.step(self.actions.index(action))

            actions_taken.append(action)
            rewards.append(reward)
            current_value = float(self.env.past_values[-1])
            portfolio.append(current_value)
            history.append((current_value, action))

        # État final
        final_state = {
            "portfolio": self.env.past_values[-1],
            "cash": self.env.cash,
            "position": self.env.position,
        }

        # Structure du log
        logs = {
            "agent": "random",
            "rewards": rewards,
            "portfolio_values": portfolio,
            "actions": actions_taken,
            "state": final_state,
            "history": history
        }

        #os.makedirs("logs", exist_ok=True)
        with open(os.path.join("./", log_file), "w") as f:
            json.dump(logs, f, indent=4)

        return portfolio[-1] - portfolio[0], portfolio, history
