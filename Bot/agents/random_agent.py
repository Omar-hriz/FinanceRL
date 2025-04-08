import random

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.actions = ["buy", "sell", "hold"]

    def train(self):
        # Pas d'entraînement pour un agent aléatoire
        pass

    def run(self):
        state = self.env.reset()
        done = False
        portfolio = []

        while not done:
            action = random.choice(self.actions)
            state, reward, done, _ = self.env.step(action)
            portfolio.append(self.env.portfolio_value[-1])

        return portfolio[-1] - portfolio[0], portfolio
