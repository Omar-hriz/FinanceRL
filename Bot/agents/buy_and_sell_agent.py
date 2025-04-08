class BuyAndSellAgent:
    def __init__(self, env):
        self.env = env

    def train(self):
        pass

    def run(self):
        state = self.env.reset()
        done = False
        step = 0
        portfolio = []

        while not done:
            if step % 2 == 0:
                action = "buy"
            else:
                action = "sell"

            state, reward, done, _ = self.env.step(action)
            portfolio.append(self.env.portfolio_value[-1])
            step += 1

        return portfolio[-1] - portfolio[0], portfolio
