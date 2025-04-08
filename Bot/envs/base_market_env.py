import numpy as np

class BaseMarketEnv:
    def __init__(self, data, initial_cash=1000.0, trading_cost=0.001, history_length=1):
        self.data = data.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.trading_cost = trading_cost
        self.history_length = history_length
        self.reset()

    def reset(self):
        self.current_step = self.history_length
        self.cash = self.initial_cash
        self.stock_owned = 0
        self.portfolio_value = [self.cash]
        return self._get_observation()

    def _get_observation(self):
        window = self.data.iloc[self.current_step - self.history_length : self.current_step]
        return window.values.flatten().astype(np.float32)

    def step(self, action):
        price = float(self.data.iloc[self.current_step]["Close"])

        if action == "buy" and self.cash >= price:
            self.stock_owned = self.cash * (1 - self.trading_cost) / price
            self.cash = 0

        elif action == "sell" and self.stock_owned > 0:
            self.cash = self.stock_owned * price * (1 - self.trading_cost)
            self.stock_owned = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        new_price = float(self.data.iloc[self.current_step]["Close"])
        portfolio_value = self.cash + self.stock_owned * new_price
        self.portfolio_value.append(portfolio_value)
        reward = portfolio_value - self.portfolio_value[-2]

        return self._get_observation(), reward, done, {}

    def get_portfolio_value(self):
        return self.portfolio_value
