import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BaseMarketEnv(gym.Env):
    def __init__(self, df, initial_cash=1000.0, trading_cost=0.001):
        super(BaseMarketEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.trading_cost = trading_cost
        self.current_step = 0

        self.cash = initial_cash
        self.position = 0
        self.portfolio_value = self.cash

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(df.shape[1] + 2,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0
        self.portfolio_value = self.cash
        return self._get_observation()

    def step(self, action):
        prev_price = self._get_price()

        if action == 0:
            self.cash += self.position * prev_price
            self.position = 0
        elif action == 2:
            if self.cash > 0:
                self.position += self.cash / prev_price
                self.cash = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        current_price = self._get_price()
        self.portfolio_value = self.cash + self.position * current_price

        reward = self._compute_reward(action)

        return self._get_observation(), reward, done, {
            'portfolio': self.portfolio_value,
            'cash': self.cash,
            'position': self.position
        }

    def _compute_reward(self, action):
        base_return = (self.portfolio_value - self.initial_cash) / self.initial_cash
        cost_penalty = self.trading_cost * abs(action - 1)  # 1 = hold
        return base_return - cost_penalty

    def _get_observation(self):
        market_state = self.df.iloc[self.current_step].values.astype(np.float32)
        cash_ratio = self.cash / (self.portfolio_value + 1e-8)
        obs = np.concatenate([market_state, [self.position, cash_ratio]])
        return obs

    def _get_price(self):
        return float(self.df.iloc[self.current_step]['Close'])
