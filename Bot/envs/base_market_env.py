import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class BaseMarketEnv(gym.Env):
    def __init__(self, df, initial_cash=1000.0, trading_cost=0.001, history_length=60):
        super(BaseMarketEnv, self).__init__()
        self.initial_cash = initial_cash
        self.trading_cost = trading_cost
        self.history_length = history_length

        self.raw_df = df.reset_index(drop=True)
        self.df = self._compute_features(self.raw_df)

        self.current_step = 0
        self.cash = initial_cash
        self.position = 0
        self.portfolio_value = self.cash
        self.past_values = []

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.history_length + 8,), dtype=np.float32
        )

    def _compute_features(self, df):
        df = df.copy()
        df["return_21d"] = df["Close"].pct_change(21)
        df["return_42d"] = df["Close"].pct_change(42)
        df["return_63d"] = df["Close"].pct_change(63)
        df["return_252d"] = df["Close"].pct_change(252)

        # MACD
        ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        df["macd"] = macd - signal

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=30).mean()
        avg_loss = loss.rolling(window=30).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Prix normalisé par la moyenne mobile (comme dans le papier)
        df["price_norm"] = (
            df["Close"] / df["Close"].rolling(window=self.history_length).mean() - 1
        )

        df = df.dropna().reset_index(drop=True)
        return df

    def reset(self, seed=None, options=None):
        self.current_step = self.history_length
        self.cash = self.initial_cash
        self.position = 0
        self.portfolio_value = self.cash
        self.past_values = []
        return self._get_observation()

    def step(self, action):
        prev_price = self._get_price()

        if action == 0:  # sell
            self.cash += self.position * prev_price
            self.position = 0
        elif action == 2:  # buy
            if self.cash > 0:
                self.position += self.cash / prev_price
                self.cash = 0

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        current_price = self._get_price()
        self.portfolio_value = self.cash + self.position * current_price
        reward = self._compute_reward(action)
        self.past_values.append(self.portfolio_value)

        return (
            self._get_observation(),
            reward,
            done,
            {
                "portfolio": self.portfolio_value,
                "cash": self.cash,
                "position": self.position,
            },
        )

    def _compute_reward(self, action):
        if len(self.past_values) < 2:
            return 0.0

        pt = self._get_price()
        pt_prev = float(self.df.iloc[self.current_step - 1]["Close"])
        rt = pt - pt_prev

        past_returns = (
            self.df["Close"].pct_change().ewm(span=60).std().iloc[self.current_step]
        )
        scaled_return = rt / (past_returns + 1e-8)

        reward = self.position * scaled_return
        trading_cost = self.trading_cost * abs(action - 1) * pt
        reward -= trading_cost / (self.portfolio_value + 1e-8)

        reward = max(min(reward, 10), -10)
        return reward

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        past_prices = self.df.iloc[
            self.current_step - self.history_length : self.current_step
        ]["price_norm"].values
        obs = np.concatenate(
            [
                past_prices.astype(np.float32),
                np.array(
                    [
                        row["return_21d"],
                        row["return_42d"],
                        row["return_63d"],
                        row["return_252d"],
                        row["macd"],
                        row["rsi"],
                        self.position,
                        self.cash / (self.portfolio_value + 1e-8),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        return obs

    def _get_price(self):
        return float(self.df.iloc[self.current_step]["Close"])
