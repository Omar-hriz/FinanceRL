import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, env, history_length=10):
        self.env = env
        self.history_length = history_length
        self.actions = ["buy", "sell", "hold"]
        self.input_dim = len(self.env.data.columns)
        self.output_dim = len(self.actions)

        self.model = QNetwork(self.input_dim, self.output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = []

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 1000:
            self.memory.pop(0)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return self.actions[torch.argmax(q_values).item()]

    def train(self, episodes=10):
        for _ in range(episodes):
            state = self._get_state(self.env.reset())
            done = False
            while not done:
                action = self.select_action(state)
                raw_state, reward, done, _ = self.env.step(action)
                next_state = self._get_state(raw_state)

                self.remember(state, action, reward, next_state, done)

                if len(self.memory) > 32:
                    self._replay(32)

                state = next_state

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def _replay(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            target = reward
            if not done:
                with torch.no_grad():
                    target += self.gamma * torch.max(self.model(next_state_tensor.unsqueeze(0)))

            output = self.model(state_tensor.unsqueeze(0))[0]
            target_f = output.clone().detach()
            target_f[self.actions.index(action)] = target

            loss = self.criterion(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _get_state(self, raw_state):
        if isinstance(raw_state, np.ndarray):
            return raw_state.flatten()
        return np.array(raw_state).flatten()

    def run(self):
        state = self._get_state(self.env.reset())
        done = False
        portfolio = []

        while not done:
            action = self.select_action(state)
            raw_state, reward, done, _ = self.env.step(action)
            state = self._get_state(raw_state)
            portfolio.append(self.env.portfolio_value[-1])

        return portfolio[-1] - portfolio[0], portfolio
