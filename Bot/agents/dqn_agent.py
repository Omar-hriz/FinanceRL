import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import json
import os
from tqdm import trange
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, lstm_layers=1):
        super(QNetwork, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.relu(out[:, -1, :])
        return self.fc(out)


class DQNAgent:
    def __init__(self, env, history_length=60, batch_size=64):
        self.env = env
        self.history_length = history_length
        self.batch_size = batch_size
        self.actions = ["buy", "sell", "hold"]
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = len(self.actions)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = QNetwork(self.input_dim, self.output_dim).to(self.device)
        self.target_model = QNetwork(self.input_dim, self.output_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.SmoothL1Loss()  # Huber loss

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.memory = []
        self.target_update_freq = 10

        self.logs = {"rewards": [], "log": [], "state": {}, "losses": []}
        self.state_history = deque(maxlen=history_length)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def select_action(self, state_seq):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions)
        state_tensor = (
            torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return self.actions[torch.argmax(q_values).item()]

    def train(self, episodes=10):
        for ep in trange(episodes, desc="📈 Entraînement DQN"):
            self.state_history.clear()
            state = self._get_state(self.env.reset())
            done = False
            step = 0
            total_reward = 0
            episode_log = []
            episode_losses = []

            while not done:
                action = self.select_action(state)
                raw_state, reward, done, info = self.env.step(
                    self.actions.index(action)
                )
                next_state = self._get_state(raw_state)

                self.remember(state, action, reward, next_state, done)
                total_reward += reward
                episode_log.append(
                    {"step": step, "action": action, "reward": reward, **info}
                )
                self.logs["state"] = info

                if len(self.memory) > self.batch_size:
                    loss = self._replay()
                    episode_losses.append(loss)

                state = next_state
                step += 1

            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            self.logs["rewards"].append(total_reward)
            self.logs["losses"].append(avg_loss)
            self.logs["log"].extend(episode_log)

            print(
                f"Épisode {ep + 1}/{episodes} | Total Reward: {total_reward:.2f} "
                f"| Loss: {avg_loss:.4f} | Portefeuille: {info['portfolio']:.2f} €"
            )

            if ep % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._save_logs()

    def _replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(
            self.device
        )
        next_states_tensor = torch.tensor(
            np.array(next_states), dtype=torch.float32
        ).to(self.device)
        actions_tensor = torch.tensor(
            [self.actions.index(a) for a in actions], dtype=torch.long
        ).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Optionnel : standardisation des récompenses
        # rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        q_values = self.model(states_tensor)
        next_q_values = self.target_model(next_states_tensor)

        q_action = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        max_next_q = torch.max(next_q_values, dim=1)[0]
        target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q

        # Clipping des cibles
        target_q = torch.clamp(target_q, -10.0, 10.0)

        loss = self.criterion(q_action, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _get_state(self, raw_state):
        self.state_history.append(raw_state)
        if len(self.state_history) < self.history_length:
            while len(self.state_history) < self.history_length:
                self.state_history.appendleft(self.state_history[0])
        return np.array(self.state_history)

    def _save_logs(self):
        os.makedirs("data", exist_ok=True)
        with open("logs.json", "w") as f:
            json.dump(self.logs, f)
        print("📁 Logs sauvegardés dans logs.json")
