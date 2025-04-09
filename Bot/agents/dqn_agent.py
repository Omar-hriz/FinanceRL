import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import json
import os
from tqdm import trange
import torch.nn.functional as F


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
    def __init__(self, env, history_length=10, batch_size=64):
        self.env = env
        self.history_length = history_length
        self.batch_size = batch_size
        self.actions = ["buy", "sell", "hold"]
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = len(self.actions)

        self.device = torch.device("cpu")
        self.model = QNetwork(self.input_dim, self.output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.gamma = 0.95
        self.epsilon = 0.5  # constante
        self.memory = []

        self.logs = {
            "rewards": [],
            "entropy": [],
            "log": [],
            "state": {},
            "losses": []
        }

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.actions), np.array([1 / 3, 1 / 3, 1 / 3])
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            probs = F.softmax(q_values, dim=1).cpu().numpy().flatten()
        return self.actions[np.argmax(probs)], probs

    def train(self, episodes=10):
        for ep in trange(episodes, desc="📈 Entraînement DQN"):
            state = self._get_state(self.env.reset())
            done = False
            step = 0
            total_reward = 0
            entropy_list = []
            episode_log = []
            episode_losses = []

            while not done:
                action, probs = self.select_action(state)
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                entropy_list.append(entropy)

                raw_state, reward, done, info = self.env.step(self.actions.index(action))
                next_state = self._get_state(raw_state)

                self.remember(state, action, reward, next_state, done)
                total_reward += reward
                episode_log.append({"step": step, "action": action, "reward": reward, **info})
                self.logs["state"] = info

                if len(self.memory) > self.batch_size:
                    loss = self._replay()
                    episode_losses.append(loss)

                state = next_state
                step += 1

            avg_loss = np.mean(episode_losses) if episode_losses else 0.0
            self.logs["rewards"].append(total_reward)
            self.logs["entropy"].append(float(np.mean(entropy_list)))
            self.logs["losses"].append(avg_loss)
            self.logs["log"].extend(episode_log)

            print(
                f"Épisode {ep + 1}/{episodes} | Total Reward: {total_reward:.2f} "
                f"| Loss: {avg_loss:.4f} | Entropy: {np.mean(entropy_list):.4f} "
                f"| Portefeuille: {info['portfolio']:.2f} €"
            )

        self._save_logs()

    def _replay(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor([self.actions.index(a) for a in actions], dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.model(states_tensor)
        next_q_values = self.model(next_states_tensor)

        target_q_values = q_values.clone().detach()
        for i in range(self.batch_size):
            target = rewards_tensor[i]
            if not dones_tensor[i]:
                target += self.gamma * torch.max(next_q_values[i])
            target_q_values[i][actions_tensor[i]] = target

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _get_state(self, raw_state):
        if isinstance(raw_state, np.ndarray):
            return raw_state.flatten()
        return np.array(raw_state).flatten()

    def _save_logs(self):
        os.makedirs("data", exist_ok=True)
        with open("logs.json", "w") as f:
            json.dump(self.logs, f)
        print("📁 Logs sauvegardés dans logs.json")
