import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
from collections import deque
import pandas as pd
import os
from tqdm import tqdm

# -----------------------------------------------------------
# Q-Network definition (same as provided in assignment)
# -----------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# -----------------------------------------------------------
# Replay Buffer
# -----------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)

# -----------------------------------------------------------
# Training function
# -----------------------------------------------------------
def train_dqn(strategy_name="strategy1_baseline"):
    # Create folders
    os.makedirs("weights", exist_ok=True)
    os.makedirs("experiments/strategy1", exist_ok=True)

    # Environment
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Hyperparameters
    num_episodes = 5000
    gamma = 0.99
    lr = 1e-3
    batch_size = 64
    replay_size = 50000
    target_update_freq = 1000
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Networks
    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=lr)

    # Replay buffer
    memory = ReplayBuffer(replay_size)

    epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay
    steps_done = 0

    # Logs
    rewards_per_episode = []

    print(f"Training {strategy_name} started on {device} ...")

    for episode in tqdm(range(num_episodes)):
        env.unwrapped.length = np.random.uniform(0.4, 1.8)
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            steps_done += 1
            # Epsilon-greedy policy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = q_network(state_tensor)
                action = q_values.argmax().item()

            # Step
            next_state, reward, done, _, __ = env.step(action)
            # Standard Gym reward = +1 for each step alive
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Update epsilon
            epsilon = max(epsilon_end, epsilon - epsilon_decay_rate)

            # Learn
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)
                states, actions, rewards, next_states, dones = (
                    states.to(device),
                    actions.to(device),
                    rewards.to(device),
                    next_states.to(device),
                    dones.to(device),
                )

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_network(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target_q_values.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if steps_done % target_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())

        rewards_per_episode.append(total_reward)

        if (episode + 1) % 100 == 0:
            avg_r = np.mean(rewards_per_episode[-100:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward (last 100): {avg_r:.2f}, ε={epsilon:.3f}")

    # Save model
    model_path = f"weights/{strategy_name}.pth"
    torch.save(q_network.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save training results
    df = pd.DataFrame({"Episode": range(1, len(rewards_per_episode)+1),
                       "Reward": rewards_per_episode})
    df.to_csv("experiments/strategy1/results.csv", index=False)
    print("Training log saved to experiments/strategy1/results.csv")

    env.close()


if __name__ == "__main__":
    train_dqn(strategy_name="strategy1_baseline")

