from __future__ import annotations
import argparse
import collections
import csv
import math
import os
import random
import time
from typing import Tuple, Deque, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# -----------------------------------------------------------
# env factory
# -----------------------------------------------------------
def make_cartpole_env(seed: int = 0):
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    wrapper = "gymnasium"
    return env, wrapper

# -----------------------------------------------------------
# Strategy 3: reward shaping
# -----------------------------------------------------------
def get_env_scales(env) -> Tuple[float, float]:
    """Return (x_max, theta_max) from env thresholds with safe fallbacks"""
    x_max_default = 2.4
    theta_max_default = 12 * math.pi / 180 # 12 degrees in radians
    unwrapped = getattr(env, "unwrapped", env)
    x_max = getattr(unwrapped, "x_threshold", x_max_default)
    theta_max = getattr(unwrapped, "theta_threshold_radians", theta_max_default)
    return float(x_max), float(theta_max)

def reward_strategy3(
        observation,
        r_env: float,
        x_max: float,
        theta_max: float,
        *,
        a: float = 0.7, # angle weight
        b: float = 0.2, #angular velocity weight
        c: float = 0.1, # cart position weight
        clip: float | None = 1.25
) -> float:
    """smooth reward shaping on theta, theta_dot and x (quadratic, scale_aware)"""
    x, x_dot, theta, theta_dot = observation

    theta_term = (theta / theta_max) ** 2
    theta_dot_scale = 4.0 * theta_max
    theta_dot_term = (theta_dot / theta_dot_scale) ** 2
    x_term = (x / x_max) ** 2

    penalty = a * theta_term + b * theta_dot_term + c * x_term
    if clip is not None:
        penalty = min(float(penalty), float(clip))
    return float(r_env - penalty)

# -----------------------------------------------------------
# DQN components
# -----------------------------------------------------------

class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: Deque = collections.deque(maxlen=capacity)

    def push(self, *transition) -> None:
        self.memory.append(tuple(transition))

    def sample(self, batch_size: int) -> List:
        batch = random.sample(self.memory, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self) -> int:
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def select_action(policy_net, state, n_actions, steps_done, eps_start, eps_end, eps_decay, device):
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
    if random.random() < eps_threshold:
        return random.randrange(n_actions), eps_threshold
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q = policy_net(s)
        a = int(q.argmax(dim=1).item())
        return a, eps_threshold

def optimize_model(policy_net, target_net, memory: ReplayMemory, optimizer, batch_size: int, gamma:int, device):
    if len(memory) < batch_size:
        return 0.0
    states, actions, rewards, next_states, dones = memory.sample(batch_size)

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = policy_net(states_t).gather(1, actions_t)

    with torch.no_grad():
        max_next_q_values = target_net(next_states_t).max(1, keepdim= True)[0]
        target_q_values = rewards_t + gamma * max_next_q_values * (1 - dones_t)

    loss = nn.functional.smooth_l1_loss(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return float(loss.item())

# -----------------------------------------------------------
# Training
# -----------------------------------------------------------

def train_strategy3(
        episodes: int = 300,
        seed: int = 0,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        memory_capacity: int = 50_000,
        target_update_every: int = 10,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay: int = 10_000,
        save_dir: str = "weights",
        results_dir: str = "experiments",
        device_str: str | None = None,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device= torch.device("cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")

    env, wrapper = make_cartpole_env(seed = seed)
    x_max, theta_max = get_env_scales(env)

    if wrapper == "gymnasium":
        state, _ = env.reset(seed = seed)
    else:
        state = env.reset()
    state_dim = len(state)
    n_actions = env.action_space.n

    policy_net = QNetwork(state_dim,n_actions).to(device)
    target_net = QNetwork(state_dim,n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(memory_capacity)

    os.makedirs(save_dir, exist_ok = True)
    exp_dir = os.path.join(results_dir, "../experiments/strategy_3")
    os.makedirs(exp_dir, exist_ok = True)
    results_csv = os.path.join(exp_dir, "results.csv")

    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "return_env", "mean_loss", "epsilon"])

    global_steps = 0
    for episode in range(1, episodes + 1):
        if wrapper == "gymnasium":
            state, _ = env.reset(seed = seed + episode)
        else:
            state = env.reset()
        done = False
        episode_return_env = 0.0
        losses: List[float] = []

        while not done:
            action, epsilon = select_action(
                policy_net, state, n_actions, global_steps, eps_start, eps_end, eps_decay, device
            )
            if wrapper == "gymnasium":
                next_state, r_env, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
            else:
                next_state, r_env, done, _ = env.step(action)

            r_train = reward_strategy3(next_state, float(r_env), x_max, theta_max)

            memory.push(state, action, r_train, next_state, float(done))

            loss_val = optimize_model(
                policy_net, target_net, memory, optimizer, batch_size, gamma, device
            )

            if loss_val:
                losses.append(loss_val)

            state = next_state
            episode_return_env += float(r_env)
            global_steps += 1

        if episode % target_update_every == 0:
            target_net.load_state_dict(policy_net.state_dict())

        mean_loss = float(np.mean(losses)) if losses else 0.0

        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow([episode, episode_return_env, mean_loss, epsilon])


        print(
            f"[Strategy3] Ep {episode:4d}/{episodes} | Return(env): {episode_return_env:6.1f} | "
            f"Loss: {mean_loss:7.4f} | eps: {epsilon:5.3f}"
        )

        save_path = os.path.join(save_dir, "../weights/strategy_3.pth")
        torch.save(policy_net.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        print(f"Training log saved to {results_csv}")
    env.close()

def parse_args():
    p = argparse.ArgumentParser(description="DQN with Strategy 3 reward shaping (single file)")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--memory_capacity", type=int, default=50000)
    p.add_argument("--target_update", type=int, default=10)
    p.add_argument("--eps_start", type=float, default=1.0)
    p.add_argument("--eps_end", type=float, default=0.05)
    p.add_argument("--eps_decay", type=int, default=10000)
    p.add_argument("--save_dir", type=str, default="weights")
    p.add_argument("--results_dir", type=str, default="experiments")
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_strategy3(
        episodes=args.episodes,
        seed=args.seed,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        memory_capacity=args.memory_capacity,
        target_update_every=args.target_update,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay=args.eps_decay,
        save_dir=args.save_dir,
        results_dir=args.results_dir,
        device_str=args.device,
    )
