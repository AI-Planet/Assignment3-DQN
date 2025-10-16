import torch
import torch.nn as nn
import numpy as np
import gym
import os
import pandas as pd
import matplotlib.pyplot as plt


# ===============================
# Q-Network (same as training)
# ===============================
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


# ===============================
# Bar plot for results
# ===============================
def bar_plot(results):
    data = results[0]
    avgs = {k: v for k, v in data.items() if k.startswith('Avg_')}
    stds = {k.replace('Avg', 'Std'): data[k.replace('Avg', 'Std')] for k in avgs.keys()}

    sorted_keys = sorted(avgs.keys(), key=lambda x: float(x.split('_')[1]))
    avg_values = [avgs[k] for k in sorted_keys]
    std_values = [stds[k.replace('Avg', 'Std')] for k in sorted_keys]

    overall_avg = np.mean(np.array(avg_values))

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(avg_values)), avg_values, yerr=std_values, capsize=5, alpha=0.7)
    plt.xticks(range(len(avg_values)), [k.split('_')[1] for k in sorted_keys], rotation=45)
    plt.xlabel('Pole length')
    plt.ylabel('Average episode length')
    plt.title(f'Average score over all pole lengths = {round(overall_avg, 1)}')
    plt.tight_layout()
    plt.savefig("bar_plot.png")
    plt.show()


# ===============================
# Testing the agent
# ===============================
def test_pole_length(env, q_network, device):
    state = env.reset()[0]
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    done = False
    total_reward = 0
    wind = 25

    while not done:
        with torch.no_grad():
            action = q_network(state).argmax().item()

        next_state, reward, done, _, __ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)
        state = next_state
        total_reward += reward

        if total_reward >= 500 and total_reward <= 1000:
            if total_reward % wind == 0:
                env.unwrapped.force_mag = 75

        if total_reward > 1000:
            env.unwrapped.force_mag = 25 + (0.01 * total_reward)

    return total_reward


# ===============================
# Main test script
# ===============================
def test_script():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    pole_lengths = np.linspace(0.4, 1.8, 30)
    all_results = []

    #path to your model (ONLY the filename, not "weights/weights/")
    trained_nn = "strategy1_baseline.pth"

    results = {}
    total_score = 0

    for length in pole_lengths:
        print(f"Testing pole length: {round(length, 2)}")
        pole_scores = []

        for _ in range(10):
            env = gym.make("CartPole-v1")
            env.unwrapped.length = length

            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n

            # Load model
            model_path = os.path.join("weights", trained_nn)
            q_net = QNetwork(state_dim, action_dim).to(device)

            # Safe loading on CPU/GPU
            state_dict = torch.load(model_path, map_location=device)
            q_net.load_state_dict(state_dict)
            q_net.eval()

            # Run test
            score = test_pole_length(env, q_net, device)
            pole_scores.append(score)

        mean_score = np.mean(pole_scores)
        std_score = np.std(pole_scores)
        total_score += mean_score

        results[f"Avg_{round(length, 2)}"] = mean_score
        results[f"Std_{round(length, 2)}"] = std_score

    results["Total"] = total_score
    all_results.append(results)

    bar_plot(all_results)

    df = pd.DataFrame(all_results)
    df.to_excel("experiment_results.xlsx", index=False)
    print("Results saved to experiment_results.xlsx and bar_plot.png")


if __name__ == "__main__":
    test_script()
