#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read data from file
pd.options.mode.copy_on_write = True
data = pd.read_json("CDs_and_Vinyl.jsonl", lines=True)

# Create reference and evaluation dataframes
reference_data = data[(data["timestamp"] > "2015-01-01") & (data["timestamp"] < "2019-06-01")].set_index("timestamp").sort_index()
evaluation_data = data[data["timestamp"] >= "2019-06-01"].set_index("timestamp").sort_index()

reference_array = reference_data["rating"].values
rolling = reference_data["rating"].rolling("30D").mean()
significance_threshold = rolling.min()
#%%
# Sliding window function
def get_sliding_window_with_newvals(x):
    windows, times, new_values, drift_flag = [], [], [], []
    start_time = x.index[0]
    day = np.timedelta64(1, 'D')
    month = np.timedelta64(30, 'D')
    while start_time + month <= x.index[-1]:
        end_time = start_time + month
        window = x[start_time:end_time]
        new_window_values = x[start_time:start_time + day]["rating"].values
        windows.append(window["rating"].values)
        new_values.append(new_window_values)
        times.append(end_time)
        drift_flag.append(np.mean(window["rating"]) < significance_threshold)
        start_time += day
    return times, windows, new_values, drift_flag

# Sliding window data
sliding_windows = get_sliding_window_with_newvals(evaluation_data)
#%%
# SW-UCB Algorithm
def sw_ucb(sliding_windows, num_trials=100, confidence_level=1.5):
    num_arms = len(sliding_windows[1])
    rewards = np.zeros(num_arms)
    selection_counts = np.zeros(num_arms)
    total_rewards = []
    for t in range(1, num_trials + 1):
        ucb_values = []
        for arm in range(num_arms):
            if selection_counts[arm] == 0:
                ucb = float('inf')
            else:
                mean_reward = rewards[arm] / selection_counts[arm]
                ucb = mean_reward + confidence_level * np.sqrt(np.log(t) / selection_counts[arm])
            ucb_values.append(ucb)
        chosen_arm = np.argmax(ucb_values)
        reward = np.mean(sliding_windows[1][chosen_arm]) if len(sliding_windows[1][chosen_arm]) > 0 else 0
        rewards[chosen_arm] += reward
        selection_counts[chosen_arm] += 1
        total_rewards.append(reward)
    return np.array(total_rewards), selection_counts
#%%

# Discounted UCB Algorithm
def discounted_ucb(sliding_windows, num_trials=100, confidence_level=1.5, discount_factor=0.9):
    num_arms = len(sliding_windows[1])
    discounted_rewards = np.zeros(num_arms)
    discounted_counts = np.zeros(num_arms)
    total_rewards = []
    for t in range(1, num_trials + 1):
        ucb_values = []
        for arm in range(num_arms):
            if discounted_counts[arm] == 0:
                ucb = float('inf')
            else:
                mean_reward = discounted_rewards[arm] / discounted_counts[arm]
                ucb = mean_reward + confidence_level * np.sqrt(np.log(t) / discounted_counts[arm])
            ucb_values.append(ucb)
        chosen_arm = np.argmax(ucb_values)
        reward = np.mean(sliding_windows[1][chosen_arm]) if len(sliding_windows[1][chosen_arm]) > 0 else 0
        discounted_rewards *= discount_factor
        discounted_counts *= discount_factor
        discounted_rewards[chosen_arm] += reward
        discounted_counts[chosen_arm] += 1
        total_rewards.append(reward)
    return np.array(total_rewards), discounted_counts
#%%
# Run both algorithms and store results
sw_rewards, sw_selection_counts = sw_ucb(sliding_windows)
discounted_rewards, discounted_selection_counts = discounted_ucb(sliding_windows)

# Metrics comparison
def compare_algorithms(sw_rewards, discounted_rewards):
    metrics = {
        "SW-UCB": {
            "Cumulative Rewards": np.sum(sw_rewards),
            "Average Reward": np.mean(sw_rewards)
        },
        "Discounted UCB": {
            "Cumulative Rewards": np.sum(discounted_rewards),
            "Average Reward": np.mean(discounted_rewards)
        }
    }
    return metrics

comparison_results = compare_algorithms(sw_rewards, discounted_rewards)

# Print results
for algo, metrics in comparison_results.items():
    print(f"Algorithm: {algo}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
#%%
# Visualization
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(sw_rewards), label="SW-UCB Cumulative Rewards")
plt.plot(np.cumsum(discounted_rewards), label="Discounted UCB Cumulative Rewards")
plt.xlabel("Trial")
plt.ylabel("Cumulative Rewards")
plt.title("Cumulative Rewards Comparison")
plt.legend()
plt.show()