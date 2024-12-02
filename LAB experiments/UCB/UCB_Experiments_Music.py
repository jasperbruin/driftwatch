import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm


class DriftEnvironment:
    """
    Environment to simulate drift using a binary array of successes (1) and failures (0).
    """
    def __init__(self, drift_array, budget_ratio=0.5):
        self.drift_array = np.array(drift_array)
        self.length = len(drift_array)  # Number of timesteps
        self.budget_ratio = budget_ratio
        self.index = 0
        self.rewards = []
        self.mode = None  # To track the current mode (train or test)

    def reset(self, mode=None):
        """
        Resets the environment to the initial state.
        :param mode: Optional mode argument ("train" or "test")
        """
        self.index = 0
        self.rewards = []
        self.mode = mode
        if mode:
            print(f"Environment reset to mode: {mode}")

    def step(self, action):
        """
        Simulates one step in the environment.
        """
        assert len(action) == len(
            self.drift_array), "Action size must match drift array size."

        # Adding small noise to the action for randomness
        action += np.random.rand(*action.shape) / 100

        # Get the indices of the top actions to "check"
        check_index = np.argsort(action)[::-1][
                      :int(self.budget_ratio * len(self.drift_array))]

        # Create the state based on selected actions
        state = np.ones_like(self.drift_array) * -1
        state[check_index] = self.drift_array[check_index]

        # Calculate reward as the success rate for checked items
        reward = state[state != -1].sum() / len(self.drift_array)
        self.rewards.append(reward)

        # Check if the simulation should stop
        stop = self.index == self.length - 1
        self.index += 1

        return reward, state, stop

    def get_total_reward(self):
        """
        Returns the mean reward over all timesteps.
        """
        return np.mean(self.rewards).round(3)



class SlidingWindowUCBAgent:
    """
    Sliding Window UCB Agent for action selection.
    """

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.c = 3
        self.recent_rewards = None
        self.recent_counts = None
        self.recent_rewards_sum = None
        self.recent_counts_sum = None
        self.total_time_steps = 0

    def initialize(self, n_actions):
        self.recent_rewards = [deque(maxlen=self.window_size) for _ in
                               range(n_actions)]
        self.recent_counts = [deque(maxlen=self.window_size) for _ in
                              range(n_actions)]
        self.recent_rewards_sum = np.zeros(n_actions)
        self.recent_counts_sum = np.zeros(n_actions)

    def get_action(self):
        """
        Returns the action vector based on UCB values.
        """
        min_time_steps = max(1, min(self.total_time_steps, self.window_size))
        recent_values = self.recent_rewards_sum / np.maximum(
            self.recent_counts_sum, 1)
        ucb_values = recent_values + self.c * np.sqrt(
            2 * np.log(min_time_steps) / np.maximum(self.recent_counts_sum, 1))
        return ucb_values

    def update(self, actions, state):
        """
        Updates the agent with the reward from the environment.
        """
        self.total_time_steps += 1
        for i, reward in enumerate(state):
            if reward >= 0:
                if len(self.recent_rewards[i]) == self.window_size:
                    self.recent_rewards_sum[i] -= self.recent_rewards[i][0]
                    self.recent_counts_sum[i] -= self.recent_counts[i][0]

                self.recent_rewards[i].append(reward)
                self.recent_counts[i].append(1)
                self.recent_rewards_sum[i] += reward
                self.recent_counts_sum[i] += 1
            else:
                self.recent_rewards[i].append(0)
                self.recent_counts[i].append(0)


def test_drift_array():
    drift_array = np.random.randint(0, 2, size=100)  # Example drift array
    max_window_size = 100
    np.random.seed(0)
    env = DriftEnvironment(drift_array)
    best_window_size = 1
    best_reward = -np.inf

    window_sizes = []
    total_rewards = []

    for window_size in range(1, max_window_size + 1):
        agent = SlidingWindowUCBAgent(window_size=window_size)
        agent.initialize(len(drift_array))

        env.reset()
        for _ in tqdm(range(env.length)):
            action = agent.get_action()
            reward, state, stop = env.step(action)
            agent.update(action, state)
            if stop:
                break

        total_reward = env.get_total_reward()
        print(f"Window size: {window_size}, Total reward: {total_reward}")

        if total_reward > best_reward:
            best_reward = total_reward
            best_window_size = window_size

        window_sizes.append(window_size)
        total_rewards.append(total_reward)

    print(f"Best window size: {best_window_size}, Total reward: {best_reward}")



import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_random_binary_arrays(num_arrays, size=10):
    """
    Generates random binary arrays of a given size.
    """
    return [np.random.randint(0, 2, size) for _ in range(num_arrays)]


def test_random_arrays(num_arrays=5, array_size=100, max_window_size=50):
    """
    Tests Sliding Window UCB on random binary arrays and evaluates its ability to handle drift.
    """
    random_arrays = generate_random_binary_arrays(num_arrays, array_size)
    drift_rewards = []

    for i, drift_array in enumerate(random_arrays):
        print(f"\nTesting array {i+1}: {drift_array}")

        # Initialize Drift Environment
        env = DriftEnvironment(drift_array, budget_ratio=0.5)
        best_window_size = 1
        best_reward = -np.inf
        window_sizes = []
        total_rewards = []

        # Evaluate performance on each drift array
        for window_size in range(1, max_window_size + 1):
            agent = SlidingWindowUCBAgent(window_size=window_size)
            agent.initialize(len(drift_array))

            env.reset()
            for _ in range(env.length):
                action = agent.get_action()
                reward, state, stop = env.step(action)
                agent.update(action, state)
                if stop:
                    break

            total_reward = env.get_total_reward()
            window_sizes.append(window_size)
            total_rewards.append(total_reward)

            if total_reward > best_reward:
                best_reward = total_reward
                best_window_size = window_size

        # Log and save rewards over time
        agent = SlidingWindowUCBAgent(window_size=best_window_size)
        agent.initialize(len(drift_array))
        env.reset()
        rewards_over_time = []

        for _ in range(env.length):
            action = agent.get_action()
            reward, state, stop = env.step(action)
            agent.update(action, state)
            rewards_over_time.append(reward)
            if stop:
                break

        drift_rewards.append((drift_array, rewards_over_time, best_window_size, best_reward))

        print(f"Best window size for array {i+1}: {best_window_size}")
        print(f"Best total reward for array {i+1}: {best_reward}")

    # Plot results
    for i, (drift_array, rewards_over_time, best_window_size, best_reward) in enumerate(drift_rewards):
        plt.plot(rewards_over_time, label=f"Array {i+1}: Best WS={best_window_size}, Reward={best_reward:.3f}")
        plt.title(f"Rewards Over Time for Random Arrays")
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()



import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_and_test(agent_class, env, hyperparameter_name, hyperparameter_values):
    best_hyperparameter = None
    best_reward = -np.inf

    # Training phase
    rewards_train = []
    for value in hyperparameter_values:
        agent = agent_class(value)  # Initialize agent with hyperparameter
        agent.initialize(len(env.drift_array))

        env.reset(mode='train')  # Set environment to training mode
        for t in tqdm(range(env.length)):
            action = agent.get_action()
            reward, state, stop = env.step(action)
            agent.update(action, state)
            if stop:
                break

        total_reward = env.get_total_reward()
        print(f"Training: {hyperparameter_name}={value}, Reward={total_reward}")
        rewards_train.append(total_reward)

        if total_reward > best_reward:
            best_reward = total_reward
            best_hyperparameter = value

    # Testing phase
    agent = agent_class(best_hyperparameter)
    agent.initialize(len(env.drift_array))
    env.reset(mode='test')  # Set environment to testing mode

    rewards_test = []
    for t in tqdm(range(env.length)):
        action = agent.get_action()
        reward, state, stop = env.step(action)
        rewards_test.append(reward)
        if stop:
            break

    test_total_reward = env.get_total_reward()
    print(f"Testing: Best {hyperparameter_name}={best_hyperparameter}, Reward={test_total_reward}")

    # Plotting results
    plt.plot(hyperparameter_values, rewards_train, label='Training Rewards')
    plt.xlabel(hyperparameter_name)
    plt.ylabel('Total Reward')
    plt.title(f"{agent_class.__name__} Performance")
    plt.legend()
    plt.grid()
    plt.show()

    return best_hyperparameter, test_total_reward


# Define the drift array
drift_array = np.random.randint(0, 2, size=10)

# Create the environment
env = DriftEnvironment(drift_array=drift_array)

# Test the train_and_test function with SlidingWindowUCBAgent
train_and_test(SlidingWindowUCBAgent, env, "Window Size", range(10, 30))






