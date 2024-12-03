import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class TraceDriftEnvironment:
    def __init__(self, trace_length=100, drift_probability=0.1):
        """
        Simulates a walking trace sequence with possible drifts.
        :param trace_length: Length of each trace (queue size).
        :param drift_probability: Probability of introducing a drift at each step.
        """
        self.trace_length = trace_length
        self.drift_probability = drift_probability
        self.current_trace = deque(np.random.randint(0, 2, size=trace_length), maxlen=trace_length)
        self.index = 0

    def generate_trace(self):
        """
        Generates the next trace dynamically by adding a random 0 or 1.
        Introduces drift with a probability defined by drift_probability.
        """
        if np.random.rand() < self.drift_probability:
            # Introduce a drift by flipping a random bit in the current trace
            random_index = np.random.choice(len(self.current_trace))
            self.current_trace[random_index] = 1 - self.current_trace[random_index]
        else:
            # Append a random 0 or 1 to the trace
            self.current_trace.append(np.random.randint(0, 2))
        return list(self.current_trace)

    def step(self):
        """
        Simulates one step in the environment.
        Returns the current trace and advances to the next.
        """
        next_trace = self.generate_trace()
        self.index += 1
        return next_trace, False  # Always return False to simulate continuous walking


def detect_drift(traces, window_size=5, threshold=0.5):
    """
    Detects drift in the traces using a sliding window approach.
    :param traces: List of traces to analyze.
    :param window_size: Size of the window for comparison.
    :param threshold: Drift score threshold for detecting drift.
    """
    drift_indices = []
    drift_scores = []  # Collect drift scores for analysis
    for i in range(len(traces) - window_size):
        current_window = traces[i:i + window_size]
        next_trace = traces[i + window_size]

        # Compare the average of the current window with the next trace
        window_mean = np.mean(current_window, axis=0)
        drift_score = np.sum(np.abs(window_mean - next_trace))  # L1 norm

        # Normalize drift score by trace length
        normalized_score = drift_score / len(next_trace)
        drift_scores.append(normalized_score)

        if normalized_score > threshold:  # Adjust threshold for sensitivity
            drift_indices.append(i + window_size)

    # Print or analyze drift scores
    print("\nDrift Scores:", drift_scores)

    return drift_indices, drift_scores


# Example usage with adjusted parameters
env = TraceDriftEnvironment(trace_length=10, drift_probability=0.3)
traces = []

# Generate 100 walking traces
for _ in range(100):
    trace, _ = env.step()
    traces.append(trace)

# Detect drift with adjusted parameters
drift_indices, drift_scores = detect_drift(traces, window_size=5, threshold=0.6)
print("\nDrift Detected at Indices:", drift_indices)

# Plotting the results
plt.figure(figsize=(12, 6))

# Plot drift scores
plt.plot(drift_scores, label="Drift Scores", marker="o", linestyle="-", color="blue")

# Highlight drift detection points
for idx in drift_indices:
    plt.axvline(x=idx - 5, color="red", linestyle="--", label="Drift Detected" if idx == drift_indices[0] else "")

plt.title("Drift Detection Over Time")
plt.xlabel("Trace Index (Sliding Window End)")
plt.ylabel("Drift Score")
plt.legend()
plt.grid()
plt.show()