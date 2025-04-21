import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


class DriftMetricsAnalyzer:
    """Class to analyze and visualize the performance of different drift detection metrics."""

    def __init__(self, results_folder="./drift_results"):
        self.results_folder = results_folder
        self.metrics_data = {}
        Path(results_folder).mkdir(exist_ok=True)

    def save_experiment_results(
        self, metric_name, drift_strengths, drift_scores, avg_predictions
    ):
        """Save experiment results to a JSON file."""
        results = {
            "metric": metric_name,
            "drift_strengths": drift_strengths,
            "drift_scores": drift_scores,
            "avg_predictions": avg_predictions,
        }

        output_file = os.path.join(self.results_folder, f"{metric_name}_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f)

        self.metrics_data[metric_name] = results

    def load_experiment_results(self):
        """Load all experiment results from JSON files in the results folder."""
        self.metrics_data = {}  # Clear existing data
        for file in os.listdir(self.results_folder):
            if file.endswith("_results.json"):
                with open(os.path.join(self.results_folder, file)) as f:
                    results = json.load(f)
                    self.metrics_data[results["metric"]] = results

        return len(self.metrics_data) > 0

    def plot_comparative_analysis(self):
        """Generate a comparative plot of distance growth rates split by metric type:
        - Distribution-based distances (left)
        - Vector-based distances (right)
        """
        # Try to load results first if metrics_data is empty
        if not self.metrics_data:
            if not self.load_experiment_results():
                print(
                    "No metrics data loaded. Please run experiments or load results first."
                )
                return

        # Create a figure with two side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Define which metrics belong to which category
        distribution_based = [
            "js",
            "wasserstein",
            "kl",
            "hellinger",
            "bhattacharyya",
            "chi_square",
            "total_variation",
        ]
        vector_based = [
            "euclidean",
            "mahalanobis",
            "cosine",
            "manhattan",
            "minkowski",
            "chebyshev",
        ]

        # Plot distribution-based distances (left)
        self._plot_distance_growth_rate_by_type(
            ax1, distribution_based, "Distribution-based Distances"
        )

        # Plot vector-based distances (right)
        self._plot_distance_growth_rate_by_type(
            ax2, vector_based, "Vector-based Distances"
        )

        # Add main title for the figure
        fig.suptitle(
            "DeepFM Model Drift Detection: Comparative Analysis of Distance Metrics",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
        )  # Adjust layout to make room for the title
        plt.savefig(os.path.join(self.results_folder, "drift_metrics_comparison.png"))
        plt.show()

    def _plot_distance_growth_rate_by_type(self, ax, metric_types, title):
        """Plot distance growth rate for a specific category of metrics.

        Args:
            ax: The matplotlib axis to plot on
            metric_types: List of metric names belonging to this category
            title: Title for the subplot

        """
        drift_strengths = next(iter(self.metrics_data.values()))["drift_strengths"]

        has_data = False
        for metric, data in self.metrics_data.items():
            # Only plot metrics that belong to the specified category
            if metric.lower() in metric_types:
                distances = data["drift_scores"]

                # Calculate distance growth between consecutive drift strengths
                growth_rates = []
                for i in range(1, len(distances)):
                    if drift_strengths[i] - drift_strengths[i - 1] > 0:
                        rate = (distances[i] - distances[i - 1]) / (
                            drift_strengths[i] - drift_strengths[i - 1]
                        )
                        growth_rates.append(rate)

                # Plot at midpoints
                midpoints = [
                    (drift_strengths[i] + drift_strengths[i - 1]) / 2
                    for i in range(1, len(drift_strengths))
                ]
                ax.plot(
                    midpoints,
                    growth_rates,
                    "o-",
                    linewidth=2,
                    markersize=8,
                    label=metric,
                )
                has_data = True

        ax.set_title(title)
        ax.set_xlabel("Drift Strength")
        ax.set_ylabel("Distance Growth Rate")
        ax.grid(True, linestyle="--", alpha=0.7)

        if has_data:
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No metrics of this type found",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )

    def _plot_raw_distances(self, ax):
        """Plot all metrics showing their actual distance values."""
        drift_strengths = next(iter(self.metrics_data.values()))["drift_strengths"]

        for metric, data in self.metrics_data.items():
            distances = np.array(data["drift_scores"])
            ax.plot(
                drift_strengths,
                distances,
                "o-",
                linewidth=2,
                markersize=8,
                label=f"{metric}",
            )

        ax.set_title("Actual Distance Values by Metric")
        ax.set_xlabel("Drift Strength")
        ax.set_ylabel("Distance Value")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    def _plot_log_distances(self, ax):
        """Plot all metrics showing their actual distance values on a log scale
        to better visualize metrics with different ranges.
        """
        drift_strengths = next(iter(self.metrics_data.values()))["drift_strengths"]

        for metric, data in self.metrics_data.items():
            distances = np.array(data["drift_scores"])
            # Add small epsilon to handle zero values when using log scale
            distances = np.maximum(distances, 1e-10)
            ax.plot(
                drift_strengths,
                distances,
                "o-",
                linewidth=2,
                markersize=8,
                label=f"{metric}",
            )

        ax.set_title("Distance Values by Metric (Log Scale)")
        ax.set_xlabel("Drift Strength")
        ax.set_ylabel("Distance Value (log scale)")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    def _plot_correlation_with_predictions(self, ax):
        """Plot relationship between distance values and prediction changes."""
        for metric, data in self.metrics_data.items():
            drift_distances = data["drift_scores"]
            predictions = data["avg_predictions"]

            # Calculate changes from baseline
            baseline_prediction = predictions[0]  # assuming first is baseline
            prediction_changes = [abs(p - baseline_prediction) for p in predictions]

            ax.scatter(
                drift_distances, prediction_changes, label=metric, alpha=0.7, s=80
            )

            # Add best fit line
            if len(drift_distances) > 1:
                z = np.polyfit(drift_distances, prediction_changes, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(drift_distances), max(drift_distances), 100)
                ax.plot(x_range, p(x_range), "--", alpha=0.5)

        ax.set_title("Correlation: Distance Values vs Prediction Changes")
        ax.set_xlabel("Distance Value")
        ax.set_ylabel("Absolute Change in Predictions")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()

    def _plot_distance_growth_rate(self, ax):
        """Plot how fast the distance values grow as drift increases."""
        drift_strengths = next(iter(self.metrics_data.values()))["drift_strengths"]

        for metric, data in self.metrics_data.items():
            distances = data["drift_scores"]

            # Calculate distance growth between consecutive drift strengths
            growth_rates = []
            for i in range(1, len(distances)):
                if drift_strengths[i] - drift_strengths[i - 1] > 0:
                    rate = (distances[i] - distances[i - 1]) / (
                        drift_strengths[i] - drift_strengths[i - 1]
                    )
                    growth_rates.append(rate)

            # Plot at midpoints
            midpoints = [
                (drift_strengths[i] + drift_strengths[i - 1]) / 2
                for i in range(1, len(drift_strengths))
            ]
            ax.plot(
                midpoints, growth_rates, "o-", linewidth=2, markersize=8, label=metric
            )

        ax.set_title("Distance Growth Rate")
        ax.set_xlabel("Drift Strength")
        ax.set_ylabel("Distance Growth Rate")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend()


def main():
    analyzer = DriftMetricsAnalyzer("./drift_results")

    # First try to load real results from files
    if analyzer.load_experiment_results():
        print("Loaded existing experiment results.")
        analyzer.plot_comparative_analysis()
        return


if __name__ == "__main__":
    main()
