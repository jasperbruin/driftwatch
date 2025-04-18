#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Generator, Tuple, Sequence

import numpy as np
import pandas as pd
import psutil
from sklearn.decomposition import IncrementalPCA, PCA
from tqdm.auto import tqdm

try:
    from datasketches import kll_floats_sketch
except ImportError as e:
    raise SystemExit(
        "datasketches package not found. Install with `pip install datasketches`."
    ) from e

###############################################################################
# Utility helpers
###############################################################################

def generate_stream(
    n_samples: int,
    n_features: int,
    batch_size: int,
    distribution: str = "normal",
    variance: float = 1.0,
    seed: int = 42,
) -> Generator[np.ndarray, None, None]:
    rng = np.random.default_rng(seed)
    n_batches = int(np.ceil(n_samples / batch_size))
    for _ in range(n_batches):
        if distribution == "normal":
            batch = rng.normal(0.0, variance, size=(batch_size, n_features))
        elif distribution == "uniform":
            batch = rng.uniform(-variance, variance, size=(batch_size, n_features))
        elif distribution == "laplace":
            batch = rng.laplace(0.0, variance, size=(batch_size, n_features))
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        yield batch.astype(np.float32)


def current_mem_rss() -> int:
    return psutil.Process().memory_info().rss


def sketch_serialized_size(sketch: kll_floats_sketch) -> int:
    return len(sketch.serialize())

###############################################################################
# Accuracy helpers
###############################################################################

def pca_reconstruction_mse(pca_obj, X: np.ndarray) -> float:
    try:
        comps = pca_obj.transform(X)
        recon = pca_obj.inverse_transform(comps)
        return float(np.mean(np.square(X - recon)))
    except Exception:
        return np.nan


def kll_median_error(sketches: Sequence[kll_floats_sketch], X: np.ndarray) -> float:
    errs = []
    for j, s in enumerate(sketches):
        approx = s.get_quantile(0.5)
        truth = np.median(X[:, j])
        errs.append(abs(float(truth - approx)))
    return float(np.mean(errs))

###############################################################################
# Main experiment
###############################################################################

def run_experiment(
    n_samples: int = 100_000,
    n_features: int = 20,
    batch_size: int = 1_000,
    n_components: int = 5,
    kll_k: int = 200,
    distribution: str = "normal",
    variance: float = 1.0,
    seed: int = 42,
    compute_accuracy: bool = False,
    include_batch_pca: bool = False,
) -> Tuple[pd.DataFrame, IncrementalPCA, list[kll_floats_sketch]]:

    stream = generate_stream(
        n_samples,
        n_features,
        batch_size,
        distribution=distribution,
        variance=variance,
        seed=seed,
    )

    n_batches = int(np.ceil(n_samples / batch_size))
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    sketches = [kll_floats_sketch(kll_k) for _ in range(n_features)]
    all_data = [] if include_batch_pca else None

    metrics: list[dict[str, float | int]] = []

    for i, batch in enumerate(tqdm(stream, total=n_batches, unit="batch"), 1):
        if include_batch_pca:
            all_data.append(batch)

        # ---- IPCA ----
        mem_before = current_mem_rss()
        t0 = time.perf_counter()
        ipca.partial_fit(batch)
        ipca_time = time.perf_counter() - t0
        ipca_mem_delta = current_mem_rss() - mem_before

        pca_mse = pca_reconstruction_mse(ipca, batch) if compute_accuracy else np.nan

        # ---- KLL ----
        mem_before = current_mem_rss()
        t0 = time.perf_counter()
        for j in range(n_features):
            sketches[j].update(batch[:, j])
        kll_time = time.perf_counter() - t0
        kll_mem_delta = current_mem_rss() - mem_before

        kll_err = kll_median_error(sketches, batch) if compute_accuracy else np.nan
        kll_bytes_payload = sum(sketch_serialized_size(s) for s in sketches)

        throughput = batch_size / (ipca_time + kll_time) if (ipca_time + kll_time) > 0 else float("inf")

        metrics.append(
            {
                "batch": i,
                "seen_samples": int(min(i * batch_size, n_samples)),
                "ipca_time_s": ipca_time,
                "kll_time_s": kll_time,
                "throughput_sps": throughput,
                "ipca_mem_delta": ipca_mem_delta,
                "kll_mem_delta": kll_mem_delta,
                "kll_bytes": kll_bytes_payload,
                "pca_mse": pca_mse,
                "kll_median_err": kll_err,
            }
        )

    df = pd.DataFrame(metrics)

    if include_batch_pca:
        full_X = np.vstack(all_data)
        batch_pca = PCA(n_components=n_components)
        t0 = time.perf_counter()
        batch_pca.fit(full_X)
        fit_time = time.perf_counter() - t0
        recon_mse = pca_reconstruction_mse(batch_pca, full_X) if compute_accuracy else np.nan
        rss = current_mem_rss()

        df.loc[len(df)] = {
            "batch": -1,
            "seen_samples": n_samples,
            "ipca_time_s": np.nan,
            "kll_time_s": np.nan,
            "throughput_sps": np.nan,
            "ipca_mem_delta": np.nan,
            "kll_mem_delta": np.nan,
            "kll_bytes": np.nan,
            "pca_mse": recon_mse,
            "kll_median_err": np.nan,
            "batch_pca_time_s": fit_time,
            "batch_pca_rss_bytes": rss,
        }

    return df, ipca, sketches

###############################################################################
# Plotting helpers
###############################################################################

def plot_metrics(df: pd.DataFrame, outdir: Path):
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    # Time per batch
    fig = plt.figure()
    plt.plot(df["batch"], df["ipca_time_s"], label="Incremental PCA")
    plt.plot(df["batch"], df["kll_time_s"], label="KLL Sketches")
    plt.xlabel("Batch #")
    plt.ylabel("Time per batch (s)")
    plt.legend()
    plt.title("Compute time per batch")
    fig.tight_layout()
    fig.savefig(outdir / "time_per_batch.png", dpi=200)

    # Memory delta – twin axis (MB vs KB)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df["batch"], df["ipca_mem_delta"] / 1e6, label="IPCA ΔMB", color="tab:blue")
    ax2.plot(df["batch"], df["kll_mem_delta"] / 1e3, label="KLL ΔKB", color="tab:orange")

    ax1.set_xlabel("Batch #")
    ax1.set_ylabel("IPCA memory delta (MB)")
    ax2.set_ylabel("KLL memory delta (KB)")
    ax1.set_title("Memory delta per batch")

    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(outdir / "memory_deltas.png", dpi=200)

    if "pca_mse" in df.columns and df["pca_mse"].notna().any():
        fig = plt.figure()
        plt.plot(df["batch"], df["pca_mse"], label="PCA Reconstruction MSE")
        plt.xlabel("Batch #")
        plt.ylabel("MSE")
        plt.title("PCA accuracy over stream")
        plt.tight_layout()
        fig.savefig(outdir / "pca_mse.png", dpi=300)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming PCA vs KLL experiment")
    parser.add_argument("--samples", type=int, default=1_000_00)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--kll-k", type=int, default=10, help="Sketch k parameter")
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["normal", "uniform", "laplace"],
        default="normal",
    )
    parser.add_argument("--variance", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute-accuracy", action="store_true")
    parser.add_argument("--include-batch-pca", action="store_true")
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    df, _, _ = run_experiment(
        n_samples=args.samples,
        n_features=args.features,
        batch_size=args.batch_size,
        n_components=args.n_components,
        kll_k=args.kll_k,
        distribution=args.distribution,
        variance=args.variance,
        seed=args.seed,
        compute_accuracy=args.compute_accuracy,
        include_batch_pca=args.include_batch_pca,
    )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics csv to {csv_path}")

    try:
        plot_metrics(df, outdir)
        print(f"Saved plots in {outdir}")
    except ImportError:
        print("matplotlib not installed; skipping plots.")

###############################################################################
# Module execution guard
###############################################################################

if __name__ == "__main__":
    main()
