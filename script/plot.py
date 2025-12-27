import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple

os.makedirs("plots", exist_ok=True)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def _tile_sort_key(label):
    # label like "16x16" or "No tiling"
    if label == "No tiling":
        return (float('inf'), float('inf'))
    try:
        a, b = label.split("x")
        return (int(a), int(b))
    except:
        return (float('inf'), float('inf'))

def remove_outliers(series, method="std", thresh=2.0):
    """Remove outliers from a pandas Series."""
    if series.empty:
        return series
    if method == "std":
        mean = series.mean()
        std = series.std()
        return series[(series >= mean - thresh*std) & (series <= mean + thresh*std)]
    elif method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return series[(series >= q1 - 1.5*iqr) & (series <= q3 + 1.5*iqr)]
    else:
        return series

# -------------------------------
# Project GEMM/GEMV (matrix sizes + tiling, include MKL)
# -------------------------------
def parse_project(json_path):
    data = load_json(json_path)
    results = []
    for b in data["benchmarks"]:
        if b.get("run_type") != "iteration":
            continue
        if any(tag in b["name"] for tag in ["_mean","_median","_stddev","_cv"]):
            continue
        if b.get("repetition_index", 0) == 0:
            continue

        name = b["name"]
        parts = name.split("/")

        # Handle BM_GEMM1 and BM_GEMV1 (matrix size scaling benchmarks)
        if name.startswith("BM_GEMM1") and len(parts) >= 6:
            matrix_size = int(parts[1])  # M dimension
            tile1, tile2 = int(parts[4]), int(parts[5])
            tile_label = f"{tile1}x{tile2}" if tile1 > 0 else "No tiling"
            bench = f"GEMM1_M{matrix_size}"
            size = matrix_size
        elif name.startswith("BM_GEMV1") and len(parts) >= 5:
            matrix_size = int(parts[1])  # M dimension
            tile1, tile2 = int(parts[3]), int(parts[4])
            tile_label = f"{tile1}x{tile2}" if tile1 > 0 else "No tiling"
            bench = f"GEMV1_M{matrix_size}"
            size = matrix_size
        # Handle original BM_GEMM and BM_GEMV (tile size scaling)
        elif name.startswith("BM_GEMM") and not name.startswith("BM_GEMM_MKL") and not name.startswith("BM_GEMM1") and len(parts) >= 6:
            tile1, tile2 = int(parts[4]), int(parts[5])
            tile_label = f"{tile1}x{tile2}" if tile1 > 0 else "No tiling"
            bench = "GEMM"
            size = int(parts[1])
        elif name.startswith("BM_GEMV") and not name.startswith("BM_GEMV_MKL") and not name.startswith("BM_GEMV1") and len(parts) >= 5:
            tile1, tile2 = int(parts[3]), int(parts[4])
            tile_label = f"{tile1}x{tile2}" if tile1 > 0 else "No tiling"
            bench = "GEMV"
            size = int(parts[1])
        # Handle MKL benchmarks
        elif name.startswith("BM_GEMM_MKL1"):
            matrix_size = int(parts[1])
            tile_label = "No tiling"
            bench = f"GEMM_MKL1_M{matrix_size}"
            size = matrix_size
        elif name.startswith("BM_GEMV_MKL1"):
            matrix_size = int(parts[1])
            tile_label = "No tiling"
            bench = f"GEMV_MKL1_M{matrix_size}"
            size = matrix_size
        elif name.startswith("BM_GEMM_MKL"):
            tile_label = "No tiling"
            bench = "GEMM_MKL"
            size = int(parts[1])
        elif name.startswith("BM_GEMV_MKL"):
            tile_label = "No tiling"
            bench = "GEMV_MKL"
            size = int(parts[1])
        else:
            continue

        val = float(b["cpu_time"])
        unit = b["time_unit"]
        # normalize to µs
        if unit == "ms":
            val *= 1000.0
        elif unit == "s":
            val *= 1_000_000.0

        results.append({
            "bench": bench,
            "tile": tile_label,
            "time_us": val,
            "size": size,
            "repetition_index": b.get("repetition_index", None)
        })
    return pd.DataFrame(results)

def plot_project(df):
    if df.empty:
        print("No GEMM/GEMV entries found.")
        return

    # Separate tile scaling benchmarks (original GEMM/GEMV)
    tiling_df = df[df["bench"].isin(["GEMM","GEMV"])].copy()
    mkl_df = df[df["bench"].isin(["GEMM_MKL","GEMV_MKL"])].copy()

    # Separate matrix size scaling benchmarks (GEMM1/GEMV1)
    size_scaling_df = df[df["bench"].str.contains("GEMM1_M|GEMV1_M", regex=True)].copy()
    mkl_size_scaling_df = df[df["bench"].str.contains("GEMM_MKL1_M|GEMV_MKL1_M", regex=True)].copy()

    # --- Plot 1: Tile size scaling (original benchmarks) ---
    if not tiling_df.empty:
        cleaned = []
        for (bench, tile), group in tiling_df.groupby(["bench","tile"]):
            times = remove_outliers(group["time_us"], method="std", thresh=2.0)
            if not times.empty:
                cleaned.append({"bench": bench, "tile": tile, "time_us": times.mean()})
        tiling_df = pd.DataFrame(cleaned)

        ordered_tiles = sorted(tiling_df["tile"].unique(), key=_tile_sort_key)

        plt.figure(figsize=(10,6))
        for bench in tiling_df["bench"].unique():
            d = tiling_df[tiling_df["bench"]==bench].copy()
            d["tile_order"] = d["tile"].apply(_tile_sort_key)
            d = d.sort_values("tile_order")
            plt.plot(d["tile"], d["time_us"], marker="o", linewidth=2, label=bench)
        plt.title("CPU GEMM/GEMV Performance by Tile Size (128x128, outliers removed)", fontsize=14)
        plt.xlabel("Tile size")
        plt.ylabel("Time (µs)")
        plt.xticks(ordered_tiles, rotation=45, ha="right")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/project_gemm_gemv_tile_scaling.png", dpi=200)
        plt.close()
        print("Saved plots/project_gemm_gemv_tile_scaling.png")

    # --- Plot 2: MKL baselines (original) ---
    if not mkl_df.empty:
        cleaned_mkl = []
        for bench, group in mkl_df.groupby("bench"):
            times = remove_outliers(group["time_us"], method="std", thresh=2.0)
            if not times.empty:
                cleaned_mkl.append({"bench": bench, "time_us": times.mean()})
        mkl_df = pd.DataFrame(cleaned_mkl)

        plt.figure(figsize=(6,5))
        colors = {"GEMM_MKL": "#4C72B0", "GEMV_MKL": "#DD8452"}
        plt.bar(mkl_df["bench"], mkl_df["time_us"], color=[colors[b] for b in mkl_df["bench"]])
        plt.title("CPU GEMM/GEMV MKL Baselines (128x128, outliers removed)", fontsize=14)
        plt.ylabel("Time (µs)")
        for i,(bench,val) in enumerate(zip(mkl_df["bench"], mkl_df["time_us"])):
            plt.text(i, val, f"{val:.2f} µs", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig("plots/project_gemm_gemv_mkl.png", dpi=200)
        plt.close()
        print("Saved plots/project_gemm_gemv_mkl.png")

    # --- Plot 3: Matrix size scaling (GEMM1/GEMV1 with fixed tile 64x64) ---
    if not size_scaling_df.empty:
        cleaned_size = []
        for bench, group in size_scaling_df.groupby("bench"):
            times = remove_outliers(group["time_us"], method="std", thresh=2.0)
            size = group["size"].iloc[0]
            if not times.empty:
                cleaned_size.append({
                    "bench": bench,
                    "size": size,
                    "time_us": times.mean(),
                    "operation": "GEMM" if "GEMM" in bench else "GEMV"
                })
        size_scaling_df = pd.DataFrame(cleaned_size)
        size_scaling_df = size_scaling_df.sort_values("size")

        plt.figure(figsize=(10,6))
        for op in size_scaling_df["operation"].unique():
            d = size_scaling_df[size_scaling_df["operation"]==op]
            plt.plot(d["size"], d["time_us"], marker="o", linewidth=2, label=f"{op} (tile 64x64)", markersize=8)

        plt.title("CPU GEMM/GEMV Performance Scaling with Matrix Size (Tile 64x64)", fontsize=14)
        plt.xlabel("Matrix Size (M=N=K)")
        plt.ylabel("Time (µs)")
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.6, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/project_gemm_gemv_size_scaling.png", dpi=200)
        plt.close()
        print("Saved plots/project_gemm_gemv_size_scaling.png")

    # --- Plot 4: MKL size scaling comparison ---
    if not mkl_size_scaling_df.empty:
        cleaned_mkl_size = []
        for bench, group in mkl_size_scaling_df.groupby("bench"):
            times = remove_outliers(group["time_us"], method="std", thresh=2.0)
            size = group["size"].iloc[0]
            if not times.empty:
                cleaned_mkl_size.append({
                    "bench": bench,
                    "size": size,
                    "time_us": times.mean(),
                    "operation": "GEMM" if "GEMM" in bench else "GEMV"
                })
        mkl_size_scaling_df = pd.DataFrame(cleaned_mkl_size)
        mkl_size_scaling_df = mkl_size_scaling_df.sort_values("size")

        # Combine custom and MKL for comparison
        plt.figure(figsize=(12,6))

        # Plot custom implementations
        for op in size_scaling_df["operation"].unique():
            d = size_scaling_df[size_scaling_df["operation"]==op]
            plt.plot(d["size"], d["time_us"], marker="o", linewidth=2,
                     label=f"Custom {op} (tile 64x64)", markersize=8, linestyle="-")

        # Plot MKL implementations
        for op in mkl_size_scaling_df["operation"].unique():
            d = mkl_size_scaling_df[mkl_size_scaling_df["operation"]==op]
            plt.plot(d["size"], d["time_us"], marker="s", linewidth=2,
                     label=f"MKL {op}", markersize=8, linestyle="--")

        plt.title("Custom vs MKL Performance Scaling with Matrix Size", fontsize=14)
        plt.xlabel("Matrix Size (M=N=K)")
        plt.ylabel("Time (µs)")
        plt.xscale("log", base=2)
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.6, which="both")
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/project_custom_vs_mkl_scaling.png", dpi=200)
        plt.close()
        print("Saved plots/project_custom_vs_mkl_scaling.png")

# -------------------------------
# Dense NN GEMM/GEMV (tiling sizes only, plot run times)
# -------------------------------
def parse_nn_cpu(json_path):
    data = load_json(json_path)
    results = []
    for b in data["benchmarks"]:
        if b.get("run_type") != "iteration":
            continue
        if any(tag in b["name"] for tag in ["_mean","_median","_stddev","_cv"]):
            continue
        if b.get("repetition_index", 0) == 0:
            continue

        name = b["name"]
        parts = name.split("/")

        # Handle DENSE benchmarks with "1" suffix (full dataset)
        if name.startswith("BM_DENSE_GEMM1") and len(parts) >= 3:
            tile1, tile2 = int(parts[1]), int(parts[2])
            tile_label = f"{tile1}x{tile2}"
            bench = "DENSE_GEMM_FULL"
        elif name.startswith("BM_DENSE_GEMV1") and len(parts) >= 3:
            tile1, tile2 = int(parts[1]), int(parts[2])
            tile_label = f"{tile1}x{tile2}"
            bench = "DENSE_GEMV_FULL"
        elif name.startswith("BM_DENSE_MKL_GEMM1"):
            tile_label = "No tiling"
            bench = "DENSE_MKL_GEMM_FULL"
        elif name.startswith("BM_DENSE_MKL_GEMV1"):
            tile_label = "No tiling"
            bench = "DENSE_MKL_GEMV_FULL"
        # Handle original DENSE benchmarks (10 samples)
        elif name.startswith("BM_DENSE_GEMM") and not name.startswith("BM_DENSE_GEMM1") and len(parts) >= 3:
            tile1, tile2 = int(parts[1]), int(parts[2])
            tile_label = f"{tile1}x{tile2}"
            bench = "DENSE_GEMM"
        elif name.startswith("BM_DENSE_GEMV") and not name.startswith("BM_DENSE_GEMV1") and len(parts) >= 3:
            tile1, tile2 = int(parts[1]), int(parts[2])
            tile_label = f"{tile1}x{tile2}"
            bench = "DENSE_GEMV"
        elif name.startswith("BM_DENSE_MKL_GEMM") and not name.startswith("BM_DENSE_MKL_GEMM1"):
            tile_label = "No tiling"
            bench = "DENSE_MKL_GEMM"
        elif name.startswith("BM_DENSE_MKL_GEMV") and not name.startswith("BM_DENSE_MKL_GEMV1"):
            tile_label = "No tiling"
            bench = "DENSE_MKL_GEMV"
        else:
            continue

        val = float(b["cpu_time"])
        unit = b["time_unit"]
        if unit == "ms":
            val *= 1000.0
        elif unit == "s":
            val *= 1_000_000.0

        results.append({
            "bench": bench,
            "tile": tile_label,
            "time_us": val,
            "repetition_index": b.get("repetition_index", None)
        })
    return pd.DataFrame(results)

def plot_nn_cpu(df):
    if df.empty:
        print("No Dense NN entries found.")
        return

    # Separate 10-sample and full dataset benchmarks
    small_df = df[df["bench"].str.contains("DENSE_GEMM$|DENSE_GEMV$|DENSE_MKL_GEMM$|DENSE_MKL_GEMV$", regex=True)].copy()
    full_df = df[df["bench"].str.contains("_FULL", regex=True)].copy()

    # --- Plot 1: Small dataset (10 samples) ---
    if not small_df.empty:
        cleaned = []
        for bench, group in small_df.groupby("bench"):
            times = remove_outliers(group["time_us"], method="std", thresh=2.0)
            if not times.empty:
                cleaned.append({"bench": bench, "time_us": times.mean()})
        small_df = pd.DataFrame(cleaned)

        colors = {
            "DENSE_GEMM": "#4C72B0",
            "DENSE_GEMV": "#55A868",
            "DENSE_MKL_GEMM": "#C44E52",
            "DENSE_MKL_GEMV": "#8172B3"
        }

        small_df = small_df[small_df["bench"].isin(colors.keys())]

        plt.figure(figsize=(8,6))
        plt.bar(small_df["bench"], small_df["time_us"], color=[colors[b] for b in small_df["bench"]])
        plt.title("CPU Dense NN GEMM/GEMV (10 samples, outliers removed)", fontsize=14)
        plt.ylabel("Time (µs)")
        for i,(bench,val) in enumerate(zip(small_df["bench"], small_df["time_us"])):
            plt.text(i, val, f"{val:.2f} µs", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig("plots/nn_cpu_gemm_gemv_small.png", dpi=200)
        plt.close()
        print("Saved plots/nn_cpu_gemm_gemv_small.png")

    # --- Plot 2: Full dataset ---
    if not full_df.empty:
        cleaned_full = []
        for bench, group in full_df.groupby("bench"):
            times = remove_outliers(group["time_us"], method="std", thresh=2.0)
            if not times.empty:
                cleaned_full.append({"bench": bench, "time_us": times.mean()})
        full_df = pd.DataFrame(cleaned_full)

        colors = {
            "DENSE_GEMM_FULL": "#4C72B0",
            "DENSE_GEMV_FULL": "#55A868",
            "DENSE_MKL_GEMM_FULL": "#C44E52",
            "DENSE_MKL_GEMV_FULL": "#8172B3"
        }

        full_df = full_df[full_df["bench"].isin(colors.keys())]

        # Clean up labels for display
        full_df["display_name"] = full_df["bench"].str.replace("_FULL", "")

        plt.figure(figsize=(8,6))
        plt.bar(range(len(full_df)), full_df["time_us"],
                color=[colors[b] for b in full_df["bench"]])
        plt.xticks(range(len(full_df)), full_df["display_name"], rotation=15, ha="right")
        plt.title("CPU Dense NN GEMM/GEMV (Full Dataset, outliers removed)", fontsize=14)
        plt.ylabel("Time (µs)")
        for i,(bench,val) in enumerate(zip(full_df["bench"], full_df["time_us"])):
            plt.text(i, val, f"{val:.2f} µs", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig("plots/nn_cpu_gemm_gemv_full.png", dpi=200)
        plt.close()
        print("Saved plots/nn_cpu_gemm_gemv_full.png")

    # --- Plot 3: Comparison between small and full datasets ---
    if not small_df.empty and not full_df.empty:
        plt.figure(figsize=(10,6))

        x_labels = ["GEMM", "GEMV", "MKL_GEMM", "MKL_GEMV"]
        x = range(len(x_labels))
        width = 0.35

        small_vals = []
        full_vals = []
        for label in x_labels:
            small_key = f"DENSE_{label}"
            full_key = f"DENSE_{label}_FULL"

            small_val = small_df[small_df["bench"] == small_key]["time_us"].values
            full_val = full_df[full_df["bench"] == full_key]["time_us"].values

            small_vals.append(small_val[0] if len(small_val) > 0 else 0)
            full_vals.append(full_val[0] if len(full_val) > 0 else 0)

        plt.bar([i - width/2 for i in x], small_vals, width, label="10 samples", alpha=0.8)
        plt.bar([i + width/2 for i in x], full_vals, width, label="Full dataset", alpha=0.8)

        plt.xlabel("Implementation")
        plt.ylabel("Time (µs)")
        plt.title("Dense NN Performance: 10 Samples vs Full Dataset", fontsize=14)
        plt.xticks(x, x_labels, rotation=15, ha="right")
        plt.legend()
        plt.grid(True, axis='y', linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("plots/nn_cpu_comparison.png", dpi=200)
        plt.close()
        print("Saved plots/nn_cpu_comparison.png")


# -------------------------------
# CPU Sparse Kernels (SPMV/SPMM)
# -------------------------------
def parse_sparse_cpu(json_path):
    """Parse CPU sparse kernel benchmarks (SPMV and SPMM)."""
    data = load_json(json_path)
    results = []
    
    for b in data["benchmarks"]:
        if b.get("run_type") != "iteration":
            continue
        if any(tag in b["name"] for tag in ["_mean", "_median", "_stddev", "_cv"]):
            continue
        if b.get("repetition_index", 0) == 0:
            continue
        
        name = b["name"]
        parts = name.split("/")
        
        # Parse SPMV: BM_SPMV/m/n/..
        if name.startswith("BM_SPMV") and len(parts) >= 3:
            try:
                m = int(parts[1])
                bench = f"SPMV_{m}x{m}"
                time_us = b.get("real_time", 0)
                results.append({
                    "bench": "SPMV",
                    "size": m,
                    "bench_name": bench,
                    "time_us": time_us
                })
            except (ValueError, IndexError):
                continue
        
        # Parse SPMM: BM_SPMM/m/n/k/..
        elif name.startswith("BM_SPMM") and len(parts) >= 4:
            try:
                m = int(parts[1])
                bench = f"SPMM_{m}x{m}"
                time_us = b.get("real_time", 0)
                results.append({
                    "bench": "SPMM",
                    "size": m,
                    "bench_name": bench,
                    "time_us": time_us
                })
            except (ValueError, IndexError):
                continue
    
    if results:
        return pd.DataFrame(results)
    return pd.DataFrame()


def plot_sparse_cpu(df):
    """Plot CPU sparse kernel performance (SPMV and SPMM)."""
    if df.empty:
        print("No sparse CPU entries found.")
        return
    
    # Separate SPMV and SPMM
    spmv_df = df[df["bench"] == "SPMV"].copy()
    spmm_df = df[df["bench"] == "SPMM"].copy()
    
    # --- Plot 1: CPU SPMV Performance ---
    if not spmv_df.empty:
        spmv_df = spmv_df.sort_values("size")
        plt.figure(figsize=(10, 6))
        plt.plot(spmv_df["size"], spmv_df["time_us"], marker="o", linewidth=2.5, 
                 markersize=8, color="#2E86AB", label="SPMV")
        plt.xlabel("Matrix Size (m=n)", fontsize=12, fontweight="bold")
        plt.ylabel("Execution Time (µs)", fontsize=12, fontweight="bold")
        plt.title("CPU SPMV Performance vs Matrix Size", fontsize=14, fontweight="bold")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, alpha=0.3, linestyle="--", which="both")
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig("plots/cpu_spmv_performance.png", dpi=200)
        plt.close()
        print("Saved plots/cpu_spmv_performance.png")
    
    # --- Plot 2: CPU SPMM Performance ---
    if not spmm_df.empty:
        spmm_df = spmm_df.sort_values("size")
        plt.figure(figsize=(10, 6))
        plt.plot(spmm_df["size"], spmm_df["time_us"], marker="s", linewidth=2.5, 
                 markersize=8, color="#D62828", label="SPMM")
        plt.xlabel("Matrix Size (m=n=k)", fontsize=12, fontweight="bold")
        plt.ylabel("Execution Time (µs)", fontsize=12, fontweight="bold")
        plt.title("CPU SPMM Performance vs Matrix Size", fontsize=14, fontweight="bold")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, alpha=0.3, linestyle="--", which="both")
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig("plots/cpu_spmm_performance.png", dpi=200)
        plt.close()
        print("Saved plots/cpu_spmm_performance.png")
    
    # --- Plot 3: SPMV vs SPMM Comparison ---
    if not spmv_df.empty and not spmm_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(spmv_df["size"], spmv_df["time_us"], marker="o", linewidth=2.5, 
                 markersize=8, color="#2E86AB", label="SPMV")
        plt.plot(spmm_df["size"], spmm_df["time_us"], marker="s", linewidth=2.5, 
                 markersize=8, color="#D62828", label="SPMM")
        plt.xlabel("Matrix Size (m=n=k)", fontsize=12, fontweight="bold")
        plt.ylabel("Execution Time (µs)", fontsize=12, fontweight="bold")
        plt.title("CPU Sparse Kernels: SPMV vs SPMM Performance", fontsize=14, fontweight="bold")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, alpha=0.3, linestyle="--", which="both")
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig("plots/cpu_sparse_spmv_vs_spmm.png", dpi=200)
        plt.close()
        print("Saved plots/cpu_sparse_spmv_vs_spmm.png")


# -------------------------------
# Sparse vs Dense NN Accuracy Comparison
# -------------------------------
def plot_sparse_vs_dense_nn(sparsity_csv_path="logs/sparsity_accuracy_python.csv"):
    """
    Plot sparse vs dense NN accuracy comparison across sparsity levels.
    
    Reads sparsity_accuracy_python.csv from the Python sparsify script and creates:
    1. Accuracy degradation plot (sparse accuracy vs sparsity level)
    2. Dual-axis plot: accuracy + model compression ratio
    """
    if not os.path.exists(sparsity_csv_path):
        print(f"Sparsity accuracy file not found: {sparsity_csv_path}")
        return
    
    try:
        df = pd.read_csv(sparsity_csv_path)
    except Exception as e:
        print(f"Error reading sparsity CSV: {e}")
        return
    
    if df.empty:
        print("Sparsity accuracy CSV is empty")
        return
    
    # Extract data
    sparsity_levels = df['target_sparsity_pct'].values
    accuracies = df['accuracy'].values
    
    # Calculate dense baseline (0% sparsity would be 100% dense)
    # Assume the first entry is close to baseline or use max accuracy
    dense_accuracy = accuracies.max() if len(accuracies) > 0 else 1.0
    
    # --- Plot 1: Accuracy Degradation vs Sparsity ---
    plt.figure(figsize=(10, 6))
    plt.plot(sparsity_levels, accuracies, marker="o", linewidth=2.5, 
             markersize=8, color="#2E86AB", label="Sparse NN Accuracy")
    plt.axhline(y=dense_accuracy, color="#A23B72", linestyle="--", linewidth=2, 
                label=f"Dense NN Baseline ({dense_accuracy*100:.2f}%)")
    
    # Add annotations showing accuracy drop
    for i, (sparsity, acc) in enumerate(zip(sparsity_levels, accuracies)):
        drop_pct = (dense_accuracy - acc) / dense_accuracy * 100
        plt.text(sparsity, acc, f"{acc*100:.1f}%\n(-{drop_pct:.1f}%)", 
                ha="center", va="bottom", fontsize=8)
    
    plt.xlabel("Sparsity Level (%)", fontsize=12, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=12, fontweight="bold")
    plt.title("CPU Sparse NN – Accuracy Degradation vs Sparsity", fontsize=14, fontweight="bold")
    plt.ylim([accuracies.min() * 0.95, 1.05])
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.legend(fontsize=11, loc="lower left")
    plt.tight_layout()
    plt.savefig("plots/sparse_vs_dense_accuracy.png", dpi=200)
    plt.close()
    print("Saved plots/sparse_vs_dense_accuracy.png")
    
    # --- Plot 2: Dual-axis plot (Accuracy + Compression Ratio) ---
    fig, ax1 = plt.subplots(figsize=(11, 6))
    
    # Left axis: Accuracy
    color1 = "#2E86AB"
    ax1.plot(sparsity_levels, accuracies * 100, marker="o", linewidth=2.5, 
             markersize=8, color=color1, label="Accuracy")
    ax1.axhline(y=dense_accuracy * 100, color="gray", linestyle="--", 
                linewidth=1.5, alpha=0.7, label="Dense Baseline")
    ax1.set_xlabel("Sparsity Level (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim([accuracies.min() * 100 * 0.95, 105])
    ax1.grid(True, alpha=0.3, linestyle="--")
    
    # Right axis: Compression ratio (1 / (1 - sparsity))
    ax2 = ax1.twinx()
    compression_ratios = 1.0 / (1.0 - sparsity_levels / 100.0)
    color2 = "#D62828"
    ax2.plot(sparsity_levels, compression_ratios, marker="s", linewidth=2.5, 
             markersize=8, color=color2, label="Model Compression Ratio")
    ax2.set_ylabel("Model Size Reduction Factor", fontsize=12, fontweight="bold", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    
    plt.title("CPU Sparse NN – Accuracy vs Model Compression", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig("plots/sparse_vs_dense_accuracy_compression.png", dpi=200)
    plt.close()
    print("Saved plots/sparse_vs_dense_accuracy_compression.png")


#!/usr/bin/env python3
import json
import os
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


def load_nvbench_json(path: str) -> Dict:
    """Load an NVBench JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_gpu_mean_time(state: Dict) -> Optional[float]:
    """Extract nv/cold/time/gpu/mean from a NVBench state."""
    for summ in state.get("summaries", []):
        if summ.get("tag") == "nv/cold/time/gpu/mean":
            for d in summ.get("data", []):
                if d.get("name") == "value":
                    try:
                        return float(d["value"])
                    except (ValueError, TypeError):
                        return None
    return None


def get_axis_value(state: Dict, axis_name: str):
    """Get the numeric or string value for a given axis from a state."""
    for av in state.get("axis_values", []):
        if av.get("name") == axis_name:
            val = av.get("value")
            # Try to cast to int/float if possible
            try:
                if "." in str(val):
                    return float(val)
                else:
                    return int(val)
            except Exception:
                return val
    return None


def infer_axis_name(bench: Dict, axis_candidates: List[str]) -> Optional[str]:
    """Pick a reasonable axis name from bench['axes'] using preferred candidates."""
    axes = bench.get("axes", [])
    if not axes:
        return None
    names = [ax.get("name") for ax in axes]

    # Prefer explicit candidates (e.g., "n", "sparsity")
    for cand in axis_candidates:
        if cand in names:
            return cand

    # Fallback: just use the first axis name
    return names[0]


def collect_scaling(
    data: Dict,
    name_substring: str,
    axis_candidates: List[str],
) -> Tuple[Optional[str], List, List[str], List[List[float]]]:
    """
    Collect scaling data for one operation type from an NVBench JSON.

    Returns:
        axis_name: the axis used on x (e.g., 'n', 'sparsity')
        x_vals: sorted list of x positions (matrix sizes or sparsity levels)
        variant_names: list of benchmark names (treated as different optimizations)
        times: times[v][i] = GPU time for variant v at x_vals[i]
    """
    variants: Dict[str, Dict] = {}
    axis_name: Optional[str] = None

    for bench in data.get("benchmarks", []):
        bench_name = bench.get("name", "")
        if name_substring not in bench_name:
            continue

        if axis_name is None:
            axis_name = infer_axis_name(bench, axis_candidates)

        vname = bench_name  # treat each benchmark as a different optimization variant
        vdict = variants.setdefault(vname, {})

        for state in bench.get("states", []):
            if state.get("is_skipped"):
                continue

            if axis_name is not None:
                x_val = get_axis_value(state, axis_name)
            else:
                # fallback to state index or name if no axis
                x_val = state.get("name")

            if x_val is None:
                continue

            gpu_time = get_gpu_mean_time(state)
            if gpu_time is None:
                continue

            vdict[x_val] = gpu_time

    if not variants:
        return None, [], [], []

    # Unified and sorted x-axis
    all_x_vals = sorted(
        {x for vdict in variants.values() for x in vdict.keys()},
        key=lambda v: (isinstance(v, str), v),
    )
    variant_names = sorted(variants.keys())

    times: List[List[float]] = []
    for vname in variant_names:
        vdict = variants[vname]
        times.append([float(vdict.get(x, 0.0)) for x in all_x_vals])

    return axis_name, all_x_vals, variant_names, times


def plot_stacked_bar(
    x_vals: List,
    variant_names: List[str],
    times_s: List[List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    outfile: str,
    time_unit: str = "ms",
) -> None:
    """Create a stacked bar plot with variants stacked on top of each other."""
    if not x_vals or not variant_names:
        return

    # Convert seconds to milliseconds for readability
    times = np.array(times_s) * 1e3  # shape: (num_variants, num_x)

    x = np.arange(len(x_vals))
    bottoms = np.zeros_like(x, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, vname in enumerate(variant_names):
        ax.bar(
            x,
            times[idx],
            bottom=bottoms,
            label=vname,
        )
        bottoms += times[idx]

    # Pretty x tick labels
    def fmt_x(v):
        if isinstance(v, (int, float)):
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return f"{v}"
        return str(v)

    ax.set_xticks(x)
    ax.set_xticklabels([fmt_x(v) for v in x_vals], rotation=45, ha="right")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"{ylabel} ({time_unit})")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    fig.savefig(outfile)
    plt.close(fig)


def make_gpu_plots(
    project_gpu_json: str = "./logs/project_gpu.json",
    nn_gpu_json: str = "./logs/nn_gpu.json",
    output_dir: str = "./plots",
) -> None:
    """
    Generate GPU stacked bar plots for MM/MV (and SpMM/SpMV if present)
    from NVBench JSON logs.
    """

    # 1) MM (GEMM) – microbenchmark scaling from project_gpu.json
    if os.path.exists(project_gpu_json):
        data = load_nvbench_json(project_gpu_json)

        # MM: GEMM, x-axis = matrix size n
        axis_name, x_vals, variant_names, times = collect_scaling(
            data,
            name_substring="gemm",   # matches 'gemm', 'gemm_naive', etc.
            axis_candidates=["n", "size", "dim"],
        )
        if x_vals:
            plot_stacked_bar(
                x_vals,
                variant_names,
                times,
                title="GPU MM (GEMM) – variants vs matrix size",
                xlabel=f"Matrix size ({axis_name})" if axis_name else "Matrix size",
                ylabel="Kernel time",
                outfile=os.path.join(output_dir, "gpu_mm_stacked.png"),
            )

        # SpMM / SpMV could also be logged in the same file (if you add them later)
        axis_name, x_vals, variant_names, times = collect_scaling(
            data,
            name_substring="spmm",
            axis_candidates=["sparsity", "density", "nnz"],
        )
        if x_vals:
            plot_stacked_bar(
                x_vals,
                variant_names,
                times,
                title="GPU SpMM – variants vs sparsity level",
                xlabel=f"Sparsity axis ({axis_name})" if axis_name else "Sparsity",
                ylabel="Kernel time",
                outfile=os.path.join(output_dir, "gpu_spmm_stacked.png"),
            )

        axis_name, x_vals, variant_names, times = collect_scaling(
            data,
            name_substring="spmv",
            axis_candidates=["sparsity", "density", "nnz"],
        )
        if x_vals:
            plot_stacked_bar(
                x_vals,
                variant_names,
                times,
                title="GPU SpMV – variants vs sparsity level",
                xlabel=f"Sparsity axis ({axis_name})" if axis_name else "Sparsity",
                ylabel="Kernel time",
                outfile=os.path.join(output_dir, "gpu_spmv_stacked.png"),
            )

    # 2) MM/MV inside NN (nn_gpu.json) – optional, but nice for NN-level plots
    if os.path.exists(nn_gpu_json):
        data = load_nvbench_json(nn_gpu_json)

        # GEMM inside NN – may not scale by n, but still generates a stacked bar
        axis_name, x_vals, variant_names, times = collect_scaling(
            data,
            name_substring="gemm",
            axis_candidates=["n", "tile1", "tile2"],
        )
        if x_vals:
            plot_stacked_bar(
                x_vals,
                variant_names,
                times,
                title="GPU Dense NN – GEMM operations",
                xlabel=f"Config ({axis_name})" if axis_name else "Config",
                ylabel="Kernel time",
                outfile=os.path.join(output_dir, "gpu_nn_gemm_stacked.png"),
            )

        # GEMV inside NN
        axis_name, x_vals, variant_names, times = collect_scaling(
            data,
            name_substring="gemv",
            axis_candidates=["n", "vecTile"],
        )
        if x_vals:
            plot_stacked_bar(
                x_vals,
                variant_names,
                times,
                title="GPU Dense NN – GEMV operations",
                xlabel=f"Config ({axis_name})" if axis_name else "Config",
                ylabel="Kernel time",
                outfile=os.path.join(output_dir, "gpu_nn_gemv_stacked.png"),
            )

        # GPU Sparse NN performance (nn_gpu.json)
        # Extract sparse NN benchmark data and plot accuracy + performance vs sparsity
        for bench in data.get("benchmarks", []):
            if "sparse_spmv" in bench["name"].lower() and "sweep" not in bench["name"].lower():
                # This is the main sparse NN benchmark with actual weights
                sparsity_levels = []
                exec_times = []
                accuracies = []
                
                for state in bench.get("states", []):
                    if state.get("is_skipped"):
                        continue
                    
                    # Extract sparsity axis value
                    sparsity = get_axis_value(state, "sparsity")
                    if sparsity is None:
                        continue
                    
                    # Extract GPU execution time
                    gpu_time = get_gpu_mean_time(state)
                    if gpu_time is None:
                        continue
                    
                    # Extract accuracy from summaries
                    accuracy = None
                    for summ in state.get("summaries", []):
                        if "accuracy" in summ.get("tag", "").lower():
                            for d in summ.get("data", []):
                                if d.get("name") == "value":
                                    try:
                                        accuracy = float(d["value"])
                                    except (ValueError, TypeError):
                                        pass
                    
                    sparsity_levels.append(int(sparsity) if isinstance(sparsity, float) else sparsity)
                    exec_times.append(gpu_time * 1e3)  # Convert to ms
                    if accuracy is not None:
                        accuracies.append(accuracy)
                
                if sparsity_levels:
                    # Sort by sparsity level
                    sorted_data = sorted(zip(sparsity_levels, exec_times, accuracies), key=lambda x: x[0])
                    if not sorted_data:
                        continue
                    sparsity_levels, exec_times, accuracies = zip(*sorted_data)
                    
                    # Plot execution time vs sparsity
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    color = "#2E86AB"
                    ax1.plot(sparsity_levels, exec_times, marker="o", linewidth=2.5, 
                            color=color, markersize=8, label="GPU Execution Time")
                    ax1.set_xlabel("Sparsity Level (%)", fontsize=12)
                    ax1.set_ylabel("GPU Execution Time (ms)", fontsize=12, color=color)
                    ax1.tick_params(axis="y", labelcolor=color)
                    
                    # Plot accuracy on secondary y-axis
                    if accuracies:
                        ax2 = ax1.twinx()
                        color2 = "#A23B72"
                        ax2.plot(sparsity_levels, accuracies, marker="s", linewidth=2.5, 
                                color=color2, markersize=8, label="Accuracy")
                        ax2.set_ylabel("Accuracy", fontsize=12, color=color2)
                        ax2.tick_params(axis="y", labelcolor=color2)
                        ax2.set_ylim([0, 1.0])
                        
                        # Combined legend
                        lines1, labels1 = ax1.get_legend_handles_labels()
                        lines2, labels2 = ax2.get_legend_handles_labels()
                        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")
                    else:
                        ax1.legend()
                    
                    ax1.set_title("GPU Sparse NN – Execution Time and Accuracy vs Sparsity", fontsize=14, fontweight="bold")
                    ax1.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "gpu_sparse_nn_performance.png"), dpi=200)
                    plt.close()
                    print(f"Saved {os.path.join(output_dir, 'gpu_sparse_nn_performance.png')}")

        # GPU Sparse kernel sweeps (spmv_sparse_sweep, spmm_sparse_sweep)
        for bench in data.get("benchmarks", []):
            bench_name = bench.get("name", "").lower()
            
            # Plot sparse kernels with sparsity sweeps
            if "spmv_sparse_sweep" in bench_name or "spmv_sweep" in bench_name:
                sparsity_levels = []
                exec_times = []
                
                for state in bench.get("states", []):
                    if state.get("is_skipped"):
                        continue
                    
                    sparsity = get_axis_value(state, "sparsity")
                    if sparsity is None:
                        continue
                    
                    gpu_time = get_gpu_mean_time(state)
                    if gpu_time is None:
                        continue
                    
                    sparsity_levels.append(int(sparsity) if isinstance(sparsity, float) else sparsity)
                    exec_times.append(gpu_time * 1e3)  # ms
                
                if sparsity_levels:
                    sorted_data = sorted(zip(sparsity_levels, exec_times), key=lambda x: x[0])
                    if not sorted_data:
                        continue
                    sparsity_levels, exec_times = zip(*sorted_data)
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(sparsity_levels, exec_times, marker="o", linewidth=2.5, markersize=8, color="#06A77D")
                    plt.xlabel("Sparsity Level (%)", fontsize=12)
                    plt.ylabel("GPU Execution Time (ms)", fontsize=12)
                    plt.title("GPU SpMV Kernel – Execution Time vs Sparsity", fontsize=14, fontweight="bold")
                    plt.grid(True, alpha=0.3)
                    for x, y in zip(sparsity_levels, exec_times):
                        plt.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "gpu_spmv_sparse_sweep.png"), dpi=200)
                    plt.close()
                    print(f"Saved {os.path.join(output_dir, 'gpu_spmv_sparse_sweep.png')}")
            
            elif "spmm_sparse_sweep" in bench_name or "spmm_sweep" in bench_name:
                sparsity_levels = []
                exec_times = []
                
                for state in bench.get("states", []):
                    if state.get("is_skipped"):
                        continue
                    
                    sparsity = get_axis_value(state, "sparsity")
                    if sparsity is None:
                        continue
                    
                    gpu_time = get_gpu_mean_time(state)
                    if gpu_time is None:
                        continue
                    
                    sparsity_levels.append(int(sparsity) if isinstance(sparsity, float) else sparsity)
                    exec_times.append(gpu_time * 1e3)  # ms
                
                if sparsity_levels:
                    sorted_data = sorted(zip(sparsity_levels, exec_times), key=lambda x: x[0])
                    if not sorted_data:
                        continue
                    sparsity_levels, exec_times = zip(*sorted_data)
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(sparsity_levels, exec_times, marker="s", linewidth=2.5, markersize=8, color="#D62828")
                    plt.xlabel("Sparsity Level (%)", fontsize=12)
                    plt.ylabel("GPU Execution Time (ms)", fontsize=12)
                    plt.title("GPU SpMM Kernel – Execution Time vs Sparsity", fontsize=14, fontweight="bold")
                    plt.grid(True, alpha=0.3)
                    for x, y in zip(sparsity_levels, exec_times):
                        plt.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, "gpu_spmm_sparse_sweep.png"), dpi=200)
                    plt.close()
                    print(f"Saved {os.path.join(output_dir, 'gpu_spmm_sparse_sweep.png')}")


if __name__ == "__main__":
    # Default usage: python script/plot_nvbench_gpu.py
    make_gpu_plots()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    print("Generating plots...")

    if os.path.exists("logs/project.json"):
        df_proj = parse_project("logs/project.json")
        plot_project(df_proj)
        
        # NEW: Parse and plot CPU sparse kernels from project.json
        print("Generating CPU sparse kernel plots...")
        df_sparse_cpu = parse_sparse_cpu("logs/project.json")
        if not df_sparse_cpu.empty:
            plot_sparse_cpu(df_sparse_cpu)

    if os.path.exists("logs/nn_cpu.json"):
        df_nn = parse_nn_cpu("logs/nn_cpu.json")
        plot_nn_cpu(df_nn)
    
    # NEW: Sparse vs Dense NN comparison
    if os.path.exists("logs/sparsity_accuracy_python.csv"):
        print("Generating sparse vs dense NN plots...")
        plot_sparse_vs_dense_nn("logs/sparsity_accuracy_python.csv")
    
# NEW: GPU plots (NVBench)
    if os.path.exists("logs/project_gpu.json") or os.path.exists("logs/nn_gpu.json"):
        print("Generating GPU plots...")
        make_gpu_plots(
            project_gpu_json="logs/project_gpu.json",
            nn_gpu_json="logs/nn_gpu.json",
            output_dir="./plots"
        )

    print("\nAll plots generated successfully!")
