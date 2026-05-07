import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from loaders import load_metadata, load_connectivity, load_timeseries, validate_shapes
from boxcount import boxcount_fd_matrix, graph_features, multi_threshold_graph_scaling
from scaling import (
    subject_mean_spectral_slope,
    subject_mean_dfa,
    subject_mean_hfd,
    subject_mean_sliding_window_dfa,
)


def generate_plots(df, plot_dir):
    plot_dir.mkdir(parents=True, exist_ok=True)

    if "Diagnosis" not in df.columns:
        print("Skipping plots: 'Diagnosis' column not found.")
        return

    plots = [
        ("Fractal_Dim", "Fractal Dimension"),
        ("Spectral_Slope", "Spectral Slope"),
        ("DFA", "DFA Scaling"),
        ("HFD", "Higuchi Fractal Dimension"),
        ("Hvar", "Sliding Window DFA Variance"),
        ("Hrange", "Sliding Window DFA Range"),
    ]

    diagnosis_order = ["CN", "MCI", "Dementia"]

    for col, title in plots:
        if col not in df.columns:
            print(f"Skipping plot for {col}: column not found.")
            continue

        groups = []
        labels = []

        for label in diagnosis_order:
            group = df[df["Diagnosis"] == label]
            vals = group[col].dropna().values

            if len(vals) > 0:
                groups.append(vals)
                labels.append(label)

        if len(groups) == 0:
            print(f"Skipping plot for {col}: no valid data.")
            continue

        plt.figure(figsize=(6, 4))
        plt.boxplot(groups, labels=labels)
        plt.title(title)
        plt.xlabel("Diagnosis")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{col}.png")
        plt.close()

def run_analysis(meta_path, conn_path, ts_path, output_path, compute_sliding_dfa=True):
    meta = load_metadata(meta_path)
    conn = load_connectivity(conn_path)
    ts = load_timeseries(ts_path)

    validate_shapes(meta, conn, ts)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []

    for i in range(len(meta)):
        if (i + 1) % 25 == 0 or i == len(meta) - 1:
            print(f"Processing {i+1}/{len(meta)}")

        row = {}
        row["Subject"] = meta.iloc[i].get("Subject", i)
        row["Diagnosis"] = meta.iloc[i].get("Diagnosis_Name", None)
        row["ResearchGroup"] = meta.iloc[i].get("ResearchGroup", None)
        row["Age"] = meta.iloc[i].get("Age", None)
        row["Sex"] = meta.iloc[i].get("Sex", None)
        row["FDMean_motion"] = meta.iloc[i].get("FDMean", None)
        row["FDMax_motion"] = meta.iloc[i].get("FDMax", None)
        row["FDPerc_motion"] = meta.iloc[i].get("FDPerc", None)

        fd = boxcount_fd_matrix(conn[i], threshold=0.30)
        graph = graph_features(conn[i], threshold=0.30)
        graph_scaling = multi_threshold_graph_scaling(conn[i])

        slope = subject_mean_spectral_slope(ts[i])
        dfa_val = subject_mean_dfa(ts[i])
        hfd_val = subject_mean_hfd(ts[i], kmax=10)

        if compute_sliding_dfa:
            hvar, hrange, htrend = subject_mean_sliding_window_dfa(ts[i])
        else:
            hvar, hrange, htrend = float("nan"), float("nan"), float("nan")

        row["Fractal_Dim"] = fd
        row["Graph_Density"] = graph["graph_density"]
        row["Graph_MeanDegree"] = graph["mean_degree"]
        row["Graph_StdDegree"] = graph["std_degree"]
        row["Graph_Entropy"] = graph["degree_entropy"]
        row["Graph_ThresholdScaling"] = graph_scaling

        row["Spectral_Slope"] = slope
        row["DFA"] = dfa_val
        row["HFD"] = hfd_val
        row["Hvar"] = hvar
        row["Hrange"] = hrange
        row["Htrend"] = htrend

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    plot_dir = output_path.parents[1] / "plots"
    generate_plots(df, plot_dir)

    print(f"\nSaved results to: {output_path}")

    if "Diagnosis" in df.columns:
        group_means = df.groupby("Diagnosis").mean(numeric_only=True)
        print("\nGroup Means:")
        print(group_means)

        group_means_name = output_path.name.replace("results_", "group_means_")
        group_means_path = output_path.parent / group_means_name
        group_means.to_csv(group_means_path)
        print(f"Saved group means to: {group_means_path}")