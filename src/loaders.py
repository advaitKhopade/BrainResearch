import numpy as np
import pandas as pd


def load_metadata(path):
    df = pd.read_csv(path)

    if "Diagnosis_Name" not in df.columns and "Diagnosis" in df.columns:
        df["Diagnosis_Name"] = df["Diagnosis"].map({1: "CN", 2: "MCI", 3: "Dementia"})

    return df


def load_connectivity(path):
    return np.load(path, mmap_mode="r")


def load_timeseries(path):
    return np.load(path, mmap_mode="r")


def validate_shapes(meta, conn, ts):
    if len(meta) != conn.shape[0] or len(meta) != ts.shape[0]:
        raise ValueError(
            f"Mismatch in subject counts: meta={len(meta)}, conn={conn.shape[0]}, ts={ts.shape[0]}"
        )

    if conn.ndim != 3 or conn.shape[1] != conn.shape[2]:
        raise ValueError(f"Connectivity must be (N, R, R), got {conn.shape}")

    if ts.ndim != 3:
        raise ValueError(f"Timeseries must be (N, T, R), got {ts.shape}")

    if conn.shape[1] != ts.shape[2]:
        raise ValueError(
            f"ROI mismatch: connectivity has {conn.shape[1]} ROIs but timeseries has {ts.shape[2]}"
        )