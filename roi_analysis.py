import numpy as np
import pandas as pd
from pathlib import Path

from loaders import load_metadata, load_timeseries
from scaling import (
    dfa_alpha_1d,
    higuchi_fd_1d,
    spectral_slope_1d,
)


def load_atlas(path, expected_rois=434):
    df = pd.read_csv(path, sep="\t", header=None, names=["ROI_ID", "ROI_Name"])
    df = df.dropna(how="all").reset_index(drop=True)

    if len(df) != expected_rois:
        raise ValueError(
            f"Atlas has {len(df)} rows, expected {expected_rois}"
        )

    return df


def infer_network(roi_name):
    if pd.isna(roi_name):
        return "Unknown"

    name = str(roi_name)

    # Schaefer-style names: Schaefer2018_17Networks_LH_DefaultB_PFCv_1
    if "17Networks_" in name:
        parts = name.split("_")
        idx = parts.index("17Networks") if "17Networks" in parts else None
        if idx is not None and idx + 2 < len(parts):
            return parts[idx + 2]

    # Fallback for Buckner labels
    if "Buckner2011_17Networks_" in name:
        return "Buckner"

    # Fallback for FreeSurfer-style subcortical labels if present
    if "FreeSurfer" in name or "Subcortical" in name:
        return "Subcortical"

    return "Unknown"


def compute_roi_features(ts_subject):
    R = ts_subject.shape[1]

    dfa_vals = []
    hfd_vals = []
    slope_vals = []

    for r in range(R):
        signal = ts_subject[:, r]
        dfa_vals.append(dfa_alpha_1d(signal))
        hfd_vals.append(higuchi_fd_1d(signal))
        slope_vals.append(spectral_slope_1d(signal))

    return {
        "DFA": np.array(dfa_vals, dtype=float),
        "HFD": np.array(hfd_vals, dtype=float),
        "Spectral_Slope": np.array(slope_vals, dtype=float),
    }


def run_roi_analysis(meta_path, ts_path, atlas_path, output_path):
    meta = load_metadata(meta_path)
    ts = load_timeseries(ts_path)

    expected_rois = ts.shape[2]
    atlas = load_atlas(atlas_path, expected_rois=expected_rois)

    print("Timeseries shape:", ts.shape)
    print("Atlas shape:", atlas.shape)
    print(atlas.head())

    if len(meta) != ts.shape[0]:
        raise ValueError(
            f"Subject mismatch: metadata={len(meta)}, timeseries={ts.shape[0]}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for i in range(len(meta)):
        if (i + 1) % 10 == 0 or i == len(meta) - 1:
            print(f"ROI Processing {i+1}/{len(meta)}")

        subject_ts = ts[i]
        feats = compute_roi_features(subject_ts)

        for r in range(expected_rois):
            roi_name = atlas.iloc[r]["ROI_Name"]

            row = {
                "Subject": meta.iloc[i].get("Subject", i),
                "Diagnosis": meta.iloc[i].get("Diagnosis_Name", None),
                "ResearchGroup": meta.iloc[i].get("ResearchGroup", None),
                "Age": meta.iloc[i].get("Age", None),
                "Sex": meta.iloc[i].get("Sex", None),
                "ROI_Index": r,
                "ROI_ID": atlas.iloc[r]["ROI_ID"],
                "ROI_Name": roi_name,
                "Network": infer_network(roi_name),
                "DFA": feats["DFA"][r],
                "HFD": feats["HFD"][r],
                "Spectral_Slope": feats["Spectral_Slope"][r],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"\nSaved ROI features to: {output_path}")