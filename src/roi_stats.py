import pandas as pd
import numpy as np


def cohens_d(x, y):
    x = x.dropna().values
    y = y.dropna().values

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)

    pooled = np.sqrt(((nx - 1)*vx + (ny - 1)*vy) / (nx + ny - 2))
    if pooled == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled


def compute_roi_effects(df):
    results = []

    for roi in df["ROI_Index"].unique():
        sub = df[df["ROI_Index"] == roi]

        cn = sub[sub["Diagnosis"] == "CN"]
        ad = sub[sub["Diagnosis"] == "Dementia"]

        for feature in ["DFA", "HFD", "Spectral_Slope"]:
            d = cohens_d(cn[feature], ad[feature])

            if np.isnan(d):
                continue  # skip invalid

            results.append({
                "ROI": roi,
                "Feature": feature,
                "EffectSize": d,
                "ROI_Name": sub.iloc[0]["ROI_Name"],
                "Network": sub.iloc[0]["Network"]
            })

    return pd.DataFrame(results)