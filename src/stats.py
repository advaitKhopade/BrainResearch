import numpy as np
import pandas as pd
from pathlib import Path
from math import erf, sqrt


def _normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def cohens_d(x, y):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)

    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled


def welch_t_stat(x, y):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    nx, ny = len(x), len(y)

    denom = np.sqrt(vx / nx + vy / ny)
    if denom == 0:
        return np.nan

    return (mx - my) / denom


def welch_df(x, y):
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    if len(x) < 2 or len(y) < 2:
        return np.nan

    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    nx, ny = len(x), len(y)

    a = vx / nx
    b = vy / ny
    denom = (a * a) / (nx - 1) + (b * b) / (ny - 1)

    if denom == 0:
        return np.nan

    return (a + b) ** 2 / denom


def welch_p_value_approx(x, y):
    """
    Normal-approximation two-sided p-value from Welch t-statistic.
    Good enough for phase testing without SciPy.
    """
    t = welch_t_stat(x, y)
    if not np.isfinite(t):
        return np.nan
    return 2.0 * (1.0 - _normal_cdf(abs(t)))


def mann_whitney_u_test(x, y):
    """
    Rank-sum style Mann-Whitney U with normal approximation.
    Returns (U, p_value).
    """
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    y = np.asarray(pd.Series(y).dropna(), dtype=float)

    nx, ny = len(x), len(y)
    if nx < 1 or ny < 1:
        return np.nan, np.nan

    combined = np.concatenate([x, y])
    ranks = pd.Series(combined).rank(method="average").to_numpy()

    rx = ranks[:nx].sum()
    u1 = rx - nx * (nx + 1) / 2.0
    u2 = nx * ny - u1
    u = min(u1, u2)

    mu = nx * ny / 2.0
    sigma = np.sqrt(nx * ny * (nx + ny + 1) / 12.0)
    if sigma == 0:
        return u, np.nan

    z = (u - mu) / sigma
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return float(u), float(p)


def fdr_bh(pvals):
    """
    Benjamini-Hochberg FDR correction.
    """
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)

    mask = np.isfinite(pvals)
    if mask.sum() == 0:
        return out

    p = pvals[mask]
    m = len(p)

    order = np.argsort(p)
    ranked = p[order]

    adj = ranked * m / (np.arange(1, m + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)

    restored = np.empty_like(adj)
    restored[order] = adj
    out[mask] = restored
    return out


def summary_by_group(df, feature, group_col="Diagnosis"):
    rows = []

    for group, sub in df.groupby(group_col):
        vals = pd.to_numeric(sub[feature], errors="coerce").dropna()
        rows.append({
            "group": group,
            "n": len(vals),
            "mean": vals.mean(),
            "std": vals.std(ddof=1),
            "median": vals.median(),
        })

    return pd.DataFrame(rows)


def pairwise_effects(df, feature, group_col="Diagnosis"):
    groups = list(df[group_col].dropna().unique())
    rows = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g1, g2 = groups[i], groups[j]
            x = pd.to_numeric(df[df[group_col] == g1][feature], errors="coerce").dropna()
            y = pd.to_numeric(df[df[group_col] == g2][feature], errors="coerce").dropna()

            u_stat, mw_p = mann_whitney_u_test(x, y)

            rows.append({
                "feature": feature,
                "group1": g1,
                "group2": g2,
                "cohens_d": cohens_d(x, y),
                "welch_t_stat": welch_t_stat(x, y),
                "welch_df": welch_df(x, y),
                "welch_p_approx": welch_p_value_approx(x, y),
                "mannwhitney_u": u_stat,
                "mannwhitney_p": mw_p,
                "mean_group1": x.mean() if len(x) else np.nan,
                "mean_group2": y.mean() if len(y) else np.nan,
                "median_group1": x.median() if len(x) else np.nan,
                "median_group2": y.median() if len(y) else np.nan,
                "n_group1": len(x),
                "n_group2": len(y),
            })

    return pd.DataFrame(rows)


def analyze_feature_table(csv_path, out_dir):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)

    df = pd.read_csv(csv_path)

    features = [
        "Fractal_Dim",
        "Graph_Density",
        "Graph_MeanDegree",
        "Graph_StdDegree",
        "Graph_Entropy",
        "Graph_ThresholdScaling",
        "Spectral_Slope",
        "DFA",
        "HFD",
        "Hvar",
        "Hrange",
        "Htrend",
    ]

    summary_frames = []
    effect_frames = []

    for feature in features:
        if feature not in df.columns:
            continue

        summary = summary_by_group(df, feature)
        summary["feature"] = feature
        summary_frames.append(summary)

        effects = pairwise_effects(df, feature)
        effect_frames.append(effects)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    effects_df = pd.concat(effect_frames, ignore_index=True) if effect_frames else pd.DataFrame()

    if not effects_df.empty:
        effects_df["fdr_welch_p"] = np.nan
        effects_df["fdr_mannwhitney_p"] = np.nan

        for (g1, g2), idx in effects_df.groupby(["group1", "group2"]).groups.items():
            idx = list(idx)
            effects_df.loc[idx, "fdr_welch_p"] = fdr_bh(effects_df.loc[idx, "welch_p_approx"].values)
            effects_df.loc[idx, "fdr_mannwhitney_p"] = fdr_bh(effects_df.loc[idx, "mannwhitney_p"].values)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "feature_group_summary.csv", index=False)
    effects_df.to_csv(out_dir / "feature_pairwise_effects.csv", index=False)

    # convenience ranked tables
    if not effects_df.empty:
        ranked = effects_df.reindex(
            effects_df["cohens_d"].abs().sort_values(ascending=False).index
        )
        ranked.to_csv(out_dir / "feature_ranked_by_effect_size.csv", index=False)

    return summary_df, effects_df