from roi_stats import compute_roi_effects
from paths import FEATURES_DIR
import pandas as pd


def run_version(version):
    path = FEATURES_DIR / f"roi_features_v{version}.csv"
    df = pd.read_csv(path)

    results = compute_roi_effects(df)

    out_path = FEATURES_DIR / f"roi_effects_v{version}.csv"
    results.to_csv(out_path, index=False)

    print(f"\nSaved ROI effects to: {out_path}")

    top = results.reindex(
        results["EffectSize"].abs().sort_values(ascending=False).index
    ).head(20)

    print("\nTop ROI effects:")
    print(top)

    network_summary = (
        results.groupby(["Network", "Feature"])["EffectSize"]
        .mean()
        .reset_index()
    )

    network_out = FEATURES_DIR / f"network_effects_v{version}.csv"
    network_summary.to_csv(network_out, index=False)

    print("\nNetwork-level effects:")
    print(network_summary.sort_values("EffectSize", key=lambda x: abs(x), ascending=False))

    print(f"\nSaved network effects to: {network_out}")


def main():
    for version in [1, 2, 3, 4, 5]:
        print(f"\nRunning ROI stats v{version}")
        run_version(version)


if __name__ == "__main__":
    main()