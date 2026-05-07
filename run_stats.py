from stats import analyze_feature_table
from paths import FEATURES_DIR


def main():
    for version in [1, 2, 3, 4, 5]:
        print(f"\nRunning stats v{version}")

        csv_path = FEATURES_DIR / f"results_v{version}.csv"
        out_dir = FEATURES_DIR / f"stats_v{version}"

        summary_df, effects_df = analyze_feature_table(csv_path, out_dir)

        print("\nSaved statistical summaries:")
        print(out_dir / "feature_group_summary.csv")
        print(out_dir / "feature_pairwise_effects.csv")
        print(out_dir / "feature_ranked_by_effect_size.csv")

        if not effects_df.empty:
            print("\nTop absolute effect sizes:")
            print(
                effects_df.reindex(
                    effects_df["cohens_d"].abs().sort_values(ascending=False).index
                ).head(10)
            )


if __name__ == "__main__":
    main()