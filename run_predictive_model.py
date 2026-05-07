from predictive_model import evaluate_roi_model
from paths import FEATURES_DIR
import pandas as pd


def run_model(version, top_k):
    roi_features_path = FEATURES_DIR / f"roi_features_v{version}.csv"
    output_dir = FEATURES_DIR / f"predictive_v{version}_topk_{top_k}"

    fold_df, selected_df, summary = evaluate_roi_model(
        roi_features_path=roi_features_path,
        output_dir=output_dir,
        group_a="CN",
        group_b="Dementia",
        top_k=top_k,
        n_splits=3,
    )

    row = {
        "Version": version,
        "TopK": top_k,
        "Mean_AUC": fold_df["AUC"].mean(),
        "Std_AUC": fold_df["AUC"].std(),
        "Mean_Accuracy": fold_df["Accuracy"].mean(),
        "Mean_BalancedAccuracy": fold_df["BalancedAccuracy"].mean(),
        "Mean_Sensitivity": fold_df["Sensitivity"].mean(),
        "Mean_Specificity": fold_df["Specificity"].mean(),
    }

    print(f"\nSummary for v{version}, top_k={top_k}:")
    print(pd.DataFrame([row]))

    return row


def main():
    versions = [1, 2, 3, 4, 5]
    top_k_values = [5, 10, 20, 30, 50]

    rows = []

    for version in versions:
        for top_k in top_k_values:
            print(f"\nRunning predictive model v{version}, top_k={top_k}")
            row = run_model(version, top_k)
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = FEATURES_DIR / "predictive_grid_summary.csv"
    summary_df.to_csv(out_path, index=False)

    print("\nFull predictive grid summary:")
    print(summary_df)

    print(f"\nSaved grid summary to: {out_path}")


if __name__ == "__main__":
    main()