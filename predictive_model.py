import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)


def cohens_d(x, y):
    x = pd.Series(x).dropna().astype(float).values
    y = pd.Series(y).dropna().astype(float).values

    if len(x) < 2 or len(y) < 2:
        return np.nan

    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)

    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return np.nan

    return (np.mean(x) - np.mean(y)) / pooled


def select_top_roi_features(train_df, group_a="CN", group_b="Dementia", top_k=20):
    rows = []

    for roi in train_df["ROI_Index"].unique():
        sub = train_df[train_df["ROI_Index"] == roi]

        a = sub[sub["Diagnosis"] == group_a]
        b = sub[sub["Diagnosis"] == group_b]

        for feature in ["DFA", "HFD", "Spectral_Slope"]:
            d = cohens_d(a[feature], b[feature])

            if np.isfinite(d):
                rows.append({
                    "ROI_Index": roi,
                    "Feature": feature,
                    "EffectSize": d,
                    "AbsEffect": abs(d),
                    "ROI_Name": sub.iloc[0]["ROI_Name"],
                    "Network": sub.iloc[0]["Network"],
                })

    effects = pd.DataFrame(rows)

    if effects.empty:
        raise ValueError("No valid ROI effects found during feature selection.")

    return effects.sort_values("AbsEffect", ascending=False).head(top_k)


def build_wide_matrix(df, selected_features):
    subject_info = (
        df[["Subject", "Diagnosis"]]
        .drop_duplicates()
        .set_index("Subject")
    )

    X = subject_info.copy()

    for _, row in selected_features.iterrows():
        roi = row["ROI_Index"]
        feature = row["Feature"]
        col_name = f"{feature}_ROI{roi}"

        sub = df[df["ROI_Index"] == roi][["Subject", feature]].copy()
        sub = sub.rename(columns={feature: col_name})
        sub = sub.set_index("Subject")

        X = X.join(sub, how="left")

    y = X["Diagnosis"].copy()
    X = X.drop(columns=["Diagnosis"])

    # Keep all subjects. Do not drop rows, because that can remove minority-class subjects.
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill missing values column-wise.
    # If an entire column is NaN, fill with 0 after mean imputation.
    X = X.fillna(X.mean(numeric_only=True))
    X = X.fillna(0.0)

    return X, y


def evaluate_roi_model(
    roi_features_path,
    output_dir,
    group_a="CN",
    group_b="Dementia",
    top_k=20,
    n_splits=3,
):
    roi_df = pd.read_csv(roi_features_path)

    roi_df = roi_df[roi_df["Diagnosis"].isin([group_a, group_b])].copy()

    subjects = roi_df[["Subject", "Diagnosis"]].drop_duplicates()
    subjects = subjects.dropna()

    X_subjects = subjects["Subject"].values
    y_subjects = subjects["Diagnosis"].map({group_a: 0, group_b: 1}).values

    class_counts = np.bincount(y_subjects)
    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        raise ValueError(
            f"Not enough samples in the smaller class. Class counts: {class_counts}"
        )

    n_splits = min(n_splits, min_class_count)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_rows = []
    selected_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_subjects, y_subjects), start=1):
        train_subjects = set(X_subjects[train_idx])
        test_subjects = set(X_subjects[test_idx])

        train_long = roi_df[roi_df["Subject"].isin(train_subjects)].copy()
        test_long = roi_df[roi_df["Subject"].isin(test_subjects)].copy()

        selected = select_top_roi_features(
            train_long,
            group_a=group_a,
            group_b=group_b,
            top_k=top_k,
        )

        selected["Fold"] = fold
        selected_rows.append(selected)

        X_train, y_train_labels = build_wide_matrix(train_long, selected)
        X_test, y_test_labels = build_wide_matrix(test_long, selected)

        y_train = y_train_labels.map({group_a: 0, group_b: 1}).values
        y_test = y_test_labels.map({group_a: 0, group_b: 1}).values

        if len(np.unique(y_train)) < 2:
            print(f"Skipping fold {fold}: only one class in training set.")
            continue

        if len(np.unique(y_test)) < 2:
            print(f"Skipping fold {fold}: only one class in test set.")
            continue

        # Align test columns to train columns.
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                random_state=42,
                max_iter=1000,
            )),
        ])

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) == 2 else np.nan

        fold_rows.append({
            "Fold": fold,
            "AUC": auc,
            "Accuracy": accuracy_score(y_test, preds),
            "BalancedAccuracy": balanced_accuracy_score(y_test, preds),
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "TopK": top_k,
            "GroupA": group_a,
            "GroupB": group_b,
            "NTrain": len(y_train),
            "NTest": len(y_test),
            "Train_CN": int(np.sum(y_train == 0)),
            "Train_Dementia": int(np.sum(y_train == 1)),
            "Test_CN": int(np.sum(y_test == 0)),
            "Test_Dementia": int(np.sum(y_test == 1)),
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_df = pd.DataFrame(fold_rows)
    selected_df = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()

    fold_df.to_csv(output_dir / "roi_model_cv_results.csv", index=False)
    selected_df.to_csv(output_dir / "roi_model_selected_features.csv", index=False)

    if not fold_df.empty:
        summary = fold_df.mean(numeric_only=True).to_frame("Mean").join(
            fold_df.std(numeric_only=True).to_frame("Std")
        )
    else:
        summary = pd.DataFrame()

    summary.to_csv(output_dir / "roi_model_summary.csv")

    return fold_df, selected_df, summary