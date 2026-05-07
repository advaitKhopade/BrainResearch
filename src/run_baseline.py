from baseline import run_analysis
from paths import METADATA_DIR, CONNECTIVITY_DIR, TIMESERIES_DIR, FEATURES_DIR


def run_version(version):
    metadata_path = METADATA_DIR / f"ADNI3_variables_v{version}.csv"
    connectivity_path = CONNECTIVITY_DIR / f"corrmatrix_229x434x434_v{version}.npy"
    timeseries_path = TIMESERIES_DIR / f"timeseries_229x197x434_v{version}.npy"

    output_path = FEATURES_DIR / f"results_v{version}.csv"

    run_analysis(
        meta_path=metadata_path,
        conn_path=connectivity_path,
        ts_path=timeseries_path,
        output_path=output_path,
        compute_sliding_dfa=True,
    )


def main():
    for version in [1, 2, 3, 4, 5]:
        print(f"\nRunning baseline v{version}")
        run_version(version)


if __name__ == "__main__":
    main()