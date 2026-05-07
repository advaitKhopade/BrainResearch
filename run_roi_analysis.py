from roi_analysis import run_roi_analysis
from paths import METADATA_DIR, TIMESERIES_DIR, ATLAS_DIR, FEATURES_DIR


def run_version(version):
    metadata_path = METADATA_DIR / f"ADNI3_variables_v{version}.csv"
    timeseries_path = TIMESERIES_DIR / f"timeseries_229x197x434_v{version}.npy"
    atlas_path = ATLAS_DIR / "atlas-Schaefer2018Combined_dseg.tsv"

    output_path = FEATURES_DIR / f"roi_features_v{version}.csv"

    run_roi_analysis(
        meta_path=metadata_path,
        ts_path=timeseries_path,
        atlas_path=atlas_path,
        output_path=output_path,
    )


def main():
    for version in [1, 2, 3, 4, 5]:
        print(f"\nRunning ROI analysis v{version}")
        run_version(version)


if __name__ == "__main__":
    main()