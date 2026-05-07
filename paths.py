from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"

METADATA_DIR = RAW_DIR / "metadata"
CONNECTIVITY_DIR = RAW_DIR / "connectivity"
TIMESERIES_DIR = RAW_DIR / "timeseries"
ATLAS_DIR = RAW_DIR / "atlas"

FEATURES_DIR = PROCESSED_DIR / "features"
PLOTS_DIR = PROCESSED_DIR / "plots"