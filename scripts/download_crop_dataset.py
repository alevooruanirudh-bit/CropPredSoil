from __future__ import annotations
import argparse
from pathlib import Path
import shutil

# Kaggle dataset commonly used for the exact 2200 x 22-crop, 7-feature setup described by the project.
DATASET_HANDLE = "atharvaingle/crop-recommendation-dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="data/raw/crop/Crop_recommendation.csv")
    args = ap.parse_args()
    try:
        import kagglehub
    except ImportError:
        raise SystemExit("Install kagglehub first: pip install kagglehub")
    path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    matches = list(path.rglob("*.csv"))
    if not matches:
        raise SystemExit(f"No CSV found in downloaded dataset: {path}")
    src = next((p for p in matches if p.name.lower() == "crop_recommendation.csv"), matches[0])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    print(f"Copied {src} -> {out}")

if __name__ == "__main__":
    main()
