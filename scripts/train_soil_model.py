from pathlib import Path
from ml.soil_model import train

if __name__ == "__main__":
    soil = Path("data/raw/soil")
    if not soil.exists():
        raise SystemExit("Create data/raw/soil with Alluvial, Black, Clay and Red folders and place the project's soil images there.")
    metrics = train(soil)
    print(metrics)
