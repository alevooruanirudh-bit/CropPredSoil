from pathlib import Path
from ml.crop_model import train

if __name__ == "__main__":
    csv = Path("data/raw/crop/Crop_recommendation.csv")
    if not csv.exists():
        raise SystemExit("Crop dataset not found. Run scripts/download_crop_dataset.py or place Crop_recommendation.csv in data/raw/crop/.")
    metrics = train(csv)
    print(f"Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Test accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"CV accuracy:    {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
