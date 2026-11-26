from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

@dataclass
class CropBundle:
    model: VotingClassifier
    metadata: dict[str, Any]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip() for c in df.columns]
    # Accept both common target spellings.
    label_col = next((c for c in df.columns if c.lower() in {"label", "crop", "target"}), None)
    if label_col is None:
        raise ValueError(f"Could not find crop label column. Columns: {df.columns.tolist()}")
    rename = {label_col: TARGET}
    for f in FEATURES:
        match = next((c for c in df.columns if c.lower() == f.lower()), None)
        if match and match != f:
            rename[match] = f
    df = df.rename(columns=rename)
    missing = [f for f in FEATURES + [TARGET] if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    df = df[FEATURES + [TARGET]].copy()
    for f in FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna()
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()
    if df.empty:
        raise ValueError("No usable records after cleaning")
    return df


def build_model() -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=3000, random_state=42)),
    ])
    gb = GradientBoostingClassifier(random_state=42)
    return VotingClassifier(
        estimators=[("rf", rf), ("lr", lr), ("gb", gb)],
        voting="soft",
        weights=[2, 1, 2],
        flatten_transform=True,
        n_jobs=-1,
    )


def crop_profiles(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for crop, g in df.groupby(TARGET):
        item = {}
        for f in FEATURES:
            q1, med, q3 = g[f].quantile([0.25, 0.5, 0.75])
            item[f] = {"q1": float(q1), "median": float(med), "q3": float(q3)}
        profiles[crop] = item
    return profiles


def train(csv_path: str | Path, models_dir: str | Path = "models") -> dict[str, Any]:
    df = load_dataset(csv_path)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = build_model()
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    cv = cross_val_score(
        build_model(), X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=1,
    )
    metrics = {
        "samples": int(len(df)),
        "classes": sorted(y.unique().tolist()),
        "n_classes": int(y.nunique()),
        "train_accuracy": float(accuracy_score(y_train, pred_train)),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "cv_accuracy_mean": float(cv.mean()),
        "cv_accuracy_std": float(cv.std()),
        "classification_report": classification_report(y_test, pred_test, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, pred_test, labels=sorted(y.unique())).tolist(),
        "confusion_labels": sorted(y.unique()),
    }
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, Path(models_dir) / "crop_voting.joblib")
    joblib.dump({"features": FEATURES, "profiles": crop_profiles(df), "classes": sorted(y.unique().tolist())}, Path(models_dir) / "crop_metadata.joblib")
    with open(Path(models_dir) / "crop_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def load_bundle(models_dir: str | Path = "models") -> CropBundle:
    models_dir = Path(models_dir)
    model = joblib.load(models_dir / "crop_voting.joblib")
    metadata = joblib.load(models_dir / "crop_metadata.joblib")
    return CropBundle(model=model, metadata=metadata)


def predict(
    bundle: CropBundle,
    values: dict[str, float],
    top_k: int = 8,
) -> dict[str, Any]:
    """Return predicted crop, full probabilities, and top-k results."""
    missing = [f for f in FEATURES if f not in values]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = pd.DataFrame([{f: float(values[f]) for f in FEATURES}])
    probs = bundle.model.predict_proba(X)[0]
    classes = bundle.model.classes_.tolist()

    probabilities = {
        crop: float(probability)
        for crop, probability in zip(classes, probs)
    }

    order = np.argsort(probs)[::-1][:min(top_k, len(classes))]
    top_crops = [
        {
            "crop": classes[i],
            "probability": round(float(probs[i]), 6),
        }
        for i in order
    ]

    best_index = int(np.argmax(probs))
    return {
        "predicted_crop": classes[best_index],
        "probability": round(float(probs[best_index]), 6),
        "probabilities": probabilities,
        "top_crops": top_crops,
    }
