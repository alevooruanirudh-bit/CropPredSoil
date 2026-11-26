from __future__ import annotations
import numpy as np
import pandas as pd
from .crop_model import FEATURES, CropBundle


def explain_prediction(bundle: CropBundle, values: dict[str, float], background_rows: pd.DataFrame | None = None) -> list[dict[str, float | str]]:
    try:
        import shap
        x = pd.DataFrame([{f: float(values[f]) for f in FEATURES}])
        if background_rows is None:
            rng = np.random.default_rng(42)
            base = pd.DataFrame([bundle.metadata["profiles"][c][f]["median"] for c in bundle.metadata["classes"] for f in []])
            # Build one representative median vector per class as a tiny background set.
            rows = []
            for c in bundle.metadata["classes"]:
                rows.append([bundle.metadata["profiles"][c][f]["median"] for f in FEATURES])
            background_rows = pd.DataFrame(rows, columns=FEATURES)
            if len(background_rows) > 20:
                background_rows = background_rows.sample(20, random_state=42)
        explainer = shap.Explainer(bundle.model.predict_proba, background_rows)
        exp = explainer(x, max_evals=200)
        vals = np.asarray(exp.values)
        # For multiclass SHAP, summarize the magnitude across outputs.
        if vals.ndim == 3:
            vals = np.abs(vals).mean(axis=2)
        vals = vals.reshape(-1)
        order = np.argsort(np.abs(vals))[::-1]
        return [{"feature": FEATURES[i], "impact": float(vals[i]), "importance": float(abs(vals[i]))} for i in order]
    except Exception as exc:
        # Explainability is optional at inference; returning a structured fallback keeps the API usable.
        return [{"feature": "unavailable", "impact": 0.0, "importance": 0.0, "note": str(exc)}]
