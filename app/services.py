from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import json
import tempfile

import numpy as np

from ml.crop_model import FEATURES, CropBundle, load_bundle
from ml.hybrid_recommender import recommend
from ml.suitability import feature_gap, agricultural_suitability, target_suitability, remedies
from ml.explainability import explain_prediction
from .crop_catalog import load_catalog


class AgricultureService:
    """Crop + soil decision-support service.

    The current supervised model is retained for the original 22 classes.
    The complete 182-entry reference catalog is enabled through a transparent
    category-profile suitability layer for crops not represented in that model.
    """

    def __init__(self, models_dir: str | Path = "models"):
        self.models_dir = Path(models_dir)
        self.crop: CropBundle = load_bundle(self.models_dir)
        self.catalog = load_catalog()
        self.soil = None
        self.soil_available = False

        try:
            from ml.soil_model import load
            soil_path = self.models_dir / "soil_mobilenetv2.keras"
            if soil_path.exists():
                self.soil = load(self.models_dir)
                self.soil_available = True
        except Exception:
            self.soil = None
            self.soil_available = False

    @staticmethod
    def validate_inputs(values: dict[str, Any]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for feature in FEATURES:
            if feature not in values:
                raise ValueError(f"Missing field: {feature}")
            try:
                value = float(values[feature])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value for {feature}") from exc
            if not np.isfinite(value):
                raise ValueError(f"Value for {feature} must be finite")
            cleaned[feature] = value
        return cleaned

    def available_crops(self) -> list[str]:
        # All reference crops are enabled in the UI.
        return [str(row["crop"]).strip() for row in self.catalog]

    def model_crops(self) -> list[str]:
        return sorted(self.crop.metadata.get("classes", []))

    def metrics(self) -> dict[str, Any]:
        path = self.models_dir / "crop_metrics.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def catalog_entry(self, crop_name: str) -> Optional[dict[str, Any]]:
        wanted = crop_name.strip().lower()
        for row in self.catalog:
            if str(row["crop"]).strip().lower() == wanted:
                return row
        return None

    def analyze(self, values: dict[str, Any], target_crop: Optional[str] = None, soil_image=None) -> dict[str, Any]:
        values = self.validate_inputs(values)

        hybrid = recommend(values, self.catalog, self.crop, top_k=8)
        top = hybrid.top_crops
        predicted = hybrid.recommended

        model_ranked = []
        # Build a model-only ranking for the existing suitability layer.
        from ml.crop_model import predict as ml_predict
        model_prediction = ml_predict(self.crop, values, top_k=8)
        model_ranked = model_prediction["top_crops"]

        overall = agricultural_suitability(
            values=values,
            ranked=model_ranked,
            bundle=self.crop,
        )

        selected_name = str(target_crop).strip() if target_crop else str(predicted.get("crop", "")).strip()
        selected_entry = self.catalog_entry(selected_name)
        if selected_entry is None:
            raise ValueError(f"Unknown crop '{selected_name}'")

        # For original 22 classes use the learned crop profile; for all other
        # reference crops use the transparent category-profile layer.
        model_class_names = {str(x).lower() for x in self.model_crops()}
        canonical = selected_name.lower()
        aliases = {
            "grape": "grapes",
            "maize / corn": "maize",
            "chickpea / gram": "chickpea",
            "pigeon pea / arhar / tur": "pigeonpeas",
            "green gram / moong": "mungbean",
            "black gram / urad": "blackgram",
            "lentil / masur": "lentil",
            "moth bean / matki": "mothbeans",
            "kidney bean / rajma": "kidneybeans",
            "orange / sweet orange": "orange",
            "coffee – arabica": "coffee",
            "coffee – robusta": "coffee",
        }
        canonical = aliases.get(canonical, canonical)
        is_model_supported = canonical in model_class_names

        selected_row = next((x for x in top if x["crop"].lower() == selected_name.lower()), None)
        all_scored = hybrid.probabilities
        selected_probability = float(all_scored.get(selected_name, 0.0))

        if is_model_supported:
            gaps = feature_gap(canonical, values, self.crop)
            selected = target_suitability(canonical, model_ranked, gaps)
            selected["display_crop"] = selected_name
            selected["details"] = gaps
            selected["source"] = "supervised_model_and_learned_profile"
            selected["model_supported"] = True
        else:
            # Use the hybrid profile result and expose the complete feature checks.
            scored = next((x for x in self._all_scored(values) if x["crop"].lower() == selected_name.lower()), None)
            if scored is None:
                raise ValueError(f"Unable to evaluate crop '{selected_name}'")
            major = [d for d in scored["details"] if d["status"] != "within"]
            # Suitability uses the raw profile fit, not the ranking score
            # (which is discounted for reference-only crops).
            fit = float(scored["heuristic_score"])
            selected = {
                "crop": selected_name,
                "display_crop": selected_name,
                "suitable": fit >= 0.65,
                "probability": float(scored["probability"]),
                "status": (
                    "Suitable under the reference-profile check"
                    if fit >= 0.65
                    else "Outside the preferred reference-profile range"
                ),
                "major_gaps": major[:5],
                "details": scored["details"],
                "method": (
                    "Transparent indicative profile scoring derived from the crop "
                    "reference category; this is not a separately trained ML class."
                ),
                "source": "reference_category_profile",
                "model_supported": False,
            }

        remedy_gaps = [x for x in selected["details"] if x["status"] != "within"]
        remedy_list = remedies(self._convert_reference_gaps(remedy_gaps)) if not is_model_supported else remedies(remedy_gaps)

        reasons = []
        for g in selected["details"]:
            status = g.get("status")
            if status == "within":
                continue
            name = g.get("name", g.get("feature", "feature"))
            reasons.append({
                "feature": g.get("feature"),
                "name": name,
                "status": status,
                "severity": float(g.get("severity", 1.0)),
                "reason": f"{name} is outside the preferred range for {selected_name}.",
            })
        reasons.sort(key=lambda x: x["severity"], reverse=True)

        response = {
            "inputs": values,
            "prediction": {
                "crop": predicted.get("crop", ""),
                "probability": round(float(predicted.get("probability", 0.0)), 6),
                "source": predicted.get("source", "reference_profile_suitability"),
            },
            "top_crops": [
                {
                    "crop": x["crop"],
                    "probability": round(float(x["probability"]), 6),
                    "source": x["source"],
                    "model_supported": x["model_supported"],
                    "major_category": x["major_category"],
                }
                for x in top
            ],
            "agriculture_suitability": overall,
            "selected_crop": selected,
            "reasons": reasons[:8],
            "remedies": remedy_list[:8],
            "explanation": explain_prediction(self.crop, values) if model_ranked else [],
            "soil_image_analysis": None,
            "soil_model_available": self.soil_available,
        }

        if soil_image is not None:
            if not self.soil_available or self.soil is None:
                response["soil_image_analysis"] = {
                    "available": False,
                    "message": "The soil image model is not trained yet.",
                }
            else:
                suffix = Path(soil_image.filename or "soil.jpg").suffix.lower()
                if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                    suffix = ".jpg"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    soil_image.save(tmp.name)
                    tmp_path = Path(tmp.name)
                try:
                    from ml.soil_model import predict
                    response["soil_image_analysis"] = predict(self.soil, str(tmp_path))
                finally:
                    tmp_path.unlink(missing_ok=True)

        return response

    def _all_scored(self, values: dict[str, float]):
        from ml.hybrid_recommender import score_catalog
        return score_catalog(values, self.catalog, self.crop)

    @staticmethod
    def _convert_reference_gaps(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Adapt hybrid profile detail rows to the existing remedies() contract.
        names = {
            "N": "Nitrogen (N)",
            "P": "Phosphorus (P)",
            "K": "Potassium (K)",
            "temperature": "Temperature",
            "humidity": "Humidity",
            "ph": "Soil pH",
            "rainfall": "Rainfall",
        }
        out = []
        for g in gaps:
            feature = g["feature"]
            out.append({
                "feature": feature,
                "name": names.get(feature, feature),
                "value": g["value"],
                "crop_q1": g["low"],
                "crop_q3": g["high"],
                "crop_median": (g["low"] + g["high"]) / 2,
                "q1": g["low"],
                "q3": g["high"],
                "status": g["status"],
                "severity": max(0.01, 1.0 - g["score"]),
                "adjustable": feature in {"N", "P", "K", "ph"},
            })
        return out
