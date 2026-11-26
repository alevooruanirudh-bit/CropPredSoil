from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re

import numpy as np
import pandas as pd

from .crop_model import FEATURES, CropBundle, predict as ml_predict
from .crop_model import load_bundle


@dataclass
class HybridRecommendation:
    recommended: dict[str, Any]
    top_crops: list[dict[str, Any]]
    probabilities: dict[str, float]


# These are intentionally broad, indicative profiles used only for crops that
# are present in the reference catalog but are not represented in the current
# supervised 22-class dataset. They are NOT presented as measured training data.
# Rainfall is expressed on the same scale as the training dataset (roughly
# annual mm / 8; e.g. maize ~70-100, rice ~200-260), not raw annual rainfall.
CATEGORY_PROFILES = {
    "Field Crops – Cereals": dict(N=(40,140),P=(20,60),K=(20,90),temperature=(18,32),humidity=(50,85),ph=(5.5,7.5),rainfall=(50,200)),
    "Field Crops – Millets": dict(N=(20,90),P=(10,50),K=(15,80),temperature=(22,35),humidity=(35,75),ph=(5.0,7.5),rainfall=(31,125)),
    "Field Crops – Pseudo-cereals": dict(N=(20,80),P=(15,50),K=(15,70),temperature=(12,28),humidity=(35,75),ph=(5.0,7.5),rainfall=(38,125)),
    "Field Crops – Pulses": dict(N=(10,75),P=(20,65),K=(15,75),temperature=(17,32),humidity=(40,80),ph=(5.5,8.0),rainfall=(38,138)),
    "Oilseeds & Oil Crops": dict(N=(20,110),P=(15,60),K=(20,90),temperature=(18,35),humidity=(40,80),ph=(5.5,8.0),rainfall=(38,188)),
    "Commercial & Fibre Crops": dict(N=(40,160),P=(20,70),K=(20,100),temperature=(20,35),humidity=(50,90),ph=(5.5,8.0),rainfall=(62,275)),
    "Horticulture – Fruits": dict(N=(30,140),P=(20,70),K=(30,120),temperature=(15,34),humidity=(50,90),ph=(5.0,7.8),rainfall=(75,275)),
    "Horticulture – Vegetables": dict(N=(30,130),P=(20,70),K=(30,120),temperature=(12,35),humidity=(45,90),ph=(5.5,7.5),rainfall=(50,225)),
    "Perennial & Plantation Crops": dict(N=(30,150),P=(20,70),K=(30,130),temperature=(18,34),humidity=(60,95),ph=(4.8,7.5),rainfall=(100,375)),
    "Spices & Condiments": dict(N=(20,120),P=(15,65),K=(25,110),temperature=(15,34),humidity=(50,95),ph=(5.0,7.5),rainfall=(62,375)),
    "Specialized Horticulture": dict(N=(20,110),P=(15,60),K=(20,100),temperature=(12,34),humidity=(45,95),ph=(5.0,7.8),rainfall=(38,275)),
    "Forage & Fodder Crops": dict(N=(30,140),P=(15,60),K=(20,100),temperature=(12,35),humidity=(40,90),ph=(5.0,8.0),rainfall=(38,225)),
}


REFERENCE_PROFILE_WEIGHT = 0.80


def normalize(name: str) -> str:
    s = str(name).strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    return s


def _alias(name: str) -> str:
    return {
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
        "coffee - arabica": "coffee",
        "coffee - robusta": "coffee",
    }.get(name, name)


def _range_from_profile(profile: dict[str, Any], feature: str) -> tuple[float, float]:
    value = profile.get(feature)
    if isinstance(value, dict):
        if "q1" in value and "q3" in value:
            return float(value["q1"]), float(value["q3"])
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])
    raise KeyError(feature)


def _score_dimension(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        mid = (low + high) / 2.0
        half = max((high - low) / 2.0, 1e-9)
        # 1 in the center and still strong near boundaries.
        return 1.0 - 0.25 * min(abs(value - mid) / half, 1.0)
    distance = (low - value) if value < low else (value - high)
    span = max(high - low, 1e-9)
    return max(0.0, 1.0 - 0.75 * (distance / span))


def build_catalog_profiles(catalog: list[dict[str, Any]], ml_bundle: CropBundle) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    learned = ml_bundle.metadata.get("profiles", {})

    for row in catalog:
        crop = str(row["crop"]).strip()
        canonical = _alias(normalize(crop))

        if canonical in learned:
            out[crop] = {
                "profile": learned[canonical],
                "source": "supervised_ml_profile",
                "model_supported": True,
            }
            continue

        category = row["major_category"]
        profile = CATEGORY_PROFILES.get(category)
        if profile is None:
            continue

        out[crop] = {
            "profile": profile,
            "source": "reference_category_profile",
            "model_supported": False,
        }

    return out


def score_catalog(values: dict[str, float], catalog: list[dict[str, Any]], bundle: CropBundle) -> list[dict[str, Any]]:
    profiles = build_catalog_profiles(catalog, bundle)

    model_result = ml_predict(bundle, values, top_k=len(bundle.model.classes_))
    model_probs = model_result["probabilities"]

    scored: list[dict[str, Any]] = []
    for row in catalog:
        crop = str(row["crop"]).strip()
        entry = profiles.get(crop)
        if not entry:
            continue

        dims = []
        details = []
        for feature in FEATURES:
            low, high = _range_from_profile(entry["profile"], feature)
            value = float(values[feature])
            score = _score_dimension(value, low, high)
            dims.append(score)
            status = "within" if low <= value <= high else ("low" if value < low else "high")
            details.append({
                "feature": feature,
                "value": value,
                "low": low,
                "high": high,
                "status": status,
                "score": round(score, 6),
            })

        heuristic = float(np.mean(dims)) if dims else 0.0
        canonical = _alias(normalize(crop))
        ml_probability = float(model_probs.get(canonical, 0.0))

        # Blend the measured-data model signal only for its trained classes.
        # Reference-only crops are ranked from their broad catalog category
        # profile, discounted because those wide, indicative ranges are far
        # easier to "fit" than the narrow learned profiles and carry no
        # measured evidence. Without this, generic profiles outrank crops the
        # model is confident about.
        if entry["model_supported"]:
            final_score = 0.60 * heuristic + 0.40 * ml_probability
            source = "hybrid_ml_and_profile"
        else:
            final_score = REFERENCE_PROFILE_WEIGHT * heuristic
            source = "reference_profile_suitability"

        scored.append({
            "crop": crop,
            "score": float(final_score),
            "probability": float(final_score),
            "heuristic_score": heuristic,
            "ml_probability": ml_probability,
            "model_supported": bool(entry["model_supported"]),
            "source": source,
            "major_category": row["major_category"],
            "crop_group": row["crop_group"],
            "details": details,
        })

    # "probability" is the crop's own 0-1 match score. It used to be each
    # crop's share of the total across all 182 crops, which made every value
    # tiny (~0.5%) and meaningless as a confidence figure.
    for x in scored:
        x["probability"] = min(max(x["score"], 0.0), 1.0)

    scored.sort(key=lambda x: x["probability"], reverse=True)
    return scored


def recommend(values: dict[str, float], catalog: list[dict[str, Any]], bundle: CropBundle, top_k: int = 8) -> HybridRecommendation:
    scored = score_catalog(values, catalog, bundle)
    top = scored[:top_k]
    best = top[0] if top else {"crop": "", "probability": 0.0}
    probs = {x["crop"]: float(x["probability"]) for x in scored}
    return HybridRecommendation(
        recommended=best,
        top_crops=top,
        probabilities=probs,
    )
