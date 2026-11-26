from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "reference" / "india_crop_catalog.json"


def load_catalog() -> list[dict[str, Any]]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Crop catalog not found: {CATALOG_PATH}")
    with CATALOG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("India crop catalog must contain a JSON list.")
    return data


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


ALIASES = {
    "grape": "grapes",
    "maize / corn": "maize",
    "chickpea / gram": "chickpea",
    "pigeon pea / arhar / tur": "pigeonpeas",
    "green gram / moong": "mungbean",
    "black gram / urad": "blackgram",
    "lentil / masur": "lentil",
    "moth bean / matki": "mothbeans",
    "kidney bean / rajma": "kidneybeans",
    "coffee – arabica": "coffee",
    "coffee – robusta": "coffee",
}


def catalog_metadata(model_crops: list[str]) -> dict[str, Any]:
    model = {normalize(x) for x in model_crops}
    entries = []
    for item in load_catalog():
        row = dict(item)
        normalized = normalize(row.get("crop", ""))
        mapped = ALIASES.get(normalized, normalized)
        row["model_supported"] = mapped in model
        row["enabled"] = True
        row["recommendation_mode"] = "ML + profile" if row["model_supported"] else "reference profile"
        entries.append(row)

    return {
        "count": len(entries),
        "unique_crops": len({normalize(x["crop"]) for x in entries}),
        "model_classes": len(model_crops),
        "categories": sorted({x["major_category"] for x in entries}),
        "crops": entries,
    }
