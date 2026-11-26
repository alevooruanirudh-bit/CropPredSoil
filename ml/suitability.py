from __future__ import annotations

from typing import Any
import numpy as np
from .crop_model import FEATURES, CropBundle

SOIL_FEATURES = {"N", "P", "K", "ph"}
CLIMATE_FEATURES = {"temperature", "humidity", "rainfall"}
DISPLAY = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature", "humidity": "Humidity", "ph": "Soil pH", "rainfall": "Rainfall"
}


def feature_gap(crop: str, values: dict[str, float], bundle: CropBundle) -> list[dict[str, Any]]:
    prof = bundle.metadata["profiles"].get(crop)
    if not prof:
        raise ValueError(f"Unknown crop: {crop}")
    rows = []
    for f in FEATURES:
        v = float(values[f])
        p = prof[f]
        if v < p["q1"]:
            direction = "low"
            severity = (p["q1"] - v) / max(abs(p["median"] - p["q1"]), 1e-9)
        elif v > p["q3"]:
            direction = "high"
            severity = (v - p["q3"]) / max(abs(p["q3"] - p["median"]), 1e-9)
        else:
            direction = "within"
            severity = 0.0
        rows.append({
            "feature": f,
            "name": DISPLAY[f],
            "value": v,
            "crop_q1": float(p["q1"]),
            "crop_median": float(p["median"]),
            "crop_q3": float(p["q3"]),
            "status": direction,
            "severity": float(severity),
            "adjustable": f in SOIL_FEATURES,
        })
    return sorted(rows, key=lambda x: x["severity"], reverse=True)


def agricultural_suitability(values: dict[str, float], ranked: list[dict[str, float | str]], bundle: CropBundle) -> dict[str, Any]:
    profiles = bundle.metadata["profiles"]
    supported = 0
    for crop, p in profiles.items():
        ok_count = 0
        for f in FEATURES:
            q1, q3 = p[f]["q1"], p[f]["q3"]
            v = float(values[f])
            if q1 <= v <= q3:
                ok_count += 1
        if ok_count >= 4:
            supported += 1
    top_prob = float(ranked[0]["probability"]) if ranked else 0.0
    # This is an explicit decision layer, not a supervised binary label.
    suitable = supported > 0 and top_prob >= 0.20
    return {
        "suitable": bool(suitable),
        "status": "Suitable for agriculture" if suitable else "Poor fit for the supported crop profiles",
        "supporting_crop_profiles": int(supported),
        "top_crop_probability": top_prob,
        "method": "Derived from learned crop-class feature ranges and model probabilities; not a separately supervised binary label.",
    }


def target_suitability(target_crop: str, ranked: list[dict[str, float | str]], gaps: list[dict[str, Any]]) -> dict[str, Any]:
    target_row = next((r for r in ranked if r["crop"] == target_crop), None)
    prob = float(target_row["probability"]) if target_row else 0.0
    major = [g for g in gaps if g["status"] != "within"]
    # threshold is deliberately conservative; users still see the numeric score.
    suitable = prob >= 0.35 and len([g for g in major if g["severity"] > 1.0]) <= 2
    return {
        "crop": target_crop,
        "suitable": bool(suitable),
        "probability": prob,
        "status": "Suitable for the selected crop" if suitable else "Not suitable for the selected crop under current conditions",
        "major_gaps": major[:5],
        "method": "Derived from multiclass probability plus target-crop empirical quartile comparison; not a separately supervised binary target.",
    }


def remedies(gaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for g in gaps:
        if g["status"] == "within":
            continue
        f = g["feature"]
        direction = g["status"]
        if f in {"N", "P", "K"}:
            nutrient = {"N": "Nitrogen", "P": "Phosphorus", "K": "Potassium"}[f]
            action = (f"Increase {nutrient} using a soil-test-guided fertilizer or organic amendment. "
                      if direction == "low" else
                      f"Reduce additional {nutrient} inputs and avoid over-fertilization; re-test the soil before applying more.")
            out.append({"feature": f, "severity": g["severity"], "remedy": action, "type": "soil"})
        elif f == "ph":
            action = ("Raise soil pH gradually with an agronomist/soil-test-guided liming program. "
                      if direction == "low" else
                      "Lower soil pH gradually with soil-test-guided acidifying amendments and organic matter.")
            out.append({"feature": f, "severity": g["severity"], "remedy": action, "type": "soil"})
        elif f == "rainfall":
            action = ("Rainfall is below the target crop profile; use supplemental irrigation and moisture-conserving practices. "
                      if direction == "low" else
                      "Rainfall is above the target crop profile; improve drainage/raised beds where appropriate and avoid waterlogging.")
            out.append({"feature": f, "severity": g["severity"], "remedy": action, "type": "climate"})
        elif f == "temperature":
            out.append({"feature": f, "severity": g["severity"], "remedy": "Temperature is outside the target crop profile; choose a more suitable sowing season/location or use protected cultivation where practical.", "type": "climate"})
        elif f == "humidity":
            out.append({"feature": f, "severity": g["severity"], "remedy": "Humidity is outside the target crop profile; adjust irrigation/ventilation and consider a crop-season combination better matched to local humidity.", "type": "climate"})
    return sorted(out, key=lambda x: x["severity"], reverse=True)
