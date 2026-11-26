from pathlib import Path
import json

EXPECTED = {
    "Field Crops – Cereals": 6,
    "Field Crops – Millets": 7,
    "Field Crops – Pseudo-cereals": 1,
    "Field Crops – Pulses": 14,
    "Oilseeds & Oil Crops": 11,
    "Commercial & Fibre Crops": 6,
    "Horticulture – Fruits": 39,
    "Horticulture – Vegetables": 42,
    "Perennial & Plantation Crops": 9,
    "Spices & Condiments": 18,
    "Specialized Horticulture": 21,
    "Forage & Fodder Crops": 8,
}

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "data" / "reference" / "india_crop_catalog.json"

data = json.loads(path.read_text(encoding="utf-8"))

counts = {}
for row in data:
    counts[row["major_category"]] = (
        counts.get(row["major_category"], 0) + 1
    )

assert len(data) == 182, (
    f"Expected 182 entries, got {len(data)}"
)
assert counts == EXPECTED, (
    f"Category counts mismatch: {counts}"
)

print("Indian crop catalog validation passed.")
print(f"Entries: {len(data)}")
for category, count in counts.items():
    print(f"{category}: {count}")
