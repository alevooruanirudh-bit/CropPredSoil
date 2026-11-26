from __future__ import annotations

from pathlib import Path
import json
import sys
import urllib.request

DATASET_ID = "zyyb98msjc"
VERSION = 1
OUT = Path("data/raw/mendeley_agro")
API_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files?folder_id=root&version={VERSION}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Querying Mendeley Data file list...")
    try:
        with urllib.request.urlopen(API_URL, timeout=30) as r:
            files = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(
            "Could not download the Mendeley dataset automatically. "
            "Check your internet connection and try again. "
            f"\nDetails: {exc}"
        )

    if not files:
        raise SystemExit("Mendeley returned no files for this dataset/version.")

    for item in files:
        name = Path(item.get("name", "download.bin")).name
        url = item.get("content_details", {}).get("download_url")
        if not url:
            print(f"Skipping {name}: no download URL returned")
            continue
        destination = OUT / name
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, destination)
        print(f"Saved {destination}")


if __name__ == "__main__":
    main()
