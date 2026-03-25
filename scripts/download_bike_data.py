#!/usr/bin/env python3
"""
Download the Bike Sharing Dataset from UCI Machine Learning Repository.

This script downloads and extracts the hour.csv file needed for
the InstaSHAP paper replication.

Usage:
    python scripts/download_bike_data.py
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# UCI dataset URL
BIKE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def download_bike_sharing():
    """Download and extract the Bike Sharing Dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    hour_csv = DATA_DIR / "hour.csv"
    if hour_csv.exists():
        print(f"[OK] hour.csv already exists at {hour_csv}")
        return

    zip_path = DATA_DIR / "bike_sharing.zip"

    print(f"Downloading Bike Sharing Dataset from UCI...")
    print(f"URL: {BIKE_URL}")
    try:
        urllib.request.urlretrieve(BIKE_URL, zip_path)
        print(f"[OK] Downloaded to {zip_path}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print(f"\nPlease manually download the dataset from:")
        print(f"  https://archive.ics.uci.edu/dataset/275/bike-sharing-dataset")
        print(f"  and place hour.csv in {DATA_DIR}/")
        sys.exit(1)

    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        print(f"[OK] Extracted to {DATA_DIR}")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        sys.exit(1)

    # Clean up zip
    zip_path.unlink(missing_ok=True)

    if hour_csv.exists():
        print(f"[OK] hour.csv ready at {hour_csv}")
    else:
        # Some zip structures have subdirectories
        for f in DATA_DIR.rglob("hour.csv"):
            f.rename(hour_csv)
            print(f"[OK] Moved hour.csv to {hour_csv}")
            break
        else:
            print(f"[WARN] hour.csv not found after extraction.")
            print(f"Please check {DATA_DIR} and move hour.csv manually.")


if __name__ == "__main__":
    download_bike_sharing()
