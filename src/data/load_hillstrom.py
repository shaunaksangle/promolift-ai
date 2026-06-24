"""Load and prepare the Hillstrom Email Marketing dataset.

This module downloads the raw Hillstrom marketing experiment data and prepares
a simple binary treatment dataset for the first uplift modeling version.
"""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
RAW_HILLSTROM_PATH = RAW_DATA_DIR / "hillstrom.csv"
PROCESSED_HILLSTROM_PATH = PROCESSED_DATA_DIR / "hillstrom_mens_email.csv"


def download_hillstrom_dataset() -> Path:
    """Download the Hillstrom CSV file into the raw data directory.

    The file is downloaded only when it is not already present locally.

    Returns:
        Path to the local raw Hillstrom CSV file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_HILLSTROM_PATH.exists():
        print(f"Hillstrom raw dataset already exists: {RAW_HILLSTROM_PATH}")
        return RAW_HILLSTROM_PATH

    print("Downloading Hillstrom dataset...")
    urlretrieve(HILLSTROM_URL, RAW_HILLSTROM_PATH)
    print(f"Hillstrom raw dataset downloaded to: {RAW_HILLSTROM_PATH}")

    return RAW_HILLSTROM_PATH


def load_raw_hillstrom() -> pd.DataFrame:
    """Load the raw Hillstrom dataset as a pandas DataFrame.

    If the raw CSV file is missing, it is downloaded first.

    Returns:
        Raw Hillstrom data.
    """
    if not RAW_HILLSTROM_PATH.exists():
        download_hillstrom_dataset()

    return pd.read_csv(RAW_HILLSTROM_PATH)


def prepare_hillstrom_binary_treatment() -> pd.DataFrame:
    """Prepare a binary treatment dataset for Mens E-Mail versus No E-Mail.

    The first project version uses Mens E-Mail as treatment and No E-Mail as
    control. Womens E-Mail rows are excluded to keep the comparison simple.

    Returns:
        Clean DataFrame with customer IDs, treatment, outcome, features, and
        business outcome columns.
    """
    raw_data = load_raw_hillstrom()
    raw_data.columns = (
        raw_data.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    filtered_data = raw_data[
        raw_data["segment"].isin(["Mens E-Mail", "No E-Mail"])
    ].copy()
    filtered_data = filtered_data.reset_index(drop=True)

    filtered_data["customer_id"] = range(1, len(filtered_data) + 1)
    filtered_data["treatment"] = (
        filtered_data["segment"].eq("Mens E-Mail").astype(int)
    )
    filtered_data["outcome"] = filtered_data["conversion"].astype(int)

    output_columns = [
        "customer_id",
        "treatment",
        "outcome",
        "recency",
        "history_segment",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
        "visit",
        "conversion",
        "spend",
    ]

    return filtered_data[output_columns]


def save_processed_hillstrom() -> pd.DataFrame:
    """Save the processed Hillstrom binary treatment dataset.

    Returns:
        The processed DataFrame that was written to disk.
    """
    processed_data = prepare_hillstrom_binary_treatment()
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    processed_data.to_csv(PROCESSED_HILLSTROM_PATH, index=False)

    print(f"Processed Hillstrom dataset saved to: {PROCESSED_HILLSTROM_PATH}")
    print(f"Saved dataset shape: {processed_data.shape}")
    print("Treatment distribution:")
    print(processed_data["treatment"].value_counts().sort_index())
    print("Outcome distribution:")
    print(processed_data["outcome"].value_counts().sort_index())

    return processed_data


if __name__ == "__main__":
    save_processed_hillstrom()
