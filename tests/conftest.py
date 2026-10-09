from pathlib import Path

import pandas as pd
import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def bronze():
    return pd.read_csv(DATA_DIR / "bronze.csv")


@pytest.fixture(scope="session")
def silver():
    return pd.read_csv(DATA_DIR / "silver.csv")


@pytest.fixture(scope="session")
def gold():
    return pd.read_csv(DATA_DIR / "gold.csv")
