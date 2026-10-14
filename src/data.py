"""Loading, splitting and small text utilities."""
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, SEED

_ARABIC = re.compile(r"[\u0600-\u06FF]")


def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    ar = sum(1 for c in letters if _ARABIC.match(c))
    return "ar" if ar / len(letters) > 0.5 else "en"


def load_tickets() -> pd.DataFrame:
    path = DATA_DIR / "tickets.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.make_data")
    return pd.read_csv(path)


def split(df: pd.DataFrame):
    """Stratified 80/20 split, done BEFORE any fitting so the test set never leaks."""
    train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["category"])
    assert set(train["text"]).isdisjoint(set(test["text"])), "train/test leakage detected"
    return train.reset_index(drop=True), test.reset_index(drop=True)
