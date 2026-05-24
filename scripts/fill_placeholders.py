"""
Fill {{placeholder}} tokens in Bitext data with realistic Faker-generated values.

WHY: 43.9% of rows contain unfilled template tokens like {{name}}, {{order_id}},
{{eta}}, {{invoice_id}}. These make examples look abstract and hard to read.
Filling them creates realistic-looking conversations useful for analysis demos.

Writes data/processed/bitext_filled.parquet — does NOT replace bitext_clean.parquet.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED_PATH
from src.data.loader import dataset_ready, load_real_dataframe

try:
    from faker import Faker
except ImportError:
    print("Install faker: pip install faker", file=sys.stderr)
    raise SystemExit(1)

_fake = Faker()

OUTPUT_PATH = DATA_PROCESSED_PATH.parent / "bitext_filled.parquet"

# Map each placeholder token to a Faker callable
_FILLERS: dict[str, callable] = {
    "name": lambda: _fake.first_name(),
    "order_id": lambda: f"ORD-{_fake.numerify('######')}",
    "invoice_id": lambda: f"INV-{_fake.numerify('######')}",
    "package_id": lambda: f"PKG-{_fake.numerify('########')}",
    "tracking_id": lambda: f"TRK-{_fake.numerify('##########')}",
    "eta": lambda: _fake.date_between(start_date="+1d", end_date="+14d").strftime("%B %d"),
    "issue": lambda: _fake.sentence(nb_words=5).rstrip(".").lower(),
    "product": lambda: _fake.word().capitalize(),
    "email": lambda: _fake.email(),
    "phone": lambda: _fake.phone_number(),
    "date": lambda: _fake.date_between(start_date="-30d", end_date="+7d").strftime("%B %d"),
    "amount": lambda: f"${_fake.pydecimal(left_digits=3, right_digits=2, positive=True)}",
    "reason": lambda: _fake.sentence(nb_words=4).rstrip(".").lower(),
    "account_type": lambda: _fake.random_element(["standard", "premium", "basic"]),
    "username": lambda: _fake.user_name(),
    "new_email": lambda: _fake.email(),
    "address": lambda: _fake.address().replace("\n", ", "),
    "city": lambda: _fake.city(),
    "country": lambda: _fake.country(),
    "zip_code": lambda: _fake.postcode(),
}

_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def _fill_text(text: str) -> str:
    """Replace all {{token}} occurrences with Faker-generated values."""
    def _replace(match: re.Match) -> str:
        key = match.group(1).lower()
        filler = _FILLERS.get(key)
        if filler:
            return str(filler())
        # Unknown placeholder: generate a plausible short phrase
        return _fake.word()

    return _PLACEHOLDER_RE.sub(_replace, text)


def fill_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with all placeholders filled."""
    out = df.copy()
    out["instruction"] = out["instruction"].astype(str).map(_fill_text)
    out["response"] = out["response"].astype(str).map(_fill_text)
    out["source"] = "filled"
    return out


def main() -> None:
    if not dataset_ready():
        print("Run: python -m src.data.preprocess", file=sys.stderr)
        raise SystemExit(1)

    print("Loading dataset...")
    df = load_real_dataframe()

    # Only process rows that actually contain placeholders
    mask = df["instruction"].str.contains(r"\{\{", na=False) | \
           df["response"].str.contains(r"\{\{", na=False)
    to_fill = df[mask].copy()
    already_clean = df[~mask].copy()

    print(f"Filling placeholders in {len(to_fill):,} rows (skipping {len(already_clean):,} clean rows)...")
    filled = fill_dataframe(to_fill)

    out = pd.concat([already_clean, filled], ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out):,} rows to {OUTPUT_PATH}")
    print(f"  {len(to_fill):,} rows had placeholders filled")
    print(f"  {len(already_clean):,} rows were already clean")


if __name__ == "__main__":
    main()
