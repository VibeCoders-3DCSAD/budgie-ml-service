from __future__ import annotations

import pandas as pd


def transactions_to_frame(transactions: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["amount"] = df["amount"].astype(float)
    df["transaction_type"] = df["transaction_type"].str.lower()
    return df
