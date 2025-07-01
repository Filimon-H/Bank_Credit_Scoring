# tests/test_data_processing.py

import pandas as pd


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from test_data import preprocess_transactions  # ✅ Now importing from test_data.py


def test_amount_filled_with_zero():
    df = pd.DataFrame({"Amount": [10.0, None, 5.5], "TransactionStartTime": ["2024-01-01", "2024-01-02", "2024-01-03"], "ProductCategory": ["Data", "Voice", "SMS"]})
    result = preprocess_transactions(df)
    assert result["Amount"].isnull().sum() == 0

def test_datetime_conversion():
    df = pd.DataFrame({
        "TransactionStartTime": ["2024-01-01", "bad-date", "2024-03-01"],
        "Amount": [1, 2, 3],
        "ProductCategory": ["A", "B", "C"]
    })
    result = preprocess_transactions(df)
    assert pd.api.types.is_datetime64_any_dtype(result["TransactionStartTime"])
