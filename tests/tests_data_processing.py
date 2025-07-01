import pandas as pd
from src.model_training import load_and_split_data

def test_split_shapes():
    X_train, X_test, y_train, y_test = load_and_split_data("data/processed/final_features.csv")
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(X_train) == len(y_train)
