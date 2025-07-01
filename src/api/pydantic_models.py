# src/api/pydantic_models.py

from pydantic import BaseModel
import pandas as pd

class CustomerFeatures(BaseModel):
    Amount: float
    CountryCode: int
    Value: int
    PricingStrategy: int
    ProductCategory: int  # encoded already

    def to_df(self):
        return pd.DataFrame([self.dict()])

class PredictionResponse(BaseModel):
    probability: float
