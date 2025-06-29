# src/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    OneHotEncoder,
    LabelEncoder,
    FunctionTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from xverse.transformer import MonotonicBinning
import warnings
warnings.filterwarnings('ignore')

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from datetime columns"""
    def __init__(self, datetime_col='transaction_date'):
        self.datetime_col = datetime_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        if self.datetime_col in X.columns:
            dt = pd.to_datetime(X[self.datetime_col])
            X['transaction_hour'] = dt.dt.hour
            X['transaction_day'] = dt.dt.day
            X['transaction_month'] = dt.dt.month
            X['transaction_year'] = dt.dt.year
            X['transaction_dayofweek'] = dt.dt.dayofweek
            X['transaction_is_weekend'] = (X['transaction_dayofweek'] >= 5).astype(int)
        return X

class Aggregator(BaseEstimator, TransformerMixin):
    """Create aggregate features at customer level"""
    def __init__(self, customer_id_col='customer_id', amount_col='amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        if self.customer_id_col in X.columns and self.amount_col in X.columns:
            agg_features = X.groupby(self.customer_id_col)[self.amount_col].agg([
                ('total_amount', 'sum'),
                ('avg_amount', 'mean'),
                ('transaction_count', 'count'),
                ('amount_std', 'std'),
                ('amount_min', 'min'),
                ('amount_max', 'max')
            ]).reset_index()
            
            # Merge aggregated features back to original data
            X = X.merge(agg_features, on=self.customer_id_col, how='left')
            
            # Fill NA for customers with single transaction (std will be NA)
            X['amount_std'] = X['amount_std'].fillna(0)
        return X

class WoeTransformer(BaseEstimator, TransformerMixin):
    """Apply Weight of Evidence encoding to categorical features"""
    def __init__(self, cat_cols=None, target_col='target'):
        self.cat_cols = cat_cols
        self.target_col = target_col
        self.woe = {}
        
    def fit(self, X, y=None):
        if self.cat_cols:
            for col in self.cat_cols:
                if col in X.columns:
                    # Simple WOE calculation without external package
                    woe_dict = self._calculate_woe(X[col], X[self.target_col])
                    self.woe[col] = woe_dict
        return self
        
    def _calculate_woe(self, feature, target):
        """Calculate Weight of Evidence for a categorical feature"""
        woe_dict = {}
        for category in feature.unique():
            if pd.isna(category):
                continue
            event_count = len(feature[(feature == category) & (target == 1)])
            non_event_count = len(feature[(feature == category) & (target == 0)])
            total_event = len(target[target == 1])
            total_non_event = len(target[target == 0])
            
            if event_count > 0 and non_event_count > 0:
                woe = np.log((event_count / total_event) / (non_event_count / total_non_event))
                woe_dict[category] = woe
            else:
                woe_dict[category] = 0
        return woe_dict
        
    def transform(self, X):
        X = X.copy()
        for col, woe_dict in self.woe.items():
            if col in X.columns:
                X[f'{col}_woe'] = X[col].map(woe_dict).fillna(0)
        return X

def get_feature_engineering_pipeline(config):
    """
    Main function to create the complete feature engineering pipeline
    
    Args:
        config (dict): Configuration dictionary containing:
            - datetime_col: Name of datetime column
            - customer_id_col: Name of customer ID column
            - amount_col: Name of transaction amount column
            - cat_cols: List of categorical columns
            - num_cols: List of numerical columns
            - target_col: Name of target variable
            - imputation_strategy: Strategy for handling missing values
            - scaling_method: 'standard' or 'minmax'
    
    Returns:
        sklearn.Pipeline: Complete feature engineering pipeline
    """
    # Extract configuration parameters
    datetime_col = config.get('datetime_col', 'transaction_date')
    customer_id_col = config.get('customer_id_col', 'customer_id')
    amount_col = config.get('amount_col', 'amount')
    cat_cols = config.get('cat_cols', [])
    num_cols = config.get('num_cols', [])
    target_col = config.get('target_col', 'target')
    imputation_strategy = config.get('imputation_strategy', 'mean')
    scaling_method = config.get('scaling_method', 'standard')
    
    # Define transformers for different feature types
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=imputation_strategy)),
        ('scaler', StandardScaler() if scaling_method == 'standard' else MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Create column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='passthrough'
    )
    



    
    # Create complete pipeline
    pipeline = Pipeline([
        ('feature_extractor', FeatureExtractor(datetime_col)),
        ('aggregator', Aggregator(customer_id_col, amount_col)),
        ('woe_encoder', WoeTransformer(cat_cols, target_col)),
        ('preprocessor', preprocessor)
    ])

   

    
    return pipeline

def save_pipeline(pipeline, filepath):
    """Save the pipeline to disk"""
    import joblib
    joblib.dump(pipeline, filepath)
    
def load_pipeline(filepath):
    """Load pipeline from disk"""
    import joblib
    return joblib.load(filepath)







def get_pipeline_feature_names(pipeline, config):
    """
    Get final feature names based on actual fitted pipeline output.
    """
    feature_names = []

    preprocessor = pipeline.named_steps["preprocessor"]

    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            try:
                ohe = transformer.named_steps["onehot"]
                ohe_cols = ohe.get_feature_names_out(cols)
                feature_names.extend(ohe_cols)
            except Exception:
                feature_names.extend(cols)
        elif name == "remainder":
            # Optional passthrough — if explicitly listed
            if isinstance(cols, list):
                feature_names.extend(cols)

    return feature_names







#def get_pipeline_feature_names(pipeline, config):
    """
    Get final feature names from the pipeline based on config and actual fitted encoders.
    
    Args:
        pipeline (Pipeline): Fitted pipeline
        config (dict): Same config used to build the pipeline

    Returns:
        List[str]: List of all output column names
    """
    cat_cols = config.get('cat_cols', [])
    num_cols = config.get('num_cols', [])
    
    # Extract from OneHotEncoder
    cat_transformer = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    ohe_feature_names = cat_transformer.named_steps['onehot'].get_feature_names_out(cat_cols).tolist()
    
    # Add original numerical columns
    numeric_feature_names = num_cols

    # Add manually engineered features from custom steps
    date_features = [
        'transaction_hour', 'transaction_day', 'transaction_month', 
        'transaction_year', 'transaction_dayofweek', 'transaction_is_weekend'
    ]
    agg_features = [
        'total_amount', 'avg_amount', 'transaction_count', 
        'amount_std', 'amount_min', 'amount_max'
    ]
    woe_features = [f'{col}_woe' for col in cat_cols]

    return numeric_feature_names + ohe_feature_names + date_features + agg_features + woe_features
