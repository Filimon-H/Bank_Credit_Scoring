import pandas as pd

def preprocess_transactions(df):
    df = df.copy()
    
    # Fill missing values
    df['Amount'] = df['Amount'].fillna(0)
    
    # Convert datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
    
    # Encode product category
    df['ProductCategory'] = df['ProductCategory'].astype('category').cat.codes
    
    return df