import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(path: str):
    """Load the processed dataset and split into train and test sets."""
    df = pd.read_csv(path)
    
    X = df.drop(columns=['is_high_risk'])
    y = df['is_high_risk']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
    
    return metrics


import mlflow
import mlflow.sklearn

def log_experiment(model, metrics, model_name):
    with mlflow.start_run():
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.set_tag("model_name", model_name)


from sklearn.model_selection import GridSearchCV

def tune_random_forest(X_train, y_train):
    params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
    model = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(model, params, scoring='f1', cv=3)
    grid.fit(X_train, y_train)
    return grid.best_estimator_
