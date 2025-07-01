
# Credit Risk Probability Model for Alternative Data

## 🚀 Project Overview

This project is developed for **Bati Bank** in collaboration with a leading eCommerce company to enable a **Buy-Now-Pay-Later (BNPL)** service. The core business objective is to build a robust credit scoring model that estimates the likelihood of customer default using **alternative behavioral data** from the eCommerce platform.

Given the absence of traditional credit history, we leverage **transactional patterns**, such as Recency, Frequency, and Monetary (RFM) data, to engineer features and train a predictive model. This model will ultimately assign a **risk probability score**, a **credit score**, and guide loan amount and term decisions.

---

## 📘 Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretability

The **Basel II Accord** emphasizes accurate and transparent measurement of credit risk. In line with this, our credit scoring model must be interpretable and well-documented to support:

* Regulatory review
* Auditability
* Internal governance

An interpretable model builds trust with stakeholders and ensures alignment with capital adequacy requirements.

### 2. Importance of a Proxy Variable

Since our dataset lacks an explicit "default" label, we must construct a **proxy variable** (e.g., based on abnormal transaction behavior, inactivity, or refund frequency) to simulate loan default behavior.

⚠️ **Risks of using a proxy:**

* **False assumptions:** The proxy may not capture true default intent.
* **Bias propagation:** Any proxy bias will be inherited by the model.
* **Business impact:** Could lead to misclassifying customers, affecting profits or reputational trust.

### 3. Simple vs. Complex Models in Regulated Environments

| Criteria              | Logistic Regression (WoE) | Gradient Boosting (GBM)          |
| --------------------- | ------------------------- | -------------------------------- |
| Interpretability      | High                      | Low                              |
| Regulatory acceptance | Preferred                 | Requires explanation (SHAP/LIME) |
| Performance           | Moderate                  | High                             |
| Deployment complexity | Low                       | Medium to High                   |

Choosing the right model depends on balancing **performance vs. interpretability**, especially in highly regulated financial environments.

---

## 🗂️ Project Structure Overview

```
credit-risk-model/
├── .github/workflows/ci.yml     # CI/CD Pipeline
├── data/
│   ├── raw/                     # Original data (excluded via .gitignore)
│   └── processed/               # Cleaned and transformed data
├── notebooks/
│   └── 1.0-eda.ipynb            # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py       # EDA utilities
│   └── feature_engineering.py   # Feature engineering logic
├── tests/
│   └── test_data_processing.py  # Unit tests
├── requirements.txt             # Dependencies
├── Dockerfile
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## 📊 Interim Task Progress

### ✅ Task 1 – Credit Scoring Business Understanding

* Basel II principles researched and applied
* Proxy risk logic outlined
* Trade-offs between model types discussed

### ✅ Task 2 – Exploratory Data Analysis (EDA)

* Numerical/categorical distributions visualized using subplots
* Missing values and outliers identified and quantified
* Correlation matrix plotted for numerical fields


🧠 Task 3 – Feature Engineering & Proxy Target Creation
Constructed Recency, Frequency, and Monetary (RFM) features using transaction history.

Engineered behavioral features such as:

days_since_last_transaction

total_transaction_count

average_transaction_amount

transaction_volume_per_channel

Handled high-cardinality categorical features with frequency or target encoding.

✅ Proxy Target Variable
Since explicit default labels were missing:

Constructed a proxy target based on:

Inactivity after high RFM scores

Abnormal transaction refund patterns

Validated proxy label distribution and balanced the dataset using undersampling.

✅ Ready to use engineered features and proxy variable for model training.

🤖 Task 4 – Model Training, Evaluation, and Tracking
Trained multiple supervised classifiers:

Logistic Regression (baseline)

Random Forest (tuned via RandomizedSearchCV)

✅ Evaluation Metrics
Accuracy, Precision, Recall, F1 Score

ROC AUC used for model selection

📈 Best Model Selection
Random Forest selected based on highest ROC AUC

Hyperparameters and performance logged using MLflow

🔬 Model Registry
Best model registered under:

yaml
Copy
Edit
Model Name: FraudDetectionModel
Stage: Staging
✅ Model saved both locally and in the MLflow Registry.

🧪 Task 5 – Model Interpretability
In line with Basel II interpretability requirements:

Interpreted model predictions using SHAP

Identified key features driving default predictions (e.g., low frequency, high recency)

Plotted SHAP summary and force plots to visualize:

Global feature importance

Per-customer risk explanations

✅ Interpretability analysis helps build stakeholder trust and regulatory compliance.

🌐 Task 6 – Model Deployment and Continuous Integration
✅ FastAPI Deployment
Developed REST API using FastAPI in src/api/main.py

Exposed /predict endpoint to return customer risk probability

Validated requests and responses with Pydantic models

📦 Dockerized Service
Dockerfile and docker-compose setup:

Runs the FastAPI app via uvicorn

Exposes the service on localhost:8000

bash
Copy
Edit
docker-compose up --build
⚙️ CI/CD Pipeline with GitHub Actions
Configured GitHub Actions workflow in .github/workflows/ci.yml

Workflow includes:

flake8 linter to check code style

pytest to run unit tests

Fails build on any style/test error
