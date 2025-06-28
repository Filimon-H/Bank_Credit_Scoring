import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from IPython.display import display
import math


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    return pd.read_csv(filepath)


def data_overview(df: pd.DataFrame):
    """Print shape and data types."""
    print('Shape:', df.shape)
    print('Data types:')
    print(df.dtypes)


def summary_statistics(df: pd.DataFrame):
    """Display summary statistics for numerical features."""
    display(df.describe())


def plot_numerical_distributions(df: pd.DataFrame) -> None:
    """
    Histogram plots of numerical features with KDE, mean, and median lines — arranged in subplots.
    """
    numerical_data = df.select_dtypes(include=['number'])
    numerical_cols = numerical_data.columns

    # Grid size
    num_cols = math.ceil(len(numerical_cols) ** 0.5)
    num_rows = math.ceil(len(numerical_cols) / num_cols)

    # Subplots setup
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(6 * num_cols, 4 * num_rows))
    axes = axes.flatten()

    for idx, column in enumerate(numerical_cols):
        data = df[column].dropna()
        mean = data.mean()
        median = data.median()

        sns.histplot(data, bins=30, kde=True, ax=axes[idx], color='skyblue')
        axes[idx].set_title(f'Distribution of {column}', fontsize=10)
        axes[idx].set_xlabel(column, fontsize=9)
        axes[idx].set_ylabel('Frequency', fontsize=9)
        axes[idx].axvline(mean, color='black', linestyle='--', linewidth=1, label='Mean')
        axes[idx].axvline(median, color='red', linestyle='-', linewidth=1, label='Median')
        axes[idx].legend()

    # Remove any unused axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()



#def plot_numerical_distributions(df: pd.DataFrame, num_cols: List[str]):
#    """Plot distributions for numerical features."""
#    for col in num_cols:
#        plt.figure(figsize=(6, 4))
#        sns.histplot(df[col].dropna(), kde=True)
#        plt.title(f'Distribution of {col}')
#        plt.show()

def plot_categorical_distributions(df: pd.DataFrame, cat_cols: List[str], top_n: int = 10, max_unique: int = 30):
    """
    Plot top N category counts for each categorical feature.
    Skips columns with too many unique values for better performance.
    
    Parameters:
        df (pd.DataFrame): The dataset
        cat_cols (List[str]): List of categorical column names
        top_n (int): Number of top categories to show
        max_unique (int): Max unique values allowed for plotting
    """
    for col in cat_cols:
        unique_vals = df[col].nunique()
        if unique_vals > max_unique:
            print(f"⏩ Skipping '{col}' – {unique_vals} unique values (too high)")
            continue
        
        plt.figure(figsize=(6, 4))
        df[col].value_counts().nlargest(top_n).plot(kind='bar', color='coral')
        plt.title(f'Top {top_n} {col} Categories')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()







def correlation_analysis(df: pd.DataFrame, num_cols: List[str]):
    """Plot correlation heatmap for numerical features."""
    plt.figure(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()


def missing_values(df: pd.DataFrame):
    """Display missing value counts per column."""
    missing = df.isnull().sum()
    print('Missing values per column:')
    print(missing[missing > 0])


def outlier_detection(df: pd.DataFrame, num_cols: List[str]):
    """Box plots for outlier detection in numerical features."""
    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.show()
