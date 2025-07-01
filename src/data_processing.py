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
    Histogram plots of numerical features with KDE, mean, and median lines
    — arranged in subplots.
    """
    numerical_data = df.select_dtypes(include=['number'])
    numerical_cols = numerical_data.columns

    # Grid size
    num_cols = math.ceil(len(numerical_cols) ** 0.5)
    num_rows = math.ceil(len(numerical_cols) / num_cols)

    # Subplots setup
    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols,
        figsize=(6 * num_cols, 4 * num_rows)
    )
    axes = axes.flatten()

    for idx, column in enumerate(numerical_cols):
        data = df[column].dropna()
        mean = data.mean()
        median = data.median()

        sns.histplot(data, bins=30, kde=True, ax=axes[idx], color='skyblue')
        axes[idx].set_title(f'Distribution of {column}', fontsize=10)
        axes[idx].set_xlabel(column, fontsize=9)
        axes[idx].set_ylabel('Frequency', fontsize=9)
        axes[idx].axvline(
            mean, color='black', linestyle='--', linewidth=1, label='Mean'
        )
        axes[idx].axvline(
            median, color='red', linestyle='-', linewidth=1, label='Median'
        )
        axes[idx].legend()

    # Remove any unused axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(df: pd.DataFrame, cat_cols: List[str],
                                   top_n: int = 10, max_unique: int = 30):
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
            print(
                f"⏩ Skipping '{col}' – {unique_vals} unique values (too high)"
            )
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


def analyze_missing_data(df: pd.DataFrame) -> None:
    """
    Comprehensive missing data analysis with counts, percentages, and visualizations.
    """
    # Calculate missing data
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    # Create summary DataFrame
    missing_summary = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Count': missing_counts.values,
        'Missing_Percentage': missing_percentages.values
    }).sort_values('Missing_Percentage', ascending=False)
    
    # Filter columns with missing data
    missing_data = missing_summary[missing_summary['Missing_Count'] > 0]
    
    print("=== MISSING DATA ANALYSIS ===")
    print(f"Total rows in dataset: {len(df)}")
    print(f"Columns with missing data: {len(missing_data)}")
    print(f"Total missing values: {missing_data['Missing_Count'].sum()}")
    print(f"Overall missing percentage: {(missing_data['Missing_Count'].sum() / (len(df) * len(df.columns))) * 100:.2f}%")
    
    if len(missing_data) > 0:
        print("\n=== MISSING DATA BY COLUMN ===")
        print(missing_data.to_string(index=False))
        
        # Visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot of missing counts
        missing_data.plot(
            x='Column', y='Missing_Count', kind='bar', ax=ax1, color='red'
        )
        ax1.set_title('Missing Values Count by Column')
        ax1.set_xlabel('Columns')
        ax1.set_ylabel('Missing Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Bar plot of missing percentages
        missing_data.plot(
            x='Column', y='Missing_Percentage', kind='bar', ax=ax2, color='orange'
        )
        ax2.set_title('Missing Values Percentage by Column')
        ax2.set_xlabel('Columns')
        ax2.set_ylabel('Missing Percentage (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Missing data heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Data Heatmap')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()
        
    else:
        print("\n✅ No missing data found in the dataset!")


def missing_values(df: pd.DataFrame):
    """Display missing value counts per column."""
    missing = df.isnull().sum()
    print('Missing values per column:')
    print(missing[missing > 0])


def outlier_detection(df: pd.DataFrame) -> None:
    """
    A function that performs outlier detection by plotting a box plot.
    """
    # create the box plots of the numeric data
    ax = sns.boxplot(data=df, palette='husl')
    ax.set_title("Box-plot of Numerical Variables", pad=30, fontweight='bold')
    ax.set_xlabel("Numerical Columns", fontweight='bold', labelpad=10)
    ax.set_ylabel("Values", fontweight='bold', labelpad=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def count_outliers(df: pd.DataFrame) -> None:
    """
    A function that counts the number of outliers in numerical columns.
    The amount of data that are outliers and also gives the cut-off point.
    The cut off points being defined as:
        - lowerbound = Q1 - 1.5 * IQR
        - upperbound = Q3 + 1.5 * IQR
    """
    # get the numeric data
    numerical_columns = list(df._get_numeric_data().columns)
    numerical_data = df[numerical_columns]

    # obtain the Q1, Q3 and IQR(Inter-Quartile Range)
    quartile_one = numerical_data.quantile(0.25)
    quartile_three = numerical_data.quantile(0.75)
    iqr = quartile_three - quartile_one

    # obtain the upperbound and lowerbound values for each column
    upper_bound = quartile_three + 1.5 * iqr
    lower_bound = quartile_one - 1.5 * iqr

    # count all the outliers for the respective columns
    outliers = {"Columns": [], "Num. of Outliers": []}
    for column in lower_bound.keys():
        column_outliers = df[
            (df[column] < lower_bound[column]) |
            (df[column] > upper_bound[column])
        ]
        count = column_outliers.shape[0]

        outliers["Columns"].append(column)
        outliers["Num. of Outliers"].append(count)

    outliers = pd.DataFrame.from_dict(outliers).sort_values(by='Num. of Outliers')
    ax = sns.barplot(
        outliers, x='Columns', y='Num. of Outliers', palette='husl'
    )
    ax.set_title("Plot of Outlier Counts in Numerical Columns", pad=20)
    ax.set_xlabel("Numerical Columns", weight='bold')
    ax.set_ylabel("Num. of Outliers", weight="bold")
    ax.tick_params(axis='x', labelrotation=45)

    columns = outliers['Columns'].unique()
    for idx, patch in enumerate(ax.patches):
        # get the coordinates to write the values
        x_coordinate = patch.get_x() + patch.get_width() / 2
        y_coordinate = patch.get_height()

        # get the value of the coordinate
        value = outliers[
            outliers['Columns'] == columns[idx]
        ]['Num. of Outliers'].values[0]
        ax.text(
            x=x_coordinate, y=y_coordinate, s=value, ha='center',
            va='bottom', weight='bold'
        )

    plt.tight_layout()
    plt.show()





