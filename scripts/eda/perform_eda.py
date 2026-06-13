"""
Exploratory Data Analysis Script for Chronic Kidney Disease Dataset
This script performs comprehensive EDA including:
- Statistical summaries
- Distribution analysis
- Correlation analysis
- Missing value analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_master_dataset():
    """Load the master kidney dataset"""
    df = pd.read_csv('datasets/kidney/kidney_master_dataset.csv')
    return df

def basic_statistics(df):
    """Generate basic statistical summaries"""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Total Samples: {len(df)}")
    print(f"Total Features: {len(df.columns)}")
    
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values found")
    
    print("\nDuplicate Rows:", df.duplicated().sum())
    
    return df.describe()

def analyze_target_distribution(df, target_col='ckd_status'):
    """Analyze target variable distribution"""
    print("\n" + "="*60)
    print("TARGET DISTRIBUTION")
    print("="*60)
    
    target_counts = df[target_col].value_counts()
    target_pct = df[target_col].value_counts(normalize=True) * 100
    
    print(f"\nTarget Variable: {target_col}")
    print("\nClass Distribution:")
    for cls, count in target_counts.items():
        print(f"  Class {cls}: {count} samples ({target_pct[cls]:.2f}%)")
    
    # Check for class imbalance
    imbalance_ratio = target_counts.max() / target_counts.min()
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 1.5:
        print("WARNING: Dataset shows class imbalance. Consider using techniques like SMOTE or class weights.")
    
    return target_counts

def correlation_analysis(df, target_col='ckd_status', top_n=10):
    """Analyze feature correlations with target"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate correlations
    correlations = df.corr()[target_col].sort_values(ascending=False)
    correlations = correlations[correlations.index != target_col]
    
    print(f"\nTop {top_n} Positively Correlated Features:")
    for i, (feat, corr) in enumerate(correlations.head(top_n).items(), 1):
        print(f"{i:2d}. {feat:30s}: {corr:6.3f}")
    
    print(f"\nTop {top_n} Negatively Correlated Features:")
    for i, (feat, corr) in enumerate(correlations.tail(top_n).items(), 1):
        print(f"{i:2d}. {feat:30s}: {corr:6.3f}")
    
    return correlations

def feature_distribution_summary(df):
    """Summarize feature distributions"""
    print("\n" + "="*60)
    print("FEATURE DISTRIBUTION SUMMARY")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"\nNumeric Features: {len(numeric_cols)}")
    print("\nSkewness Analysis:")
    skewness = df[numeric_cols].skew().sort_values(ascending=False)
    
    highly_skewed = skewness[abs(skewness) > 1]
    if len(highly_skewed) > 0:
        print("\nHighly Skewed Features (|skew| > 1):")
        for feat, skew in highly_skewed.items():
            print(f"  {feat:30s}: {skew:6.3f}")
    else:
        print("No highly skewed features found")
    
    return skewness

def outlier_detection(df):
    """Detect outliers using IQR method"""
    print("\n" + "="*60)
    print("OUTLIER DETECTION")
    print("="*60)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_summary = {}
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        
        if len(outliers) > 0:
            outlier_summary[col] = len(outliers)
    
    if outlier_summary:
        print("\nFeatures with Outliers:")
        for feat, count in sorted(outlier_summary.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(df)) * 100
            print(f"  {feat:30s}: {count:4d} outliers ({pct:5.2f}%)")
    else:
        print("No outliers detected")
    
    return outlier_summary

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("Chronic Kidney Disease Dataset")
    print("="*60)
    
    # Load data
    df = load_master_dataset()
    
    # Perform analyses
    stats = basic_statistics(df)
    target_dist = analyze_target_distribution(df)
    correlations = correlation_analysis(df)
    skewness = feature_distribution_summary(df)
    outliers = outlier_detection(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print("\nFor visualizations, check: reports/kidney/eda/")


