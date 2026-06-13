"""
Exploratory Data Analysis Script for Heart Disease Dataset
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
    """Load the master heart dataset"""
    df = pd.read_csv('datasets/heart/heart_master_dataset.csv')
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

def analyze_target_distribution(df, target_col='target'):
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

def correlation_analysis(df, target_col='target', top_n=10):
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

def analyze_heart_specific_features(df):
    """Analyze heart disease specific features"""
    print("\n" + "="*60)
    print("HEART DISEASE SPECIFIC ANALYSIS")
    print("="*60)
    
    # Age groups analysis
    if 'age' in df.columns:
        print("\nAge Distribution:")
        print(f"  Mean Age: {df['age'].mean():.1f} years")
        print(f"  Median Age: {df['age'].median():.1f} years")
        print(f"  Age Range: {df['age'].min():.0f} - {df['age'].max():.0f} years")
    
    # Sex distribution
    if 'sex' in df.columns:
        print("\nSex Distribution:")
        sex_counts = df['sex'].value_counts()
        for sex, count in sex_counts.items():
            pct = (count / len(df)) * 100
            sex_label = "Male" if sex == 1 else "Female"
            print(f"  {sex_label}: {count} ({pct:.1f}%)")
    
    # Chest pain types
    if 'chest_pain_type' in df.columns:
        print("\nChest Pain Type Distribution:")
        cp_counts = df['chest_pain_type'].value_counts().sort_index()
        cp_labels = {1: "Typical Angina", 2: "Atypical Angina", 
                     3: "Non-anginal Pain", 4: "Asymptomatic"}
        for cp_type, count in cp_counts.items():
            pct = (count / len(df)) * 100
            label = cp_labels.get(cp_type, f"Type {cp_type}")
            print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Blood pressure analysis
    if 'resting_bp' in df.columns:
        print("\nResting Blood Pressure:")
        print(f"  Mean: {df['resting_bp'].mean():.1f} mm Hg")
        print(f"  Median: {df['resting_bp'].median():.1f} mm Hg")
        hypertension = (df['resting_bp'] > 140).sum()
        print(f"  Hypertension (>140): {hypertension} ({(hypertension/len(df)*100):.1f}%)")
    
    # Cholesterol analysis
    if 'cholesterol' in df.columns:
        print("\nCholesterol Levels:")
        print(f"  Mean: {df['cholesterol'].mean():.1f} mg/dl")
        print(f"  Median: {df['cholesterol'].median():.1f} mg/dl")
        high_chol = (df['cholesterol'] > 240).sum()
        print(f"  High Cholesterol (>240): {high_chol} ({(high_chol/len(df)*100):.1f}%)")
    
    # Max heart rate analysis
    if 'max_heart_rate' in df.columns:
        print("\nMaximum Heart Rate:")
        print(f"  Mean: {df['max_heart_rate'].mean():.1f} bpm")
        print(f"  Median: {df['max_heart_rate'].median():.1f} bpm")
        print(f"  Range: {df['max_heart_rate'].min():.0f} - {df['max_heart_rate'].max():.0f} bpm")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("Heart Disease Dataset")
    print("="*60)
    
    # Load data
    df = load_master_dataset()
    
    # Perform analyses
    stats = basic_statistics(df)
    target_dist = analyze_target_distribution(df)
    correlations = correlation_analysis(df)
    skewness = feature_distribution_summary(df)
    outliers = outlier_detection(df)
    analyze_heart_specific_features(df)
    
    print("\n" + "="*60)
    print("EDA COMPLETE!")
    print("="*60)
    print("\nFor visualizations, check: reports/heart/eda/")

# Made with Bob
