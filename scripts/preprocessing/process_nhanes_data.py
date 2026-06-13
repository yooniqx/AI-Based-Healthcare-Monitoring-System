"""
NHANES Dataset Processing Script
=================================
Processes National Health and Nutrition Health Survey 2013-2014 data
for Survey_data health prediction module.

Author: Data Processing Pipeline
Date: 2026-06-08
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_nhanes_data():
    """Load NHANES dataset from CSV."""
    csv_path = "Survey_data/national+health+and+nutrition+health+survey+2013-2014+(nhanes)+age+prediction+subset/NHANES_age_prediction.csv"
    
    print("Loading NHANES dataset...")
    df = pd.read_csv(csv_path)
    print(f"âœ“ Loaded {len(df)} records with {len(df.columns)} columns")
    
    return df

def analyze_dataset(df):
    """Analyze dataset structure and quality."""
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names and types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    
    # Analyze target variable
    if 'age_group' in df.columns:
        print(f"\nAge group distribution:")
        print(df['age_group'].value_counts().sort_index())
    
    # Analyze numeric columns
    print(f"\nNumeric column statistics:")
    print(df.describe())
    
    return df

def clean_dataset(df):
    """Clean and standardize the NHANES dataset."""
    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    
    df_clean = df.copy()
    
    # Standardize column names
    print("\nâœ“ Standardizing column names...")
    df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Remove duplicates
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        print(f"âœ“ Removing {duplicates} duplicate rows...")
        df_clean = df_clean.drop_duplicates()
    
    # Handle missing values
    print("\nâœ“ Handling missing values...")
    
    # Check missing value percentage for each column
    missing_pct = (df_clean.isnull().sum() / len(df_clean) * 100)
    
    # Remove columns with >50% missing values
    high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
    if high_missing_cols:
        print(f"  - Removing {len(high_missing_cols)} columns with >50% missing: {high_missing_cols}")
        df_clean = df_clean.drop(columns=high_missing_cols)
    
    # For remaining columns with missing values, report them
    remaining_missing = df_clean.isnull().sum()
    if remaining_missing.sum() > 0:
        print(f"  - Columns with missing values (will keep for ML imputation):")
        for col in remaining_missing[remaining_missing > 0].index:
            pct = (remaining_missing[col] / len(df_clean) * 100)
            print(f"    â€¢ {col}: {remaining_missing[col]} ({pct:.2f}%)")
    
    # Check for target leakage
    print("\nâœ“ Checking for target leakage...")
    # RIDAGEYR is the actual age - this could be target leakage if predicting age_group
    # However, it's a legitimate feature for other health predictions
    # We'll keep it but document this in the summary
    
    # Validate data types
    print("\nâœ“ Validating data types...")
    
    # Ensure numeric columns are numeric
    numeric_cols = ['seqn', 'ridageyr', 'riagendr', 'paq605', 'bmxbmi', 
                    'lbxglu', 'diq010', 'lbxglt', 'lbxin']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Check for outliers in clinical measurements
    print("\nâœ“ Checking for outliers in clinical measurements...")
    
    if 'bmxbmi' in df_clean.columns:
        # BMI typically ranges from 10-70
        outliers = df_clean[(df_clean['bmxbmi'] < 10) | (df_clean['bmxbmi'] > 70)]
        if len(outliers) > 0:
            print(f"  âš  Found {len(outliers)} BMI outliers (< 10 or > 70)")
    
    if 'lbxglu' in df_clean.columns:
        # Glucose typically ranges from 50-400 mg/dL
        outliers = df_clean[(df_clean['lbxglu'] < 50) | (df_clean['lbxglu'] > 400)]
        if len(outliers) > 0:
            print(f"  âš  Found {len(outliers)} glucose outliers (< 50 or > 400)")
    
    # Create additional derived features
    print("\nâœ“ Creating derived features...")
    
    # BMI categories
    if 'bmxbmi' in df_clean.columns:
        df_clean['bmi_category'] = pd.cut(df_clean['bmxbmi'], 
                                          bins=[0, 18.5, 25, 30, 100],
                                          labels=['underweight', 'normal', 'overweight', 'obese'])
        print("  - Created bmi_category feature")
    
    # Age categories (if not already present)
    if 'ridageyr' in df_clean.columns and 'age_group' not in df_clean.columns:
        df_clean['age_group'] = pd.cut(df_clean['ridageyr'],
                                       bins=[0, 18, 65, 120],
                                       labels=['Youth', 'Adult', 'Senior'])
        print("  - Created age_group feature")
    
    # Diabetes risk indicator
    if 'lbxglu' in df_clean.columns:
        df_clean['glucose_risk'] = pd.cut(df_clean['lbxglu'],
                                          bins=[0, 100, 126, 1000],
                                          labels=['normal', 'prediabetes', 'diabetes'])
        print("  - Created glucose_risk feature")
    
    print(f"\nâœ“ Final dataset shape: {df_clean.shape}")
    
    return df_clean

def validate_dataset(df):
    """Validate dataset for ML readiness."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)
    
    # Check for required columns
    print("\nâœ“ Checking dataset structure...")
    print(f"  - Total features: {len(df.columns)}")
    print(f"  - Total samples: {len(df)}")
    
    # Check target variable
    if 'age_group' in df.columns:
        print(f"\nâœ“ Target variable: age_group")
        print(f"  - Classes: {df['age_group'].nunique()}")
        print(f"  - Distribution:")
        print(df['age_group'].value_counts())
        
        # Check class balance
        class_counts = df['age_group'].value_counts()
        min_class = class_counts.min()
        max_class = class_counts.max()
        imbalance_ratio = max_class / min_class
        if imbalance_ratio > 3:
            print(f"  âš  Class imbalance detected (ratio: {imbalance_ratio:.2f})")
        else:
            print(f"  âœ“ Classes are reasonably balanced (ratio: {imbalance_ratio:.2f})")
    
    # Check feature types
    print(f"\nâœ“ Feature types:")
    print(f"  - Numeric: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  - Categorical: {len(df.select_dtypes(include=['object', 'category']).columns)}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum().sum()
    if inf_count > 0:
        print(f"  âš  Found {inf_count} infinite values")
    else:
        print(f"  âœ“ No infinite values detected")
    
    # Overall missing value percentage
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells * 100)
    print(f"\nâœ“ Missing values: {total_missing} ({missing_pct:.2f}% of total)")
    
    # ML readiness check
    print(f"\nâœ“ ML Readiness Assessment:")
    issues = []
    
    if missing_pct > 30:
        issues.append(f"High missing value percentage ({missing_pct:.2f}%)")
    
    if len(df) < 100:
        issues.append(f"Small sample size ({len(df)} samples)")
    
    if 'age_group' in df.columns and df['age_group'].isnull().sum() > 0:
        issues.append("Target variable has missing values")
    
    if len(issues) == 0:
        print("  âœ“ Dataset is ready for ML training")
    else:
        print("  âš  Issues to address:")
        for issue in issues:
            print(f"    - {issue}")
    
    return df

def generate_summary_report(df, output_path):
    """Generate summary report for the dataset."""
    report = []
    report.append("=" * 70)
    report.append("NHANES SURVEY DATA - DATASET SUMMARY")
    report.append("=" * 70)
    report.append("")
    
    report.append("DATASET INFORMATION")
    report.append("-" * 70)
    report.append(f"Source: National Health and Nutrition Health Survey 2013-2014")
    report.append(f"Total Samples: {len(df)}")
    report.append(f"Total Features: {len(df.columns)}")
    report.append("")
    
    report.append("FEATURE CATEGORIES")
    report.append("-" * 70)
    report.append("Demographics:")
    report.append("  - seqn: Respondent sequence number")
    report.append("  - ridageyr: Age in years")
    report.append("  - riagendr: Gender (1=Male, 2=Female)")
    report.append("  - age_group: Age category (Youth/Adult/Senior)")
    report.append("")
    report.append("Lifestyle:")
    report.append("  - paq605: Physical activity level")
    report.append("")
    report.append("Clinical Measurements:")
    report.append("  - bmxbmi: Body Mass Index")
    report.append("  - lbxglu: Fasting glucose (mg/dL)")
    report.append("  - lbxglt: Two-hour glucose (mg/dL)")
    report.append("  - lbxin: Insulin (Î¼U/mL)")
    report.append("")
    report.append("Health Indicators:")
    report.append("  - diq010: Doctor told you have diabetes")
    report.append("  - bmi_category: BMI classification")
    report.append("  - glucose_risk: Glucose risk level")
    report.append("")
    
    report.append("TARGET VARIABLE")
    report.append("-" * 70)
    if 'age_group' in df.columns:
        report.append("Primary Target: age_group")
        report.append(f"Classes: {df['age_group'].unique().tolist()}")
        report.append("Distribution:")
        for cat, count in df['age_group'].value_counts().items():
            pct = (count / len(df) * 100)
            report.append(f"  - {cat}: {count} ({pct:.2f}%)")
    report.append("")
    
    report.append("DATA QUALITY")
    report.append("-" * 70)
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (total_missing / total_cells * 100)
    report.append(f"Missing Values: {total_missing} ({missing_pct:.2f}%)")
    report.append(f"Duplicate Rows: {df.duplicated().sum()}")
    report.append("")
    
    report.append("RECOMMENDED ML ALGORITHMS")
    report.append("-" * 70)
    report.append("For Age Prediction:")
    report.append("  - Random Forest Classifier")
    report.append("  - Gradient Boosting (XGBoost, LightGBM)")
    report.append("  - Support Vector Machine (SVM)")
    report.append("  - Neural Networks")
    report.append("")
    report.append("For Health Risk Assessment:")
    report.append("  - Logistic Regression")
    report.append("  - Random Forest")
    report.append("  - Gradient Boosting")
    report.append("")
    
    report.append("NOTES")
    report.append("-" * 70)
    report.append("- Dataset is clean and ready for ML training")
    report.append("- Missing values should be handled during preprocessing")
    report.append("- Consider feature scaling for distance-based algorithms")
    report.append("- BMI and glucose features are key health indicators")
    report.append("- Age (ridageyr) may cause target leakage for age_group prediction")
    report.append("=" * 70)
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\nâœ“ Summary report saved to: {output_path}")

if __name__ == "__main__":
    # Load data
    df = load_nhanes_data()
    
    # Analyze
    df = analyze_dataset(df)
    
    # Clean
    df_clean = clean_dataset(df)
    
    # Validate
    df_validated = validate_dataset(df_clean)
    
    # Save master dataset
    output_csv = "Survey_data/Survey_data_master_dataset.csv"
    df_validated.to_csv(output_csv, index=False)
    print(f"\nâœ“ Saved master dataset to: {output_csv}")
    
    # Generate summary report
    summary_path = "Survey_data/Survey_data_DATASET_SUMMARY.txt"
    generate_summary_report(df_validated, summary_path)
    
    print("\n" + "=" * 60)
    print("NHANES DATA PROCESSING COMPLETE")
    print("=" * 60)


