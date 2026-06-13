"""
Heart Disease Dataset - Initial Assessment Script
This script analyzes the heart disease dataset to understand its structure,
identify data quality issues, and prepare for cleaning.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except AttributeError:
        pass  # Python < 3.7

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'datasets' / 'heart'
REPORTS_DIR = BASE_DIR / 'reports' / 'heart'

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_heart_data():
    """Load and parse the heart disease dataset"""
    print("\n" + "="*80)
    print("LOADING HEART DISEASE DATASET")
    print("="*80)
    
    file_path = DATA_DIR / 'heart_master_dataset.csv'
    
    try:
        # Read the data
        df = pd.read_csv(file_path)
        print(f"[OK] Successfully loaded {len(df)} records")
        print(f"[OK] Columns: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"[ERROR] Error loading heart data: {e}")
        return None


def analyze_dataset(df):
    """Perform comprehensive analysis on the dataset"""
    if df is None:
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING HEART DISEASE DATASET")
    print(f"{'='*80}")
    
    # Basic info
    print(f"\n1. BASIC INFORMATION")
    print(f"   Shape: {df.shape}")
    print(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # Data types
    print(f"\n2. DATA TYPES")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    # Column descriptions
    print(f"\n3. COLUMN DESCRIPTIONS")
    descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (1 = male; 0 = female)',
        'chest_pain_type': 'Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)',
        'resting_bp': 'Resting blood pressure (in mm Hg)',
        'cholesterol': 'Serum cholesterol in mg/dl',
        'fasting_blood_sugar': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
        'resting_ecg': 'Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)',
        'max_heart_rate': 'Maximum heart rate achieved',
        'exercise_induced_angina': 'Exercise induced angina (1 = yes; 0 = no)',
        'st_depression': 'ST depression induced by exercise relative to rest',
        'st_slope': 'Slope of the peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)',
        'num_major_vessels': 'Number of major vessels (0-3) colored by fluoroscopy',
        'thalassemia': 'Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)',
        'target': 'Heart disease diagnosis (0 = no disease, 1 = disease)'
    }
    
    for col in df.columns:
        desc = descriptions.get(col, 'No description available')
        print(f"   {col:25s}: {desc}")
    
    # Missing values
    print(f"\n4. MISSING VALUES")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_data = []
    for col in missing.index:
        if missing[col] > 0:
            missing_data.append({
                'Column': col,
                'Missing_Count': int(missing[col]),
                'Missing_Percentage': float(missing_pct[col])
            })
    
    if missing_data:
        missing_df = pd.DataFrame(missing_data)
        missing_df = missing_df.sort_values('Missing_Count', ascending=False)
        print(f"   Columns with missing values: {len(missing_df)}")
        print(missing_df.to_string(index=False))
        missing_df.to_csv(REPORTS_DIR / 'initial_missing_values.csv', index=False)
    else:
        print("   [OK] No missing values found")
        pd.DataFrame(columns=['Column', 'Missing_Count', 'Missing_Percentage']).to_csv(
            REPORTS_DIR / 'initial_missing_values.csv', index=False
        )
    
    # Numerical columns analysis
    print(f"\n5. NUMERICAL COLUMNS ANALYSIS")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"   Number of numerical columns: {len(numeric_cols)}")
        stats_df = df[numeric_cols].describe().T
        stats_df.to_csv(REPORTS_DIR / 'initial_numerical_stats.csv')
        print(f"\n   Summary Statistics:")
        print(stats_df.to_string())
    
    # Target variable analysis
    print(f"\n6. TARGET VARIABLE ANALYSIS")
    if 'target' in df.columns:
        target_dist = df['target'].value_counts().sort_index()
        target_pct = (target_dist / len(df) * 100).round(2)
        target_df = pd.DataFrame({
            'Target': target_dist.index,
            'Count': target_dist.values,
            'Percentage': target_pct.values
        })
        print(target_df.to_string(index=False))
        target_df.to_csv(REPORTS_DIR / 'initial_target_distribution.csv', index=False)
        
        # Check for class imbalance
        if len(target_dist) > 1:
            imbalance_ratio = target_dist.max() / target_dist.min()
            print(f"\n   Class imbalance ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 2:
                print(f"   ⚠ Note: Moderate class imbalance detected")
            else:
                print(f"   ✓ Classes are relatively balanced")
    
    # Duplicates
    print(f"\n7. DUPLICATE RECORDS")
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    if duplicates > 0:
        dup_df = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))
        dup_df.to_csv(REPORTS_DIR / 'initial_duplicate_rows.csv', index=False)
        print(f"   [OK] Saved duplicate rows to initial_duplicate_rows.csv")
        print(f"   ⚠ WARNING: {duplicates} duplicate records found - should be removed during cleaning")
    else:
        print(f"   [OK] No duplicate records found")
    
    # Outlier detection (IQR method)
    print(f"\n8. OUTLIER DETECTION (IQR METHOD)")
    outlier_summary = {}
    for col in numeric_cols:
        if col != 'target':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_summary[col] = {
                    'count': outliers,
                    'percentage': (outliers / len(df) * 100).round(2)
                }
    
    if outlier_summary:
        print(f"   Columns with outliers: {len(outlier_summary)}")
        outlier_data = []
        for col, info in sorted(outlier_summary.items(), key=lambda x: x[1]['count'], reverse=True):
            print(f"   {col:25s}: {info['count']:4d} outliers ({info['percentage']:5.2f}%)")
            outlier_data.append({
                'Column': col,
                'Outlier_Count': info['count'],
                'Outlier_Percentage': info['percentage']
            })
        outlier_df = pd.DataFrame(outlier_data)
        outlier_df.to_csv(REPORTS_DIR / 'initial_outliers.csv', index=False)
    else:
        print(f"   [OK] No outliers detected")
    
    # Data quality issues
    print(f"\n9. DATA QUALITY CHECKS")
    quality_issues = []
    
    # Check for zero values in cholesterol (likely missing data)
    if 'cholesterol' in df.columns:
        zero_chol = (df['cholesterol'] == 0).sum()
        if zero_chol > 0:
            quality_issues.append(f"Cholesterol: {zero_chol} zero values (likely missing data)")
    
    # Check for unrealistic age values
    if 'age' in df.columns:
        invalid_age = ((df['age'] < 20) | (df['age'] > 100)).sum()
        if invalid_age > 0:
            quality_issues.append(f"Age: {invalid_age} unrealistic values (< 20 or > 100)")
    
    # Check for unrealistic blood pressure
    if 'resting_bp' in df.columns:
        invalid_bp = ((df['resting_bp'] < 80) | (df['resting_bp'] > 200)).sum()
        if invalid_bp > 0:
            quality_issues.append(f"Resting BP: {invalid_bp} unrealistic values (< 80 or > 200)")
    
    # Check for unrealistic heart rate
    if 'max_heart_rate' in df.columns:
        invalid_hr = ((df['max_heart_rate'] < 60) | (df['max_heart_rate'] > 220)).sum()
        if invalid_hr > 0:
            quality_issues.append(f"Max heart rate: {invalid_hr} unrealistic values (< 60 or > 220)")
    
    if quality_issues:
        print(f"   ⚠ Data quality issues found:")
        for issue in quality_issues:
            print(f"      - {issue}")
    else:
        print(f"   [OK] No major data quality issues detected")
    
    # Categorical features analysis
    print(f"\n10. CATEGORICAL FEATURES ANALYSIS")
    categorical_features = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 'resting_ecg', 
                           'exercise_induced_angina', 'st_slope', 'num_major_vessels', 'thalassemia']
    
    for col in categorical_features:
        if col in df.columns:
            unique_vals = df[col].nunique()
            print(f"   {col:25s}: {unique_vals} unique values")
            value_counts = df[col].value_counts().sort_index()
            for val, count in value_counts.items():
                pct = (count / len(df) * 100)
                print(f"      {val}: {count:4d} ({pct:5.2f}%)")


def generate_summary_report(df):
    """Generate a comprehensive summary report"""
    print(f"\n{'='*80}")
    print(f"GENERATING SUMMARY REPORT")
    print(f"{'='*80}")
    
    summary = []
    summary.append("="*80)
    summary.append("HEART DISEASE DATASET - INITIAL ASSESSMENT SUMMARY")
    summary.append("="*80)
    summary.append("")
    
    # Dataset overview
    summary.append("DATASET OVERVIEW")
    summary.append("-"*80)
    summary.append(f"Total Records: {len(df)}")
    summary.append(f"Total Features: {len(df.columns)}")
    summary.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    summary.append("")
    
    # Data quality summary
    summary.append("DATA QUALITY SUMMARY")
    summary.append("-"*80)
    missing_count = df.isnull().sum().sum()
    duplicate_count = df.duplicated().sum()
    summary.append(f"Missing Values: {missing_count}")
    summary.append(f"Duplicate Records: {duplicate_count}")
    summary.append("")
    
    # Target distribution
    if 'target' in df.columns:
        summary.append("TARGET DISTRIBUTION")
        summary.append("-"*80)
        target_dist = df['target'].value_counts().sort_index()
        for val, count in target_dist.items():
            pct = (count / len(df) * 100)
            summary.append(f"Class {val}: {count:4d} ({pct:5.2f}%)")
        summary.append("")
    
    # Recommendations
    summary.append("RECOMMENDATIONS FOR DATA CLEANING")
    summary.append("-"*80)
    summary.append("1. Remove duplicate records if any")
    summary.append("2. Handle zero cholesterol values (likely missing data)")
    summary.append("3. Validate and clean unrealistic values in age, BP, and heart rate")
    summary.append("4. Check for outliers and decide on treatment strategy")
    summary.append("5. Verify categorical feature encodings")
    summary.append("")
    summary.append("="*80)
    
    # Save summary report
    report_path = REPORTS_DIR / 'INITIAL_ASSESSMENT_SUMMARY.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    print(f"\n[OK] Summary report saved to {report_path}")
    
    # Print summary to console
    print('\n'.join(summary))


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("HEART DISEASE DATASET - INITIAL ASSESSMENT")
    print("="*80)
    
    # Load data
    df = load_heart_data()
    
    if df is not None:
        # Analyze dataset
        analyze_dataset(df)
        
        # Generate summary report
        generate_summary_report(df)
        
        print("\n" + "="*80)
        print("INITIAL ASSESSMENT COMPLETE")
        print("="*80)
        print(f"\nReports saved to: {REPORTS_DIR}")
        print("\nNext steps:")
        print("1. Review the assessment reports")
        print("2. Run data cleaning script (data_cleaning_heart.py)")
        print("3. Perform exploratory data analysis (perform_eda_heart.py)")
        print("4. Execute feature engineering (feature_engineering_heart.py)")
        print("="*80 + "\n")
    else:
        print("\n[ERROR] Failed to load dataset. Please check the file path and try again.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Initial assessment failed: {e}")
        import traceback
        traceback.print_exc()

# Made with Bob
