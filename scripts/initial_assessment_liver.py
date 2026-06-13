"""
BUPA Liver Disorders Dataset - Initial Assessment Script
This script analyzes the liver disorders dataset to understand its structure,
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
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'datasets' / 'liver'
REPORTS_DIR = BASE_DIR / 'reports' / 'liver'

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_liver_data():
    """Load and parse the BUPA liver disorders dataset"""
    print("\n" + "="*80)
    print("LOADING BUPA LIVER DISORDERS DATASET")
    print("="*80)
    
    # Column names from bupa.names file
    columns = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
    
    file_path = DATA_DIR / 'bupa.data'
    
    try:
        # Read the data
        df = pd.read_csv(file_path, names=columns)
        print(f"[OK] Successfully loaded {len(df)} records")
        print(f"[OK] Columns: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"[ERROR] Error loading liver data: {e}")
        return None


def analyze_dataset(df):
    """Perform comprehensive analysis on the dataset"""
    if df is None:
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING BUPA LIVER DISORDERS DATASET")
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
        'mcv': 'Mean Corpuscular Volume',
        'alkphos': 'Alkaline Phosphotase',
        'sgpt': 'Alamine Aminotransferase (ALT)',
        'sgot': 'Aspartate Aminotransferase (AST)',
        'gammagt': 'Gamma-Glutamyl Transpeptidase',
        'drinks': 'Number of half-pint equivalents of alcoholic beverages drunk per day',
        'selector': 'Field used to split data into two sets (target variable)'
    }
    
    for col, desc in descriptions.items():
        print(f"   {col:10s}: {desc}")
    
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
    print(f"\n6. TARGET VARIABLE ANALYSIS (selector)")
    if 'selector' in df.columns:
        selector_dist = df['selector'].value_counts().sort_index()
        selector_pct = (selector_dist / len(df) * 100).round(2)
        selector_df = pd.DataFrame({
            'Selector': selector_dist.index,
            'Count': selector_dist.values,
            'Percentage': selector_pct.values
        })
        print(selector_df.to_string(index=False))
        selector_df.to_csv(REPORTS_DIR / 'initial_selector_distribution.csv', index=False)
        
        # Check for class imbalance
        if len(selector_dist) > 1:
            imbalance_ratio = selector_dist.max() / selector_dist.min()
            print(f"\n   Class imbalance ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 2:
                print(f"   ⚠ Note: Moderate class imbalance detected")
    
    # Duplicates
    print(f"\n7. DUPLICATE RECORDS")
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    if duplicates > 0:
        dup_df = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))
        dup_df.to_csv(REPORTS_DIR / 'initial_duplicate_rows.csv', index=False)
        print(f"   [OK] Saved duplicate rows to initial_duplicate_rows.csv")
        print(f"   ⚠ WARNING: {duplicates} duplicate records found - should be removed during cleaning")
    
    # Data quality issues
    print(f"\n8. DATA QUALITY ISSUES")
    issues = []
    
    # Check for negative values
    for col in numeric_cols:
        if col not in ['selector']:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"   ⚠ {col}: {neg_count} negative values")
    
    # Check for zero values in drinks (valid but noteworthy)
    zero_drinks = (df['drinks'] == 0).sum()
    print(f"   ℹ drinks = 0: {zero_drinks} records ({zero_drinks/len(df)*100:.2f}%) - non-drinkers")
    
    # Check for outliers using IQR method
    outlier_summary = []
    for col in numeric_cols:
        if col != 'selector':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_summary.append({
                    'Feature': col,
                    'Outlier_Count': outliers,
                    'Outlier_Percentage': round(outliers/len(df)*100, 2),
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR
                })
                issues.append(f"   ⚠ {col}: {outliers} potential outliers ({outliers/len(df)*100:.2f}%)")
    
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df.to_csv(REPORTS_DIR / 'initial_outlier_summary.csv', index=False)
        print(f"   [OK] Saved outlier analysis to initial_outlier_summary.csv")
    
    if issues:
        for issue in issues:
            print(issue)
    else:
        print("   [OK] No obvious data quality issues detected")
    
    # Clinical reference ranges
    print(f"\n9. CLINICAL REFERENCE RANGES")
    print("   Normal ranges for liver enzymes:")
    print("   - SGPT (ALT): 7-56 U/L")
    print("   - SGOT (AST): 10-40 U/L")
    print("   - Gamma-GT: 9-48 U/L")
    print("   - Alkaline Phosphatase: 30-120 U/L")
    print("   - MCV: 80-100 fL")
    
    # Check how many values are outside normal ranges
    elevated_counts = {
        'sgpt': (df['sgpt'] > 56).sum(),
        'sgot': (df['sgot'] > 40).sum(),
        'gammagt': (df['gammagt'] > 48).sum(),
        'alkphos': (df['alkphos'] > 120).sum(),
        'mcv': (df['mcv'] > 100).sum()
    }
    
    print(f"\n   Records with elevated values:")
    for enzyme, count in elevated_counts.items():
        pct = count / len(df) * 100
        print(f"   - {enzyme}: {count} ({pct:.1f}%)")
    
    # AST/ALT ratio analysis
    print(f"\n10. AST/ALT RATIO ANALYSIS")
    df_temp = df.copy()
    df_temp['ast_alt_ratio'] = df_temp['sgot'] / df_temp['sgpt']
    high_ratio = (df_temp['ast_alt_ratio'] > 2).sum()
    print(f"   Records with AST/ALT ratio > 2: {high_ratio} ({high_ratio/len(df)*100:.2f}%)")
    print(f"   (Ratio > 2 may suggest alcoholic liver disease)")
    
    print(f"\n{'='*80}\n")


def generate_summary_report(df):
    """Generate a comprehensive summary report"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    summary = {
        'Dataset': 'BUPA Liver Disorders',
        'Total_Records': len(df),
        'Total_Features': len(df.columns),
        'Numeric_Features': len(df.select_dtypes(include=[np.number]).columns),
        'Categorical_Features': len(df.select_dtypes(include=['object']).columns),
        'Missing_Values': df.isnull().sum().sum(),
        'Duplicate_Rows': df.duplicated().sum(),
        'Memory_Usage_KB': round(df.memory_usage(deep=True).sum() / 1024, 2),
        'Selector_1_Count': (df['selector'] == 1).sum(),
        'Selector_2_Count': (df['selector'] == 2).sum()
    }
    
    summary_df = pd.DataFrame([summary])
    print(summary_df.T.to_string(header=False))
    summary_df.to_csv(REPORTS_DIR / 'initial_assessment_summary.csv', index=False)
    print(f"\n[OK] Summary report saved to initial_assessment_summary.csv")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BUPA LIVER DISORDERS DATASET - INITIAL ASSESSMENT")
    print("="*80)
    
    # Load dataset
    df = load_liver_data()
    
    if df is not None:
        # Analyze dataset
        analyze_dataset(df)
        
        # Generate summary report
        generate_summary_report(df)
        
        print("\n" + "="*80)
        print("ASSESSMENT COMPLETE")
        print("="*80)
        print(f"[OK] Reports saved to: {REPORTS_DIR}")
        print("\nNext steps:")
        print("1. Review the generated reports in the 'reports/liver' directory")
        print("2. Create a data cleaning plan based on identified issues")
        print("3. Run data_cleaning_liver.py to clean the data")
        print("="*80 + "\n")
    else:
        print("\n[ERROR] Failed to load dataset. Please check the file path.")


if __name__ == "__main__":
    main()

# Made with Bob
