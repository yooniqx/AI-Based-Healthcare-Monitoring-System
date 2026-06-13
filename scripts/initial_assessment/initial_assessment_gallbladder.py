"""
Gallbladder Disease Dataset - Initial Assessment Script
This script analyzes the gallbladder dataset to understand its structure,
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
DATA_DIR = BASE_DIR / 'datasets' / 'gallbladder'
REPORTS_DIR = BASE_DIR / 'reports' / 'gallbladder'

# Ensure reports directory exists
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_gallbladder_data():
    """Load and parse the gallbladder dataset"""
    print("\n" + "="*80)
    print("LOADING GALLBLADDER DATASET")
    print("="*80)
    
    file_path = DATA_DIR / 'dataset-uci.csv'
    
    try:
        # Read the data
        df = pd.read_csv(file_path)
        print(f"[OK] Successfully loaded {len(df)} records")
        print(f"[OK] Columns: {len(df.columns)}")
        return df
    except Exception as e:
        print(f"[ERROR] Error loading gallbladder data: {e}")
        return None


def analyze_dataset(df):
    """Perform comprehensive analysis on the dataset"""
    if df is None:
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING GALLBLADDER DATASET")
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
    
    # Column names analysis
    print(f"\n3. COLUMN NAMES")
    print(f"   Original column names (with special characters):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
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
        # Save missing values report
        missing_df.to_csv(REPORTS_DIR / 'initial_missing_values.csv', index=False)
    else:
        print("   [OK] No missing values found")
        # Save empty report
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
        print(f"   [OK] Saved statistics to initial_numerical_stats.csv")
        
        # Show summary
        print(f"\n   Summary Statistics (first 10 features):")
        print(stats_df.head(10).to_string())
    
    # Target variable analysis
    print(f"\n6. TARGET VARIABLE ANALYSIS")
    target_col = 'Gallstone Status'
    if target_col in df.columns:
        target_dist = df[target_col].value_counts()
        target_pct = (target_dist / len(df) * 100).round(2)
        target_df = pd.DataFrame({
            'Class': target_dist.index,
            'Count': target_dist.values,
            'Percentage': target_pct.values
        })
        print(target_df.to_string(index=False))
        target_df.to_csv(REPORTS_DIR / 'initial_target_distribution.csv', index=False)
        
        # Check for class imbalance
        if len(target_dist) > 1:
            imbalance_ratio = target_dist.max() / target_dist.min()
            print(f"\n   Class imbalance ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 3:
                print(f"   âš  WARNING: Significant class imbalance detected!")
    
    # Duplicates
    print(f"\n7. DUPLICATE RECORDS")
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    if duplicates > 0:
        dup_df = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))
        dup_df.to_csv(REPORTS_DIR / 'initial_duplicate_rows.csv', index=False)
        print(f"   [OK] Saved duplicate rows to initial_duplicate_rows.csv")
    
    # Data quality issues
    print(f"\n8. DATA QUALITY ISSUES")
    issues = []
    
    # Check for negative values in columns that shouldn't have them
    for col in numeric_cols:
        if col not in ['Gallstone Status']:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"   âš  {col}: {neg_count} negative values")
    
    # Check for outliers using IQR method
    outlier_summary = []
    for col in numeric_cols:
        if col != 'Gallstone Status':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                outlier_summary.append({
                    'Feature': col,
                    'Outlier_Count': outliers,
                    'Outlier_Percentage': round(outliers/len(df)*100, 2)
                })
                issues.append(f"   âš  {col}: {outliers} potential outliers ({outliers/len(df)*100:.2f}%)")
    
    if outlier_summary:
        outlier_df = pd.DataFrame(outlier_summary)
        outlier_df.to_csv(REPORTS_DIR / 'initial_outlier_summary.csv', index=False)
    
    if issues:
        for issue in issues[:15]:  # Show first 15 issues
            print(issue)
        if len(issues) > 15:
            print(f"   ... and {len(issues) - 15} more issues")
    else:
        print("   [OK] No obvious data quality issues detected")
    
    # Feature categories
    print(f"\n9. FEATURE CATEGORIES")
    categories = {
        'Demographics': ['Age', 'Gender'],
        'Medical History': ['Comorbidity', 'Coronary Artery Disease (CAD)', 
                           'Hypothyroidism', 'Hyperlipidemia', 'Diabetes Mellitus (DM)'],
        'Body Measurements': ['Height', 'Weight', 'Body Mass Index (BMI)'],
        'Body Composition': ['Total Body Water (TBW)', 'Extracellular Water (ECW)', 
                            'Intracellular Water (ICW)', 'Extracellular Fluid/Total Body Water (ECF/TBW)',
                            'Total Body Fat Ratio (TBFR) (%)', 'Lean Mass (LM) (%)', 
                            'Body Protein Content (Protein) (%)', 'Visceral Fat Rating (VFR)',
                            'Bone Mass (BM)', 'Muscle Mass (MM)', 'Obesity (%)', 
                            'Total Fat Content (TFC)', 'Visceral Fat Area (VFA)', 
                            'Visceral Muscle Area (VMA) (Kg)', 'Hepatic Fat Accumulation (HFA)'],
        'Blood Tests': ['Glucose', 'Total Cholesterol (TC)', 'Low Density Lipoprotein (LDL)',
                       'High Density Lipoprotein (HDL)', 'Triglyceride',
                       'Aspartat Aminotransferaz (AST)', 'Alanin Aminotransferaz (ALT)',
                       'Alkaline Phosphatase (ALP)', 'Creatinine', 
                       'Glomerular Filtration Rate (GFR)', 'C-Reactive Protein (CRP)',
                       'Hemoglobin (HGB)', 'Vitamin D']
    }
    
    for category, features in categories.items():
        available = [f for f in features if f in df.columns]
        print(f"   {category}: {len(available)}/{len(features)} features")
    
    print(f"\n{'='*80}\n")


def generate_summary_report(df):
    """Generate a comprehensive summary report"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    summary = {
        'Dataset': 'Gallbladder Disease',
        'Total_Records': len(df),
        'Total_Features': len(df.columns),
        'Numeric_Features': len(df.select_dtypes(include=[np.number]).columns),
        'Categorical_Features': len(df.select_dtypes(include=['object']).columns),
        'Missing_Values': df.isnull().sum().sum(),
        'Duplicate_Rows': df.duplicated().sum(),
        'Memory_Usage_KB': round(df.memory_usage(deep=True).sum() / 1024, 2)
    }
    
    summary_df = pd.DataFrame([summary])
    print(summary_df.T.to_string(header=False))
    summary_df.to_csv(REPORTS_DIR / 'initial_assessment_summary.csv', index=False)
    print(f"\n[OK] Summary report saved to initial_assessment_summary.csv")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("GALLBLADDER DISEASE DATASET - INITIAL ASSESSMENT")
    print("="*80)
    
    # Load dataset
    df = load_gallbladder_data()
    
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
        print("1. Review the generated reports in the 'reports/gallbladder' directory")
        print("2. Create a data cleaning plan based on identified issues")
        print("3. Run data_cleaning_gallbladder.py to clean the data")
        print("="*80 + "\n")
    else:
        print("\n[ERROR] Failed to load dataset. Please check the file path.")


if __name__ == "__main__":
    main()


