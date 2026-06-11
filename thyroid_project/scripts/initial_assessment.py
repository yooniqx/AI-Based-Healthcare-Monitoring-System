"""
Thyroid Disease Dataset - Initial Assessment Script
This script analyzes the thyroid disease datasets to understand their structure,
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
DATA_DIR = BASE_DIR
REPORTS_DIR = BASE_DIR / 'reports'

# Ensure reports directory exists
REPORTS_DIR.mkdir(exist_ok=True)

def load_hypothyroid_data():
    """Load and parse the hypothyroid dataset"""
    print("\n" + "="*80)
    print("LOADING HYPOTHYROID DATASET")
    print("="*80)
    
    # Column names from hypothyroid.names file
    columns = [
        'class', 'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
        'on_antithyroid_medication', 'thyroid_surgery', 'query_hypothyroid',
        'query_hyperthyroid', 'pregnant', 'sick', 'tumor', 'lithium',
        'goitre', 'TSH_measured', 'TSH', 'T3_measured', 'T3',
        'TT4_measured', 'TT4', 'T4U_measured', 'T4U', 'FTI_measured',
        'FTI', 'TBG_measured', 'TBG'
    ]
    
    file_path = DATA_DIR / 'hypothyroid.data'
    
    try:
        # Read the data
        df = pd.read_csv(file_path, names=columns, na_values='?')
        print(f"[OK] Successfully loaded {len(df)} records")
        print(f"[OK] Columns: {len(df.columns)}")
        return df, 'hypothyroid'
    except Exception as e:
        print(f"[ERROR] Error loading hypothyroid data: {e}")
        return None, None

def load_ann_thyroid_data():
    """Load and parse the ANN thyroid dataset"""
    print("\n" + "="*80)
    print("LOADING ANN-THYROID DATASET")
    print("="*80)
    
    # Based on ann-thyroid.names: 21 attributes (15 binary, 6 continuous) + 1 class
    columns = [
        'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
        'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
        'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
        'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
        'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'class'
    ]
    
    train_path = DATA_DIR / 'ann-train.data'
    test_path = DATA_DIR / 'ann-test.data'
    
    try:
        # Read training data
        df_train = pd.read_csv(train_path, sep=' ', names=columns, skipinitialspace=True)
        print(f"[OK] Successfully loaded {len(df_train)} training records")
        
        # Read test data
        df_test = pd.read_csv(test_path, sep=' ', names=columns, skipinitialspace=True)
        print(f"[OK] Successfully loaded {len(df_test)} test records")
        
        # Combine datasets
        df_train['dataset'] = 'train'
        df_test['dataset'] = 'test'
        df = pd.concat([df_train, df_test], ignore_index=True)
        print(f"[OK] Combined dataset: {len(df)} total records")
        
        return df, 'ann-thyroid'
    except Exception as e:
        print(f"[ERROR] Error loading ANN thyroid data: {e}")
        return None, None

def load_new_thyroid_data():
    """Load and parse the new-thyroid dataset"""
    print("\n" + "="*80)
    print("LOADING NEW-THYROID DATASET")
    print("="*80)
    
    # Column names from new-thyroid.names file
    columns = [
        'class', 'T3_resin_uptake', 'total_serum_thyroxin',
        'total_serum_triiodothyronine', 'basal_TSH', 'max_TSH_difference'
    ]
    
    file_path = DATA_DIR / 'new-thyroid.data'
    
    try:
        df = pd.read_csv(file_path, names=columns)
        print(f"[OK] Successfully loaded {len(df)} records")
        print(f"[OK] Columns: {len(df.columns)}")
        return df, 'new-thyroid'
    except Exception as e:
        print(f"[ERROR] Error loading new-thyroid data: {e}")
        return None, None

def analyze_dataset(df, dataset_name):
    """Perform comprehensive analysis on a dataset"""
    if df is None:
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING {dataset_name.upper()} DATASET")
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
    
    # Missing values
    print(f"\n3. MISSING VALUES")
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
        missing_df.to_csv(REPORTS_DIR / f'{dataset_name}_missing_values.csv', index=False)
    else:
        print("   [OK] No missing values found")
        # Save empty report
        pd.DataFrame(columns=['Column', 'Missing_Count', 'Missing_Percentage']).to_csv(
            REPORTS_DIR / f'{dataset_name}_missing_values.csv', index=False
        )
    
    # Numerical columns analysis
    print(f"\n4. NUMERICAL COLUMNS ANALYSIS")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"   Number of numerical columns: {len(numeric_cols)}")
        stats_df = df[numeric_cols].describe().T
        stats_df.to_csv(REPORTS_DIR / f'{dataset_name}_numerical_stats.csv')
        print(f"   [OK] Saved statistics to {dataset_name}_numerical_stats.csv")
    
    # Categorical columns analysis
    print(f"\n5. CATEGORICAL COLUMNS ANALYSIS")
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print(f"   Number of categorical columns: {len(cat_cols)}")
        cat_info = []
        for col in cat_cols:
            unique_count = df[col].nunique()
            top_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A'
            cat_info.append({
                'Column': col,
                'Unique_Values': unique_count,
                'Most_Common': top_value,
                'Most_Common_Count': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            })
        cat_df = pd.DataFrame(cat_info)
        print(cat_df.to_string(index=False))
        cat_df.to_csv(REPORTS_DIR / f'{dataset_name}_categorical_info.csv', index=False)
    
    # Class distribution (if class column exists)
    if 'class' in df.columns:
        print(f"\n6. CLASS DISTRIBUTION")
        class_dist = df['class'].value_counts()
        class_pct = (class_dist / len(df) * 100).round(2)
        class_df = pd.DataFrame({
            'Class': class_dist.index,
            'Count': class_dist.values,
            'Percentage': class_pct.values
        })
        print(class_df.to_string(index=False))
        class_df.to_csv(REPORTS_DIR / f'{dataset_name}_class_distribution.csv', index=False)
    
    # Duplicates
    print(f"\n7. DUPLICATE RECORDS")
    duplicates = df.duplicated().sum()
    print(f"   Duplicate rows: {duplicates} ({duplicates/len(df)*100:.2f}%)")
    
    # Data quality issues
    print(f"\n8. DATA QUALITY ISSUES")
    issues = []
    
    # Check for negative values in columns that shouldn't have them
    for col in numeric_cols:
        if col not in ['age']:  # age can be negative in some encodings
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                issues.append(f"   ⚠ {col}: {neg_count} negative values")
    
    # Check for outliers using IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        if outliers > 0:
            issues.append(f"   ⚠ {col}: {outliers} potential outliers ({outliers/len(df)*100:.2f}%)")
    
    if issues:
        for issue in issues[:10]:  # Show first 10 issues
            print(issue)
        if len(issues) > 10:
            print(f"   ... and {len(issues) - 10} more issues")
    else:
        print("   [OK] No obvious data quality issues detected")
    
    print(f"\n{'='*80}\n")

def generate_summary_report():
    """Generate a summary report of all datasets"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    summary = []
    
    # Check which datasets were successfully loaded
    datasets = [
        ('hypothyroid', 'hypothyroid.data'),
        ('ann-thyroid', 'ann-train.data'),
        ('new-thyroid', 'new-thyroid.data')
    ]
    
    for name, file in datasets:
        file_path = DATA_DIR / file
        if file_path.exists():
            summary.append({
                'Dataset': name,
                'Status': 'Available',
                'File': file
            })
        else:
            summary.append({
                'Dataset': name,
                'Status': 'Not Found',
                'File': file
            })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(REPORTS_DIR / 'datasets_summary.csv', index=False)
    print(f"\n[OK] Summary report saved to datasets_summary.csv")

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("THYROID DISEASE DATASET - INITIAL ASSESSMENT")
    print("="*80)
    
    # Load and analyze each dataset
    datasets = []
    
    # 1. Hypothyroid dataset
    df_hypo, name_hypo = load_hypothyroid_data()
    if df_hypo is not None:
        datasets.append((df_hypo, name_hypo))
        analyze_dataset(df_hypo, name_hypo)
    
    # 2. ANN-Thyroid dataset
    df_ann, name_ann = load_ann_thyroid_data()
    if df_ann is not None:
        datasets.append((df_ann, name_ann))
        analyze_dataset(df_ann, name_ann)
    
    # 3. New-Thyroid dataset
    df_new, name_new = load_new_thyroid_data()
    if df_new is not None:
        datasets.append((df_new, name_new))
        analyze_dataset(df_new, name_new)
    
    # Generate summary report
    generate_summary_report()
    
    print("\n" + "="*80)
    print("ASSESSMENT COMPLETE")
    print("="*80)
    print(f"[OK] Analyzed {len(datasets)} datasets")
    print(f"[OK] Reports saved to: {REPORTS_DIR}")
    print("\nNext steps:")
    print("1. Review the generated reports in the 'reports' directory")
    print("2. Create a data cleaning plan based on identified issues")
    print("3. Implement data cleaning scripts")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

