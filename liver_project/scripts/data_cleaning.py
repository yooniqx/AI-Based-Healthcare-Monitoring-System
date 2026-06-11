"""
Data Cleaning Script for BUPA Liver Disorders Dataset
This script performs comprehensive data cleaning based on the data cleaning plan.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# Define column names
column_names = [
    'mcv',           # mean corpuscular volume
    'alkphos',       # alkaline phosphotase
    'sgpt',          # alamine aminotransferase
    'sgot',          # aspartate aminotransferase
    'gammagt',       # gamma-glutamyl transpeptidase
    'drinks',        # number of half-pint equivalents of alcoholic beverages drunk per day
    'selector'       # field used to split data into two sets
]

print("="*80)
print("BUPA LIVER DISORDERS DATASET - DATA CLEANING")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# PHASE 1: DATA LOADING AND VALIDATION
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: DATA LOADING AND VALIDATION")
print("="*80)

data_path = '../bupa.data'
print(f"\nLoading data from: {data_path}")
df_original = pd.read_csv(data_path, header=None, names=column_names)
print(f"[OK] Loaded {len(df_original)} records with {len(df_original.columns)} features")

# Verify data types
print("\nData types:")
for col in df_original.columns:
    print(f"  {col:15s}: {df_original[col].dtype}")

# Check for missing values
missing_count = df_original.isnull().sum().sum()
print(f"\nMissing values: {missing_count}")
if missing_count == 0:
    print("[OK] No missing values found")

# Create a copy for cleaning
df = df_original.copy()

# ============================================================================
# PHASE 2: DUPLICATE HANDLING
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: DUPLICATE HANDLING")
print("="*80)

# Identify duplicates
duplicates_mask = df.duplicated(keep='first')
n_duplicates = duplicates_mask.sum()
print(f"\nDuplicates found: {n_duplicates}")

if n_duplicates > 0:
    # Get duplicate rows before removal
    duplicate_rows = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())  # type: ignore
    print("\nDuplicate rows (all occurrences):")
    print(duplicate_rows)
    
    # Save duplicate information
    os.makedirs('../reports', exist_ok=True)
    duplicate_rows.to_csv('../reports/removed_duplicates.csv', index=True)
    print("\n[OK] Saved duplicate information to: reports/removed_duplicates.csv")
    
    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(keep='first')
    print(f"\n[OK] Removed {n_duplicates} duplicate rows")
    print(f"[OK] Records after deduplication: {len(df)}")
else:
    print("[OK] No duplicates to remove")

# ============================================================================
# PHASE 3: OUTLIER ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PHASE 3: OUTLIER ANALYSIS")
print("="*80)

outlier_info = []

for col in df.columns:
    if df[col].dtype in ['int64', 'float64'] and col != 'selector':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = outliers_mask.sum()
        
        outlier_info.append({
            'Feature': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'N_Outliers': n_outliers,
            'Percent_Outliers': f"{n_outliers/len(df)*100:.2f}%",
            'Min_Value': df[col].min(),
            'Max_Value': df[col].max()
        })
        
        if n_outliers > 0:
            print(f"\n{col}:")
            print(f"  Outliers: {n_outliers} ({n_outliers/len(df)*100:.1f}%)")
            print(f"  Range: [{df[col].min()}, {df[col].max()}]")
            print(f"  Expected range: [{lower_bound:.2f}, {upper_bound:.2f}]")

# Save outlier analysis
outlier_df = pd.DataFrame(outlier_info)
outlier_df.to_csv('../reports/outlier_analysis.csv', index=False)
print("\n[OK] Saved outlier analysis to: reports/outlier_analysis.csv")
print("\nDecision: Keeping all outliers (likely valid medical values)")

# ============================================================================
# PHASE 4: FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("PHASE 4: FEATURE ENGINEERING")
print("="*80)

# Create AST/ALT ratio (clinically significant)
df['ast_alt_ratio'] = df['sgot'] / df['sgpt']
df['ast_alt_ratio'] = df['ast_alt_ratio'].replace([np.inf, -np.inf], np.nan)
print("\n[OK] Created AST/ALT ratio (sgot/sgpt)")
print(f"  Mean ratio: {df['ast_alt_ratio'].mean():.2f}")
print(f"  Median ratio: {df['ast_alt_ratio'].median():.2f}")

# Create alcohol consumption categories
def categorize_drinks(drinks):
    if drinks == 0:
        return 'None'
    elif drinks <= 1:
        return 'Light'
    elif drinks <= 4:
        return 'Moderate'
    elif drinks <= 8:
        return 'Heavy'
    else:
        return 'Very Heavy'

df['alcohol_category'] = df['drinks'].apply(categorize_drinks)
print("\n[OK] Created alcohol consumption categories")
print(df['alcohol_category'].value_counts().sort_index())

# Create total enzyme score (sum of normalized enzymes)
enzyme_cols = ['alkphos', 'sgpt', 'sgot', 'gammagt']
for col in enzyme_cols:
    df[f'{col}_normalized'] = (df[col] - df[col].mean()) / df[col].std()

df['total_enzyme_score'] = df[[f'{col}_normalized' for col in enzyme_cols]].sum(axis=1)
print("\n[OK] Created total enzyme score")
print(f"  Mean score: {df['total_enzyme_score'].mean():.2f}")
print(f"  Std score: {df['total_enzyme_score'].std():.2f}")

# Drop temporary normalized columns
df = df.drop(columns=[f'{col}_normalized' for col in enzyme_cols])

# ============================================================================
# PHASE 5: DATA VALIDATION
# ============================================================================
print("\n" + "="*80)
print("PHASE 5: DATA VALIDATION")
print("="*80)

validation_results = []

# Check for negative values
print("\nChecking for negative values:")
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        negative_count = (df[col] < 0).sum()
        validation_results.append({
            'Check': 'Negative Values',
            'Column': col,
            'Status': 'PASS' if negative_count == 0 else 'FAIL',
            'Details': f"{negative_count} negative values"
        })
        if negative_count > 0:
            print(f"  [X] {col}: {negative_count} negative values")
        else:
            print(f"  [OK] {col}: No negative values")

# Validate selector field
print("\nValidating selector field:")
unique_selectors = df['selector'].unique()
selector_valid = all(s in [1, 2] for s in unique_selectors)
validation_results.append({
    'Check': 'Selector Values',
    'Column': 'selector',
    'Status': 'PASS' if selector_valid else 'FAIL',
    'Details': f"Unique values: {sorted(unique_selectors)}"
})
if selector_valid:
    print(f"  [OK] Selector contains only valid values: {sorted(unique_selectors)}")
else:
    print(f"  [X] Selector contains invalid values: {sorted(unique_selectors)}")

# Check data types
print("\nValidating data types:")
expected_types = {
    'mcv': 'int64',
    'alkphos': 'int64',
    'sgpt': 'int64',
    'sgot': 'int64',
    'gammagt': 'int64',
    'drinks': 'float64',
    'selector': 'int64'
}

for col, expected_type in expected_types.items():
    actual_type = str(df[col].dtype)
    type_valid = actual_type == expected_type
    validation_results.append({
        'Check': 'Data Type',
        'Column': col,
        'Status': 'PASS' if type_valid else 'FAIL',
        'Details': f"Expected: {expected_type}, Actual: {actual_type}"
    })
    if type_valid:
        print(f"  [OK] {col}: {actual_type}")
    else:
        print(f"  [X] {col}: Expected {expected_type}, got {actual_type}")

# Check for missing values in cleaned data
print("\nChecking for missing values:")
missing_after = df.isnull().sum()
for col in df.columns:
    validation_results.append({
        'Check': 'Missing Values',
        'Column': col,
        'Status': 'PASS' if missing_after[col] == 0 else 'FAIL',
        'Details': f"{missing_after[col]} missing values"
    })
    if missing_after[col] > 0:
        print(f"  [X] {col}: {missing_after[col]} missing values")
    else:
        print(f"  [OK] {col}: No missing values")

# Save validation results
validation_df = pd.DataFrame(validation_results)
validation_df.to_csv('../reports/validation_results.csv', index=False)
print("\n[OK] Saved validation results to: reports/validation_results.csv")

# ============================================================================
# PHASE 6: EXPORT CLEAN DATA
# ============================================================================
print("\n" + "="*80)
print("PHASE 6: EXPORT CLEAN DATA")
print("="*80)

# Save cleaned dataset
os.makedirs('../output', exist_ok=True)
output_path = '../output/liver_cleaned.csv'
df.to_csv(output_path, index=False)
print(f"\n[OK] Saved cleaned dataset to: {output_path}")
print(f"  Records: {len(df)}")
print(f"  Features: {len(df.columns)}")

# Generate data cleaning report
cleaning_report = {
    'Metric': [
        'Original Records',
        'Duplicates Removed',
        'Final Records',
        'Original Features',
        'Engineered Features',
        'Final Features',
        'Missing Values (Original)',
        'Missing Values (Final)',
        'Outliers Detected',
        'Outliers Removed',
        'Data Quality Score'
    ],
    'Value': [
        len(df_original),
        n_duplicates,
        len(df),
        len(df_original.columns),
        3,  # ast_alt_ratio, alcohol_category, total_enzyme_score
        len(df.columns),
        df_original.isnull().sum().sum(),
        df.isnull().sum().sum(),
        sum([info['N_Outliers'] for info in outlier_info]),
        0,  # We kept all outliers
        f"{(1 - n_duplicates/len(df_original)) * 100:.2f}%"
    ]
}

report_df = pd.DataFrame(cleaning_report)
report_df.to_csv('../reports/data_cleaning_report.csv', index=False)
print(f"\n[OK] Saved cleaning report to: reports/data_cleaning_report.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DATA CLEANING SUMMARY")
print("="*80)

print(f"\nOriginal dataset: {len(df_original)} records, {len(df_original.columns)} features")
print(f"Cleaned dataset: {len(df)} records, {len(df.columns)} features")
print(f"\nChanges made:")
print(f"  • Removed {n_duplicates} duplicate rows")
print(f"  • Added 3 engineered features:")
print(f"    - ast_alt_ratio (AST/ALT ratio)")
print(f"    - alcohol_category (categorical)")
print(f"    - total_enzyme_score (composite)")
print(f"  • Kept all outliers (valid medical values)")
print(f"  • No missing values in original or cleaned data")

print(f"\nFiles generated:")
print(f"  • output/liver_cleaned.csv")
print(f"  • reports/data_cleaning_report.csv")
print(f"  • reports/removed_duplicates.csv")
print(f"  • reports/outlier_analysis.csv")
print(f"  • reports/validation_results.csv")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "="*80)
print("DATA CLEANING COMPLETE")
print("="*80)

