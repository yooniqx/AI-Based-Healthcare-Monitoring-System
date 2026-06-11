"""
Initial Assessment Script for BUPA Liver Disorders Dataset
This script performs initial data exploration and assessment.
"""

import pandas as pd
import numpy as np
import os

# Define column names based on bupa.names file
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
print("BUPA LIVER DISORDERS DATASET - INITIAL ASSESSMENT")
print("="*80)

# Load the data
data_path = '../bupa.data'
df = pd.read_csv(data_path, header=None, names=column_names)

print(f"\n1. DATASET OVERVIEW")
print("-"*80)
print(f"Total records: {len(df)}")
print(f"Total features: {len(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")

print(f"\n2. COLUMN INFORMATION")
print("-"*80)
print(df.info())

print(f"\n3. FIRST 10 ROWS")
print("-"*80)
print(df.head(10))

print(f"\n4. LAST 10 ROWS")
print("-"*80)
print(df.tail(10))

print(f"\n5. BASIC STATISTICS")
print("-"*80)
print(df.describe())

print(f"\n6. DATA TYPES")
print("-"*80)
for col in df.columns:
    print(f"{col:15s}: {df[col].dtype}")

print(f"\n7. MISSING VALUES")
print("-"*80)
missing = df.isnull().sum()
if missing.sum() == 0:
    print("No missing values found!")
else:
    print(missing[missing > 0])

print(f"\n8. DUPLICATE ROWS")
print("-"*80)
duplicates = df.duplicated().sum()
print(f"Total duplicate rows: {duplicates}")
if duplicates > 0:
    print("\nDuplicate rows:")
    dup_df = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())  # type: ignore
    print(dup_df)
    
    # Known duplicates from noteDuplicates.txt
    print("\nKnown duplicates (from noteDuplicates.txt):")
    print("- Rows 84 and 86: 94,58,21,18,26,2.0,2")
    print("- Rows 141 and 318: 92,80,10,26,20,6.0,1")
    print("- Rows 143 and 150: 91,63,25,26,15,6.0,1")
    print("- Rows 170 and 176: 97,71,29,22,52,8.0,1")

print(f"\n9. UNIQUE VALUES PER COLUMN")
print("-"*80)
for col in df.columns:
    n_unique = df[col].nunique()
    print(f"{col:15s}: {n_unique:4d} unique values")

print(f"\n10. VALUE RANGES")
print("-"*80)
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        print(f"{col:15s}: [{df[col].min():.2f}, {df[col].max():.2f}]")

print(f"\n11. SELECTOR FIELD DISTRIBUTION")
print("-"*80)
print(df['selector'].value_counts().sort_index())
print(f"\nSelector 1: {(df['selector'] == 1).sum()} records ({(df['selector'] == 1).sum()/len(df)*100:.1f}%)")
print(f"Selector 2: {(df['selector'] == 2).sum()} records ({(df['selector'] == 2).sum()/len(df)*100:.1f}%)")

print(f"\n12. DRINKS DISTRIBUTION")
print("-"*80)
print(f"Mean drinks per day: {df['drinks'].mean():.2f}")
print(f"Median drinks per day: {df['drinks'].median():.2f}")
print(f"Std drinks per day: {df['drinks'].std():.2f}")
print(f"\nDrinks distribution:")
print(df['drinks'].value_counts().sort_index().head(20))

print(f"\n13. POTENTIAL OUTLIERS (using IQR method)")
print("-"*80)
for col in df.columns:
    if df[col].dtype in ['int64', 'float64'] and col != 'selector':
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            print(f"{col:15s}: {len(outliers):3d} potential outliers ({len(outliers)/len(df)*100:.1f}%)")

print(f"\n14. CORRELATION ANALYSIS")
print("-"*80)
correlation_matrix = df.corr()
print("\nCorrelation with selector:")
print(correlation_matrix['selector'].sort_values(ascending=False))  # type: ignore

print(f"\n15. ZERO VALUES")
print("-"*80)
for col in df.columns:
    if df[col].dtype in ['int64', 'float64']:
        zero_count = (df[col] == 0).sum()
        if zero_count > 0:
            print(f"{col:15s}: {zero_count:3d} zero values ({zero_count/len(df)*100:.1f}%)")

# Save summary report
print(f"\n16. SAVING SUMMARY REPORT")
print("-"*80)
os.makedirs('../reports', exist_ok=True)

# Create summary dataframe
summary_data = {
    'Metric': [
        'Total Records',
        'Total Features',
        'Missing Values',
        'Duplicate Rows',
        'Selector 1 Count',
        'Selector 2 Count',
        'Mean Drinks/Day',
        'Median Drinks/Day'
    ],
    'Value': [
        len(df),
        len(df.columns),
        df.isnull().sum().sum(),
        df.duplicated().sum(),
        (df['selector'] == 1).sum(),
        (df['selector'] == 2).sum(),
        f"{df['drinks'].mean():.2f}",
        f"{df['drinks'].median():.2f}"
    ]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('../reports/initial_assessment_summary.csv', index=False)
print("Saved: reports/initial_assessment_summary.csv")

# Save detailed statistics
df.describe().to_csv('../reports/detailed_statistics.csv')
print("Saved: reports/detailed_statistics.csv")

# Save correlation matrix
correlation_matrix.to_csv('../reports/correlation_matrix.csv')
print("Saved: reports/correlation_matrix.csv")

# Save duplicate information
if duplicates > 0:
    dup_df = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())  # type: ignore
    dup_df.to_csv('../reports/duplicate_rows.csv', index=True)
    print("Saved: reports/duplicate_rows.csv")

print("\n" + "="*80)
print("INITIAL ASSESSMENT COMPLETE")
print("="*80)
