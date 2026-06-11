"""
Phase 2: Initial Data Assessment
This script performs comprehensive data quality assessment on the gallstone dataset.
"""

import pandas as pd
import numpy as np
import os
import sys

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

def load_data(file_path):
    """Load the CSV data"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"\n[OK] Loading file: {file_path}")
        df = pd.read_csv(file_path)
        print(f"[OK] Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        
        return df
    
    except Exception as e:
        print(f"[ERROR] Error loading file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def assess_missing_values(df):
    """Identify and quantify missing values"""
    print("\n" + "=" * 80)
    print("MISSING VALUES ASSESSMENT")
    print("=" * 80)
    
    # Count missing values
    missing_counts = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    missing_report = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': missing_counts.values,
        'Missing_Percentage': missing_percentage.values,
        'Data_Type': df.dtypes.values
    })
    
    # Sort by missing percentage
    missing_report = missing_report.sort_values('Missing_Percentage', ascending=False)
    
    # Display columns with missing values
    columns_with_missing = missing_report[missing_report['Missing_Count'] > 0]
    
    if len(columns_with_missing) > 0:
        print(f"\n⚠️  Found {len(columns_with_missing)} columns with missing values:")
        print("-" * 80)
        print(columns_with_missing.to_string(index=False))
        
        # Summary statistics
        total_missing = missing_counts.sum()
        total_cells = df.shape[0] * df.shape[1]
        overall_missing_pct = (total_missing / total_cells) * 100
        
        print(f"\n📊 Missing Values Summary:")
        print(f"   Total missing cells: {total_missing:,}")
        print(f"   Total cells: {total_cells:,}")
        print(f"   Overall missing percentage: {overall_missing_pct:.2f}%")
        
        # Categorize by severity
        critical = columns_with_missing[columns_with_missing['Missing_Percentage'] > 50]
        high = columns_with_missing[(columns_with_missing['Missing_Percentage'] > 20) & 
                                   (columns_with_missing['Missing_Percentage'] <= 50)]
        moderate = columns_with_missing[(columns_with_missing['Missing_Percentage'] > 5) & 
                                       (columns_with_missing['Missing_Percentage'] <= 20)]
        low = columns_with_missing[columns_with_missing['Missing_Percentage'] <= 5]
        
        print(f"\n🔴 Critical (>50% missing): {len(critical)} columns")
        if len(critical) > 0:
            print(f"   {', '.join(critical['Column'].tolist())}")
        
        print(f"🟠 High (20-50% missing): {len(high)} columns")
        if len(high) > 0:
            print(f"   {', '.join(high['Column'].tolist())}")
        
        print(f"🟡 Moderate (5-20% missing): {len(moderate)} columns")
        if len(moderate) > 0:
            print(f"   {', '.join(moderate['Column'].tolist())}")
        
        print(f"🟢 Low (<5% missing): {len(low)} columns")
        if len(low) > 0:
            print(f"   {', '.join(low['Column'].tolist())}")
    else:
        print("\n✅ No missing values found in the dataset!")
    
    return missing_report

def assess_duplicates(df):
    """Identify duplicate records"""
    print("\n" + "=" * 80)
    print("DUPLICATE RECORDS ASSESSMENT")
    print("=" * 80)
    
    # Check for complete duplicates
    duplicate_count = df.duplicated().sum()
    duplicate_percentage = (duplicate_count / len(df)) * 100
    
    print(f"\n📋 Duplicate Analysis:")
    print(f"   Total rows: {len(df):,}")
    print(f"   Duplicate rows: {duplicate_count:,}")
    print(f"   Duplicate percentage: {duplicate_percentage:.2f}%")
    
    if duplicate_count > 0:
        print(f"\n⚠️  Found {duplicate_count} duplicate rows")
        
        # Show sample duplicates
        duplicates = df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist())
        print(f"\n📄 Sample of duplicate records (first 10):")
        print("-" * 80)
        print(duplicates.head(10))
    else:
        print("\n✅ No duplicate rows found!")
    
    return duplicate_count

def assess_data_types(df):
    """Analyze data types and potential issues"""
    print("\n" + "=" * 80)
    print("DATA TYPES ASSESSMENT")
    print("=" * 80)
    
    # Data type summary
    dtype_summary = df.dtypes.value_counts()
    
    print(f"\n📊 Data Types Distribution:")
    print("-" * 80)
    for dtype, count in dtype_summary.items():
        print(f"   {str(dtype):15s}: {count:2d} columns ({count/len(df.columns)*100:.1f}%)")
    
    # Detailed column analysis
    print(f"\n📋 Column-by-Column Analysis:")
    print("-" * 80)
    
    type_report = []
    for col in df.columns:
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        unique_pct = (unique_count / len(df)) * 100
        
        # Check if numeric column has non-numeric values
        potential_issue = ""
        if dtype == 'object':
            # Check if it could be numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                potential_issue = "Could be numeric"
            except:
                pass
        
        type_report.append({
            'Column': col,
            'Type': str(dtype),
            'Unique_Values': unique_count,
            'Unique_%': f"{unique_pct:.1f}",
            'Note': potential_issue
        })
    
    type_df = pd.DataFrame(type_report)
    print(type_df.to_string(index=False))
    
    return type_df

def assess_value_ranges(df):
    """Analyze value ranges for numerical columns"""
    print("\n" + "=" * 80)
    print("VALUE RANGES ASSESSMENT (Numerical Columns)")
    print("=" * 80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    range_df = None
    
    if len(numerical_cols) > 0:
        print(f"\n📊 Found {len(numerical_cols)} numerical columns")
        print("-" * 80)
        
        range_report = []
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                range_report.append({
                    'Column': col,
                    'Min': f"{col_data.min():.2f}",
                    'Max': f"{col_data.max():.2f}",
                    'Mean': f"{col_data.mean():.2f}",
                    'Median': f"{col_data.median():.2f}",
                    'Std': f"{col_data.std():.2f}",
                    'Zeros': (col_data == 0).sum(),
                    'Negatives': (col_data < 0).sum()
                })
        
        range_df = pd.DataFrame(range_report)
        print(range_df.to_string(index=False))
    else:
        print("\n[!] No numerical columns found")
    
    return range_df

def assess_categorical_values(df):
    """Analyze categorical columns"""
    print("\n" + "=" * 80)
    print("CATEGORICAL VALUES ASSESSMENT")
    print("=" * 80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        print(f"\n📊 Found {len(categorical_cols)} categorical columns")
        print("-" * 80)
        
        for col in categorical_cols:
            unique_values = df[col].dropna().unique()
            value_counts = df[col].value_counts()
            
            print(f"\n📋 Column: {col}")
            print(f"   Unique values: {len(unique_values)}")
            print(f"   Value distribution:")
            for val, count in value_counts.head(10).items():
                pct = (count / len(df)) * 100
                print(f"      '{val}': {count} ({pct:.1f}%)")
            
            if len(value_counts) > 10:
                print(f"      ... and {len(value_counts) - 10} more values")
    else:
        print("\n✅ No categorical columns found")

def generate_summary_report(df, missing_report, duplicate_count):
    """Generate overall data quality summary"""
    print("\n" + "=" * 80)
    print("DATA QUALITY SUMMARY REPORT")
    print("=" * 80)
    
    # Calculate quality scores
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    data_completeness = ((total_cells - missing_cells) / total_cells) * 100
    
    duplicate_score = 100 - (duplicate_count / len(df) * 100)
    
    print(f"\n📊 Overall Data Quality Metrics:")
    print("-" * 80)
    print(f"   Dataset Size: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"   Data Completeness: {data_completeness:.2f}%")
    print(f"   Duplicate-Free Score: {duplicate_score:.2f}%")
    
    # Issues summary
    columns_with_missing = len(missing_report[missing_report['Missing_Count'] > 0])
    
    print(f"\n⚠️  Issues Found:")
    print(f"   Columns with missing values: {columns_with_missing}")
    print(f"   Duplicate rows: {duplicate_count}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if columns_with_missing > 0:
        print(f"   ✓ Handle missing values in {columns_with_missing} columns")
    if duplicate_count > 0:
        print(f"   ✓ Remove {duplicate_count} duplicate rows")
    
    critical_missing = missing_report[missing_report['Missing_Percentage'] > 50]
    if len(critical_missing) > 0:
        print(f"   ✓ Consider dropping {len(critical_missing)} columns with >50% missing data")
    
    print(f"   ✓ Standardize column names and formats")
    print(f"   ✓ Validate data types")
    print(f"   ✓ Check for outliers in numerical columns")

def main():
    """Main function to execute Phase 2"""
    print("\n" + "=" * 80)
    print("PHASE 2: INITIAL DATA ASSESSMENT")
    print("=" * 80)
    
    # Define file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "data", "dataset-uci.csv")
    file_path = os.path.normpath(file_path)
    
    # Load data
    df = load_data(file_path)
    
    if df is not None:
        # Perform assessments
        missing_report = assess_missing_values(df)
        duplicate_count = assess_duplicates(df)
        type_report = assess_data_types(df)
        range_report = assess_value_ranges(df)
        assess_categorical_values(df)
        generate_summary_report(df, missing_report, duplicate_count)
        
        # Save reports
        reports_dir = os.path.join(script_dir, "..", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        missing_report.to_csv(os.path.join(reports_dir, "missing_values_report.csv"), index=False)
        type_report.to_csv(os.path.join(reports_dir, "data_types_report.csv"), index=False)
        if range_report is not None:
            range_report.to_csv(os.path.join(reports_dir, "value_ranges_report.csv"), index=False)
        
        print("\n" + "=" * 80)
        print("[SUCCESS] PHASE 2 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n📁 Reports saved to: {reports_dir}")
        print(f"   - missing_values_report.csv")
        print(f"   - data_types_report.csv")
        print(f"   - value_ranges_report.csv")
        print(f"\nNext steps: Run phase3_handle_missing_values.py")
        
        return df, missing_report
    else:
        print("\n[FAILED] Phase 2 failed. Please check the error messages above.")
        return None, None

if __name__ == "__main__":
    df, missing_report = main()

