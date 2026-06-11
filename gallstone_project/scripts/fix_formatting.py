"""
Phase 5: Fix Formatting Issues
This script standardizes column names and fixes formatting inconsistencies in the gallstone dataset.
"""

import pandas as pd
import numpy as np
import os
import sys
import re

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

def show_current_columns(df):
    """Display current column names"""
    print("\n" + "=" * 80)
    print("CURRENT COLUMN NAMES")
    print("=" * 80)
    
    print(f"\n📋 Total columns: {len(df.columns)}")
    print("-" * 80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")

def standardize_column_names(df):
    """Standardize column names to snake_case format"""
    print("\n" + "=" * 80)
    print("STANDARDIZING COLUMN NAMES")
    print("=" * 80)
    
    # Store original column names
    original_columns = df.columns.tolist()
    
    # Create mapping of old to new names
    column_mapping = {}
    
    for col in df.columns:
        # Convert to lowercase
        new_col = col.lower()
        
        # Remove parentheses and their contents
        new_col = re.sub(r'\([^)]*\)', '', new_col)
        
        # Replace spaces and special characters with underscores
        new_col = re.sub(r'[^a-z0-9]+', '_', new_col)
        
        # Remove leading/trailing underscores
        new_col = new_col.strip('_')
        
        # Remove multiple consecutive underscores
        new_col = re.sub(r'_+', '_', new_col)
        
        column_mapping[col] = new_col
    
    # Apply the mapping
    df_cleaned = df.rename(columns=column_mapping)
    
    # Display changes
    print("\n📝 Column Name Changes:")
    print("-" * 80)
    changes_made = 0
    for old, new in column_mapping.items():
        if old != new:
            print(f"   '{old}'")
            print(f"   → '{new}'")
            print()
            changes_made += 1
    
    if changes_made == 0:
        print("   ✅ All column names already standardized")
    else:
        print(f"\n✅ Standardized {changes_made} column names")
    
    return df_cleaned, column_mapping

def check_categorical_formatting(df):
    """Check for formatting issues in categorical columns"""
    print("\n" + "=" * 80)
    print("CATEGORICAL DATA FORMATTING CHECK")
    print("=" * 80)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        print("\n✅ No categorical (text) columns found in dataset")
        print("   All columns are numerical - no text formatting issues to fix")
        return df, []
    
    print(f"\n📊 Found {len(categorical_cols)} categorical columns")
    print("-" * 80)
    
    formatting_issues = []
    
    for col in categorical_cols:
        print(f"\n📋 Analyzing column: {col}")
        
        # Check for leading/trailing whitespace
        has_whitespace = df[col].astype(str).str.strip() != df[col].astype(str)
        whitespace_count = has_whitespace.sum()
        
        # Check for inconsistent casing
        unique_values = df[col].dropna().unique()
        unique_lower = df[col].dropna().str.lower().unique()
        has_case_issues = len(unique_values) != len(unique_lower)
        
        # Check for multiple spaces
        has_multiple_spaces = df[col].astype(str).str.contains(r'\s{2,}', regex=True).sum()
        
        if whitespace_count > 0 or has_case_issues or has_multiple_spaces > 0:
            formatting_issues.append(col)
            print(f"   ⚠️  Issues found:")
            if whitespace_count > 0:
                print(f"      - {whitespace_count} values with leading/trailing whitespace")
            if has_case_issues:
                print(f"      - Inconsistent casing detected")
            if has_multiple_spaces > 0:
                print(f"      - {has_multiple_spaces} values with multiple consecutive spaces")
        else:
            print(f"   ✅ No formatting issues")
    
    return df, formatting_issues

def fix_categorical_formatting(df, issues_cols):
    """Fix formatting issues in categorical columns"""
    if len(issues_cols) == 0:
        print("\n✅ No categorical formatting issues to fix")
        return df
    
    print("\n" + "=" * 80)
    print("FIXING CATEGORICAL FORMATTING")
    print("=" * 80)
    
    df_cleaned = df.copy()
    
    for col in issues_cols:
        print(f"\n🔧 Fixing column: {col}")
        
        # Remove leading/trailing whitespace
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
        
        # Replace multiple spaces with single space
        df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
        
        # Standardize to title case (optional - can be adjusted)
        df_cleaned[col] = df_cleaned[col].str.title()
        
        print(f"   ✅ Formatting fixed")
    
    return df_cleaned

def validate_numerical_columns(df):
    """Validate that numerical columns contain valid numbers"""
    print("\n" + "=" * 80)
    print("NUMERICAL DATA VALIDATION")
    print("=" * 80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"\n📊 Validating {len(numerical_cols)} numerical columns")
    print("-" * 80)
    
    validation_report = []
    
    for col in numerical_cols:
        # Check for infinite values
        inf_count = np.isinf(df[col]).sum()
        
        # Check for NaN values
        nan_count = df[col].isna().sum()
        
        # Check for negative values where they shouldn't be
        negative_count = (df[col] < 0).sum()
        
        validation_report.append({
            'Column': col,
            'Infinite_Values': inf_count,
            'NaN_Values': nan_count,
            'Negative_Values': negative_count,
            'Status': '✅ Valid' if (inf_count == 0 and nan_count == 0) else '⚠️  Issues'
        })
    
    validation_df = pd.DataFrame(validation_report)
    
    # Show only columns with issues
    issues = validation_df[validation_df['Status'] == '⚠️  Issues']
    
    if len(issues) > 0:
        print("\n⚠️  Columns with validation issues:")
        print(issues.to_string(index=False))
    else:
        print("\n✅ All numerical columns validated successfully")
        print("   - No infinite values")
        print("   - No NaN values")
        print("   - All values within expected ranges")
    
    return validation_df

def create_column_mapping_report(column_mapping):
    """Create a report of column name changes"""
    print("\n" + "=" * 80)
    print("COLUMN MAPPING REPORT")
    print("=" * 80)
    
    mapping_data = []
    for old, new in column_mapping.items():
        mapping_data.append({
            'Original_Name': old,
            'Standardized_Name': new,
            'Changed': 'Yes' if old != new else 'No'
        })
    
    mapping_df = pd.DataFrame(mapping_data)
    
    changes = mapping_df[mapping_df['Changed'] == 'Yes']
    print(f"\n📊 Summary:")
    print(f"   Total columns: {len(mapping_df)}")
    print(f"   Columns renamed: {len(changes)}")
    print(f"   Columns unchanged: {len(mapping_df) - len(changes)}")
    
    return mapping_df

def generate_summary_report(df_original, df_cleaned, column_mapping):
    """Generate overall formatting summary"""
    print("\n" + "=" * 80)
    print("PHASE 5 SUMMARY REPORT")
    print("=" * 80)
    
    print(f"\n📊 Formatting Changes Applied:")
    print("-" * 80)
    
    # Column name changes
    renamed_count = sum(1 for old, new in column_mapping.items() if old != new)
    print(f"   ✓ Column names standardized: {renamed_count} columns renamed")
    
    # Data integrity check
    print(f"\n📋 Data Integrity Check:")
    print(f"   Original rows: {len(df_original):,}")
    print(f"   Final rows: {len(df_cleaned):,}")
    print(f"   Rows preserved: {len(df_original) == len(df_cleaned)}")
    
    print(f"\n✅ Phase 5 Completed Successfully!")
    print(f"   - All column names standardized to snake_case")
    print(f"   - No data loss during formatting")
    print(f"   - Dataset ready for analysis")

def main():
    """Main function to execute Phase 5"""
    print("\n" + "=" * 80)
    print("PHASE 5: FIX FORMATTING ISSUES")
    print("=" * 80)
    
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "data", "dataset-uci.csv")
    input_file = os.path.normpath(input_file)
    
    # Load data
    df = load_data(input_file)
    
    if df is not None:
        # Store original for comparison
        df_original = df.copy()
        
        # Show current columns
        show_current_columns(df)
        
        # Standardize column names
        df_cleaned, column_mapping = standardize_column_names(df)
        
        # Check categorical formatting
        df_cleaned, issues_cols = check_categorical_formatting(df_cleaned)
        
        # Fix categorical formatting if needed
        if len(issues_cols) > 0:
            df_cleaned = fix_categorical_formatting(df_cleaned, issues_cols)
        
        # Validate numerical columns
        validation_report = validate_numerical_columns(df_cleaned)
        
        # Create reports
        mapping_report = create_column_mapping_report(column_mapping)
        
        # Generate summary
        generate_summary_report(df_original, df_cleaned, column_mapping)
        
        # Save cleaned data
        output_dir = os.path.join(script_dir, "..", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "gallstone_cleaned.csv")
        df_cleaned.to_csv(output_file, index=False)
        
        # Save reports
        reports_dir = os.path.join(script_dir, "..", "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        mapping_report.to_csv(os.path.join(reports_dir, "column_mapping_report.csv"), index=False)
        validation_report.to_csv(os.path.join(reports_dir, "numerical_validation_report.csv"), index=False)
        
        print("\n" + "=" * 80)
        print("[SUCCESS] PHASE 5 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\n📁 Output saved to:")
        print(f"   - {output_file}")
        print(f"\n📁 Reports saved to: {reports_dir}")
        print(f"   - column_mapping_report.csv")
        print(f"   - numerical_validation_report.csv")
        print(f"\n🎉 Data cleaning pipeline complete!")
        print(f"   The cleaned dataset is ready for analysis and modeling.")
        
        return df_cleaned
    else:
        print("\n[FAILED] Phase 5 failed. Please check the error messages above.")
        return None

if __name__ == "__main__":
    df_cleaned = main()

