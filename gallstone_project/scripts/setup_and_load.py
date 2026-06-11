"""
Phase 1: Setup and Data Loading
This script loads the gallstone dataset and explores its structure.
"""

import pandas as pd
import numpy as np
import os
import sys

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

def load_data(file_path):
    """
    Load Excel file and return the dataframe
    """
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load Excel file directly (handles files with sheet issues)
        print(f"\n[OK] Loading file: {file_path}")
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # Try to get sheet names
        try:
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names if excel_file.sheet_names else ['default']
            print(f"[OK] Sheet names: {sheet_names}")
        except:
            sheet_names = ['default']
            print(f"[OK] Loaded data successfully (sheet detection unavailable)")
        
        print(f"[OK] Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
        
        return df, sheet_names
    
    except Exception as e:
        print(f"[ERROR] Error loading file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def explore_structure(df):
    """
    Explore the structure of the dataframe
    """
    print("\n" + "=" * 80)
    print("DATA STRUCTURE EXPLORATION")
    print("=" * 80)
    
    # Dataset shape
    print(f"\n📊 Dataset Shape:")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")
    
    # Column names and data types
    print(f"\n📋 Column Names and Data Types:")
    print("-" * 80)
    for idx, (col, dtype) in enumerate(zip(df.columns, df.dtypes), 1):
        print(f"   {idx:2d}. {col:40s} | {str(dtype):15s}")
    
    # Memory usage
    print(f"\n💾 Memory Usage:")
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"   Total: {memory_mb:.2f} MB")
    
    # First few rows
    print(f"\n👀 First 5 Rows:")
    print("-" * 80)
    print(df.head())
    
    # Last few rows
    print(f"\n👀 Last 5 Rows:")
    print("-" * 80)
    print(df.tail())
    
    # Basic statistics for numerical columns
    print(f"\n📈 Basic Statistics (Numerical Columns):")
    print("-" * 80)
    print(df.describe())
    
    # Data types summary
    print(f"\n📊 Data Types Summary:")
    print("-" * 80)
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   {str(dtype):15s}: {count} columns")
    
    # Detailed info
    print(f"\n🔍 Detailed Dataset Info:")
    print("-" * 80)
    df.info()
    
    return df

def main():
    """
    Main function to execute Phase 1
    """
    print("\n" + "=" * 80)
    print("PHASE 1: SETUP AND DATA LOADING")
    print("=" * 80)
    
    # Define file path - use absolute path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "..", "data", "dataset-uci.xlsx")
    file_path = os.path.normpath(file_path)
    print(f"\nLooking for file at: {file_path}")
    
    # Step 1: Load data
    df, sheet_names = load_data(file_path)
    
    if df is not None:
        # Step 2: Explore structure
        df = explore_structure(df)
        
        print("\n" + "=" * 80)
        print("[SUCCESS] PHASE 1 COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nDataset is ready for Phase 2: Initial Data Assessment")
        print(f"Next steps: Run phase2_initial_assessment.py")
        
        return df
    else:
        print("\n[FAILED] Phase 1 failed. Please check the error messages above.")
        return None

if __name__ == "__main__":
    df = main()


