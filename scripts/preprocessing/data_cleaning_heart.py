"""
Heart Disease Dataset - Data Cleaning Script
This script implements comprehensive data cleaning pipeline for the heart dataset.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import cast
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except AttributeError:
        pass

# Define paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'datasets' / 'heart'
OUTPUT_DIR = BASE_DIR / 'datasets' / 'heart'
REPORTS_DIR = BASE_DIR / 'reports' / 'heart'

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class HeartDataCleaner:
    """Class to handle cleaning of heart disease dataset"""
    
    def __init__(self):
        self.cleaning_report = []
        
    def log_action(self, action, details):
        """Log cleaning actions for reporting"""
        self.cleaning_report.append({
            'Action': action,
            'Details': details
        })
        print(f"   {action}: {details}")
    
    def clean_dataset(self) -> pd.DataFrame:
        """Clean the heart disease dataset"""
        print("\n" + "="*80)
        print("CLEANING HEART DISEASE DATASET")
        print("="*80)
        
        # Load data
        df: pd.DataFrame = pd.read_csv(DATA_DIR / 'heart_master_dataset.csv')
        initial_rows = len(df)
        initial_cols = len(df.columns)
        self.log_action('Loaded', f'{initial_rows} rows, {initial_cols} columns')
        
        # 1. Check for missing values
        print("\n1. Handling Missing Values")
        print("-" * 70)
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            self.log_action('Missing values found', f'{missing_count} total missing values')
            
            # Impute numerical columns with median
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if col != 'target':
                    missing = df[col].isnull().sum()
                    if missing > 0:
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                        self.log_action('Imputed missing', f'{col}: {missing} values with median ({median_val:.2f})')
        else:
            self.log_action('No missing values', 'Dataset is complete')
        
        # 2. Remove duplicates
        print("\n2. Removing Duplicates")
        print("-" * 70)
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            # Save duplicate rows before removing
            dup_df = df[df.duplicated(keep=False)]
            dup_df.to_csv(REPORTS_DIR / 'removed_duplicates.csv', index=False)
            
            df = df.drop_duplicates(keep='first')
            self.log_action('Removed duplicates', f'{duplicates} rows ({duplicates/initial_rows*100:.2f}%)')
        else:
            self.log_action('No duplicates', 'Dataset is clean')
        
        # 3. Validate data types
        print("\n3. Validating Data Types")
        print("-" * 70)
        
        # Ensure all columns are numeric
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                    self.log_action('Converted to numeric', f'{col}')
                except ValueError:
                    self.log_action('Warning', f'{col} could not be converted to numeric')
        
        # 4. Handle outliers
        print("\n4. Handling Outliers")
        print("-" * 70)
        
        outlier_summary = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col != 'target':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                
                if outliers > 0:
                    outlier_summary[col] = outliers
                    # Cap outliers instead of removing
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    self.log_action('Capped outliers', f'{col}: {outliers} values')
        
        if not outlier_summary:
            self.log_action('No extreme outliers', 'Dataset is clean')
        
        # 5. Validate feature ranges
        print("\n5. Validating Feature Ranges")
        print("-" * 70)
        
        # Age should be reasonable (typically 20-100)
        if 'age' in df.columns:
            invalid_age = ((df['age'] < 20) | (df['age'] > 100)).sum()
            if invalid_age > 0:
                df = cast(pd.DataFrame, df[(df['age'] >= 20) & (df['age'] <= 100)].copy())
                self.log_action('Removed invalid ages', f'{invalid_age} rows with age < 20 or > 100')
        
        # Resting BP should be reasonable (typically 80-200)
        if 'resting_bp' in df.columns:
            invalid_bp = ((df['resting_bp'] < 80) | (df['resting_bp'] > 200)).sum()
            if invalid_bp > 0:
                df = cast(pd.DataFrame, df[(df['resting_bp'] >= 80) & (df['resting_bp'] <= 200)].copy())
                self.log_action('Removed invalid BP', f'{invalid_bp} rows with BP < 80 or > 200')
        
        # Cholesterol should be reasonable (typically 100-600)
        if 'cholesterol' in df.columns:
            # Remove zero cholesterol values (likely missing data coded as 0)
            zero_chol = (df['cholesterol'] == 0).sum()
            if zero_chol > 0:
                df = cast(pd.DataFrame, df[df['cholesterol'] > 0].copy())
                self.log_action('Removed zero cholesterol', f'{zero_chol} rows with cholesterol = 0')
            
            invalid_chol = ((df['cholesterol'] < 100) | (df['cholesterol'] > 600)).sum()
            if invalid_chol > 0:
                df = cast(pd.DataFrame, df[(df['cholesterol'] >= 100) & (df['cholesterol'] <= 600)].copy())
                self.log_action('Removed invalid cholesterol', f'{invalid_chol} rows')
        
        # Max heart rate should be reasonable (typically 60-220)
        if 'max_heart_rate' in df.columns:
            invalid_hr = ((df['max_heart_rate'] < 60) | (df['max_heart_rate'] > 220)).sum()
            if invalid_hr > 0:
                df = cast(pd.DataFrame, df[(df['max_heart_rate'] >= 60) & (df['max_heart_rate'] <= 220)].copy())
                self.log_action('Removed invalid heart rate', f'{invalid_hr} rows')
        
        # 6. Validate target variable
        print("\n6. Validating Target Variable")
        print("-" * 70)
        
        if 'target' in df.columns:
            unique_targets = df['target'].unique()
            self.log_action('Target classes', f'{len(unique_targets)} classes: {sorted(unique_targets)}')
            
            # Ensure target is binary (0 or 1)
            if not all(val in [0, 1] for val in unique_targets):
                self.log_action('Warning', 'Target variable contains values other than 0 and 1')
        
        # 7. Final statistics
        print("\n7. Final Dataset Statistics")
        print("-" * 70)
        
        final_rows = len(df)
        final_cols = len(df.columns)
        rows_removed = initial_rows - final_rows
        
        self.log_action('Final dataset', f'{final_rows} rows, {final_cols} columns')
        self.log_action('Total rows removed', f'{rows_removed} ({rows_removed/initial_rows*100:.2f}%)')
        
        # Save cleaned dataset
        output_path = OUTPUT_DIR / 'heart_cleaned.csv'
        df.to_csv(output_path, index=False)
        self.log_action('Saved cleaned dataset', str(output_path))
        
        # Save cleaning report
        report_df = pd.DataFrame(self.cleaning_report)
        report_path = REPORTS_DIR / 'cleaning_report.csv'
        report_df.to_csv(report_path, index=False)
        self.log_action('Saved cleaning report', str(report_path))
        
        return df
    
    def generate_summary_report(self, df: pd.DataFrame) -> None:
        """Generate a summary report of the cleaned dataset"""
        print("\n" + "="*80)
        print("CLEANING SUMMARY REPORT")
        print("="*80)
        
        summary = []
        summary.append("="*80)
        summary.append("HEART DISEASE DATASET - CLEANING SUMMARY")
        summary.append("="*80)
        summary.append("")
        
        summary.append("DATASET OVERVIEW")
        summary.append("-"*80)
        summary.append(f"Total Samples: {len(df)}")
        summary.append(f"Total Features: {len(df.columns)}")
        summary.append(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        summary.append("")
        
        summary.append("FEATURE INFORMATION")
        summary.append("-"*80)
        for col in df.columns:
            dtype = str(df[col].dtype)
            missing = df[col].isnull().sum()
            unique = df[col].nunique()
            summary.append(f"{col:30s} | Type: {dtype:10s} | Missing: {missing:4d} | Unique: {unique:4d}")
        summary.append("")
        
        summary.append("TARGET DISTRIBUTION")
        summary.append("-"*80)
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            for cls, count in target_counts.items():
                pct = (count / len(df)) * 100
                summary.append(f"Class {cls}: {count:4d} samples ({pct:5.2f}%)")
        summary.append("")
        
        summary.append("CLEANING ACTIONS")
        summary.append("-"*80)
        for action in self.cleaning_report:
            summary.append(f"{action['Action']:30s} | {action['Details']}")
        summary.append("")
        summary.append("="*80)
        
        # Save summary report
        summary_path = REPORTS_DIR / 'CLEANING_SUMMARY.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))
        
        print(f"\n[OK] Summary report saved to {summary_path}")
        
        # Print summary to console
        print('\n'.join(summary))


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("HEART DISEASE DATASET - DATA CLEANING PIPELINE")
    print("="*80)
    
    cleaner = HeartDataCleaner()
    
    try:
        # Clean dataset
        df = cleaner.clean_dataset()
        
        # Generate summary report
        cleaner.generate_summary_report(df)
        
        print("\n" + "="*80)
        print("DATA CLEANING COMPLETE")
        print("="*80)
        print(f"\nCleaned dataset saved to: {OUTPUT_DIR / 'heart_cleaned.csv'}")
        print(f"Reports saved to: {REPORTS_DIR}")
        print("\nNext steps:")
        print("1. Run exploratory data analysis (EDA)")
        print("2. Perform feature engineering")
        print("3. Create train/test splits")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Data cleaning failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# Made with Bob
