"""
BUPA Liver Disorders Dataset - Data Cleaning Script
This script implements comprehensive data cleaning pipeline for the liver dataset.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except AttributeError:
        pass

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'datasets' / 'liver'
OUTPUT_DIR = BASE_DIR / 'datasets' / 'liver'
REPORTS_DIR = BASE_DIR / 'reports' / 'liver'

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class LiverDataCleaner:
    """Class to handle cleaning of liver disorders dataset"""
    
    def __init__(self):
        self.cleaning_report = []
        
    def log_action(self, action, details):
        """Log cleaning actions for reporting"""
        self.cleaning_report.append({
            'Action': action,
            'Details': details
        })
        print(f"   {action}: {details}")
    
    def clean_dataset(self):
        """Clean the liver dataset"""
        print("\n" + "="*80)
        print("CLEANING BUPA LIVER DISORDERS DATASET")
        print("="*80)
        
        # Column names from bupa.names file
        columns = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
        
        # Load data
        df = pd.read_csv(DATA_DIR / 'bupa.data', names=columns)
        initial_rows = len(df)
        initial_cols = len(df.columns)
        self.log_action('Loaded', f'{initial_rows} rows, {initial_cols} columns')
        
        # 1. Check for missing values
        print("\n1. Handling Missing Values")
        print("-" * 70)
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            self.log_action('Missing values found', f'{missing_count} total missing values')
            for col in df.columns:
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
                except:
                    self.log_action('Conversion failed', f'{col}: could not convert to numeric')
        
        # 4. Handle outliers (cap at 1.5*IQR, but keep them as valid medical values)
        print("\n4. Analyzing Outliers")
        print("-" * 70)
        
        outlier_summary = []
        for col in df.columns:
            if col != 'selector':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    outlier_summary.append({
                        'Feature': col,
                        'Outlier_Count': outliers_count,
                        'Outlier_Percentage': round(outliers_count/len(df)*100, 2),
                        'Q1': Q1,
                        'Q3': Q3,
                        'IQR': IQR,
                        'Lower_Bound': lower_bound,
                        'Upper_Bound': upper_bound
                    })
                    self.log_action('Outliers detected', 
                                  f'{col}: {outliers_count} values ({outliers_count/len(df)*100:.2f}%) - kept as valid')
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary)
            outlier_df.to_csv(REPORTS_DIR / 'outlier_analysis.csv', index=False)
            self.log_action('Saved outlier analysis', f'{len(outlier_summary)} features with outliers')
        
        # 5. Create engineered features
        print("\n5. Creating Basic Engineered Features")
        print("-" * 70)
        
        # AST/ALT ratio (clinically significant)
        df['ast_alt_ratio'] = df['sgot'] / df['sgpt']
        self.log_action('Created feature', 'ast_alt_ratio (AST/ALT ratio)')
        
        # Alcohol consumption categories
        df['alcohol_category'] = pd.cut(df['drinks'], 
                                        bins=[-0.1, 0, 1, 2, 4, float('inf')],
                                        labels=['None', 'Light', 'Moderate', 'Heavy', 'Very Heavy'])
        self.log_action('Created feature', 'alcohol_category (5 levels)')
        
        # Total enzyme score (normalized sum of enzymes)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        enzyme_cols = ['sgpt', 'sgot', 'gammagt', 'alkphos']
        df['total_enzyme_score'] = scaler.fit_transform(df[enzyme_cols]).sum(axis=1)
        self.log_action('Created feature', 'total_enzyme_score (normalized enzyme sum)')
        
        # 6. Validate data ranges
        print("\n6. Validating Data Ranges")
        print("-" * 70)
        
        # Check for negative values
        for col in df.select_dtypes(include=[np.number]).columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                self.log_action('Negative values found', f'{col}: {neg_count} values')
        
        # 7. Final validation
        print("\n7. Final Validation")
        print("-" * 70)
        
        self._validate_dataset(df)
        
        # 8. Save cleaned data
        output_path = OUTPUT_DIR / 'liver_cleaned.csv'
        df.to_csv(output_path, index=False)
        self.log_action('Saved', f'{len(df)} rows, {len(df.columns)} columns to {output_path.name}')
        
        print(f"\n[OK] Liver dataset cleaned successfully")
        print(f"   Original: {initial_rows} rows, {initial_cols} columns")
        print(f"   Cleaned: {len(df)} rows, {len(df.columns)} columns")
        print(f"   New features: 3 (ast_alt_ratio, alcohol_category, total_enzyme_score)")
        
        return df
    
    def _validate_dataset(self, df):
        """Validate the cleaned dataset"""
        issues = []
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            issues.append("Missing values still present")
        
        # Check for duplicates
        if df.duplicated().sum() > 0:
            issues.append("Duplicate rows still present")
        
        # Check selector values
        if 'selector' in df.columns:
            unique_selectors = df['selector'].unique()
            if not all(val in [1, 2] for val in unique_selectors):
                issues.append(f"Invalid selector values: {unique_selectors}")
        
        # Check for required columns
        required_cols = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing required columns: {missing_cols}")
        
        if issues:
            print("   âš  Validation issues:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print("   âœ“ All validation checks passed")
    
    def save_cleaning_report(self):
        """Save the cleaning report"""
        report_df = pd.DataFrame(self.cleaning_report)
        report_path = REPORTS_DIR / 'data_cleaning_report.csv'
        report_df.to_csv(report_path, index=False)
        print(f"\n[OK] Cleaning report saved to {report_path}")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BUPA LIVER DISORDERS DATASET - DATA CLEANING")
    print("="*80)
    
    cleaner = LiverDataCleaner()
    df = cleaner.clean_dataset()
    cleaner.save_cleaning_report()
    
    print("\n" + "="*80)
    print("DATA CLEANING COMPLETE")
    print("="*80)
    print(f"[OK] Cleaned dataset: {OUTPUT_DIR / 'liver_cleaned.csv'}")
    print(f"[OK] Cleaning report: {REPORTS_DIR / 'data_cleaning_report.csv'}")
    print("\nNext steps:")
    print("1. Review the cleaned dataset")
    print("2. Run exploratory_analysis_liver.py for EDA")
    print("3. Run feature_engineering_liver.py to create advanced features")
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    main()

