"""
Thyroid Disease Dataset - Data Cleaning Script
This script implements the comprehensive data cleaning pipeline for all three thyroid datasets.
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
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / 'output'
REPORTS_DIR = BASE_DIR / 'reports'

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


class ThyroidDataCleaner:
    """Class to handle cleaning of thyroid disease datasets"""
    
    def __init__(self):
        self.cleaning_report = []
        
    def log_action(self, dataset, action, details):
        """Log cleaning actions for reporting"""
        self.cleaning_report.append({
            'Dataset': dataset,
            'Action': action,
            'Details': details
        })
        print(f"   [{dataset}] {action}: {details}")
    
    def clean_hypothyroid_dataset(self):
        """Clean the hypothyroid dataset"""
        print("\n" + "="*80)
        print("CLEANING HYPOTHYROID DATASET")
        print("="*80)
        
        # Column names
        columns = [
            'class', 'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
            'on_antithyroid_medication', 'thyroid_surgery', 'query_hypothyroid',
            'query_hyperthyroid', 'pregnant', 'sick', 'tumor', 'lithium',
            'goitre', 'TSH_measured', 'TSH', 'T3_measured', 'T3',
            'TT4_measured', 'TT4', 'T4U_measured', 'T4U', 'FTI_measured',
            'FTI', 'TBG_measured', 'TBG'
        ]
        
        # Load data
        df = pd.read_csv(DATA_DIR / 'hypothyroid.data', names=columns, na_values='?')
        initial_rows = len(df)
        initial_cols = len(df.columns)
        self.log_action('hypothyroid', 'Loaded', f'{initial_rows} rows, {initial_cols} columns')
        
        # 1. Drop TBG column (>90% missing)
        if 'TBG' in df.columns:
            missing_pct = df['TBG'].isnull().sum() / len(df) * 100
            df = df.drop('TBG', axis=1)
            self.log_action('hypothyroid', 'Dropped column', f'TBG ({missing_pct:.1f}% missing)')
        
        # Also drop TBG_measured since TBG is dropped
        if 'TBG_measured' in df.columns:
            df = df.drop('TBG_measured', axis=1)
            self.log_action('hypothyroid', 'Dropped column', 'TBG_measured (related to dropped TBG)')
        
        # 2. Handle missing values in numerical columns BEFORE removing duplicates
        numerical_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        for col in numerical_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    median_value = df[col].median()
                    df[col] = df[col].fillna(median_value)
                    self.log_action('hypothyroid', 'Imputed missing',
                                  f'{col}: {missing_count} values with median ({median_value:.2f})')
        
        # 3. Handle missing values in categorical columns
        if 'sex' in df.columns:
            missing_count = df['sex'].isnull().sum()
            if missing_count > 0:
                mode_series = df['sex'].mode()
                if len(mode_series) > 0:
                    mode_value = str(mode_series.iloc[0])
                    df['sex'] = df['sex'].fillna(mode_value)
                    self.log_action('hypothyroid', 'Imputed missing',
                                  f'sex: {missing_count} values with mode ({mode_value})')
        
        # 4. Fill remaining missing values in binary columns with mode (0)
        binary_cols = ['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
                      'thyroid_surgery', 'query_hypothyroid', 'query_hyperthyroid',
                      'pregnant', 'sick', 'tumor', 'lithium', 'goitre',
                      'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured']
        
        for col in binary_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    df[col] = df[col].fillna('f')  # Fill with 'f' before conversion
                    self.log_action('hypothyroid', 'Imputed missing',
                                  f'{col}: {missing_count} values with mode (f)')
        
        # 5. Remove duplicates AFTER handling missing values (keep first occurrence)
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates(keep='first')
            self.log_action('hypothyroid', 'Removed duplicates', f'{duplicates} rows ({duplicates/initial_rows*100:.2f}%)')
        
        # 6. Convert binary categorical to numeric
        binary_map = {'f': 0, 't': 1}
        binary_cols = ['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
                      'thyroid_surgery', 'query_hypothyroid', 'query_hyperthyroid',
                      'pregnant', 'sick', 'tumor', 'lithium', 'goitre',
                      'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured']
        
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map(binary_map)  # type: ignore
                # Fill any remaining NaN values (from unmapped values) with 0
                nan_count = df[col].isnull().sum()
                if nan_count > 0:
                    df[col] = df[col].fillna(0)
                self.log_action('hypothyroid', 'Converted to binary', f'{col}: f/t -> 0/1')
        
        # 7. Convert sex to binary
        if 'sex' in df.columns:
            sex_map = {'M': 1, 'F': 0}
            df['sex'] = df['sex'].map(sex_map)  # type: ignore
            # Fill any remaining NaN values with 0 (Female)
            nan_count = df['sex'].isnull().sum()
            if nan_count > 0:
                df['sex'] = df['sex'].fillna(0)
            self.log_action('hypothyroid', 'Converted to binary', 'sex: M/F -> 1/0')
        
        # 8. Convert class to binary
        if 'class' in df.columns:
            class_map = {'negative': 0, 'hypothyroid': 1}
            df['class'] = df['class'].map(class_map)  # type: ignore
            # Fill any remaining NaN values with 0 (negative)
            nan_count = df['class'].isnull().sum()
            if nan_count > 0:
                df['class'] = df['class'].fillna(0)
            self.log_action('hypothyroid', 'Converted to binary', 'class: negative/hypothyroid -> 0/1')
        
        # 9. Handle outliers (cap using IQR method)
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    self.log_action('hypothyroid', 'Capped outliers',
                                  f'{col}: {outliers_count} values ({outliers_count/len(df)*100:.2f}%)')
        
        # 10. Final duplicate check (after all transformations)
        final_duplicates = df.duplicated().sum()
        if final_duplicates > 0:
            df = df.drop_duplicates(keep='first')
            self.log_action('hypothyroid', 'Removed final duplicates',
                          f'{final_duplicates} rows created by transformations')
        
        # 11. Validate data
        self._validate_dataset(df, 'hypothyroid')
        
        # 11. Save cleaned data
        output_path = OUTPUT_DIR / 'hypothyroid_cleaned.csv'
        df.to_csv(output_path, index=False)
        self.log_action('hypothyroid', 'Saved', f'{len(df)} rows, {len(df.columns)} columns to {output_path.name}')
        
        print(f"\n[OK] Hypothyroid dataset cleaned successfully")
        print(f"   Original: {initial_rows} rows, {initial_cols} columns")
        print(f"   Cleaned: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def clean_ann_thyroid_dataset(self):
        """Clean the ANN-thyroid dataset"""
        print("\n" + "="*80)
        print("CLEANING ANN-THYROID DATASET")
        print("="*80)
        
        # Column names
        columns = [
            'age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
            'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
            'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid',
            'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
            'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'class'
        ]
        
        # Load training and test data
        df_train = pd.read_csv(DATA_DIR / 'ann-train.data', sep=r'\s+',
                              names=columns, header=None, engine='python')
        df_test = pd.read_csv(DATA_DIR / 'ann-test.data', sep=r'\s+',
                             names=columns, header=None, engine='python')
        
        # Add dataset identifier
        df_train['dataset_split'] = 'train'
        df_test['dataset_split'] = 'test'
        
        # Combine datasets
        df = pd.concat([df_train, df_test], ignore_index=True)
        initial_rows = len(df)
        initial_cols = len(df.columns)
        self.log_action('ann-thyroid', 'Loaded', 
                       f'{len(df_train)} train + {len(df_test)} test = {initial_rows} rows, {initial_cols} columns')
        
        # 1. Fix class column (it's the last column with values 1, 2, 3)
        # Based on documentation: 1=normal, 2=hyperfunction, 3=subnormal
        class_map = {1: 'normal', 2: 'hyperfunction', 3: 'subnormal'}
        df['class'] = df['class'].map(class_map)  # type: ignore
        self.log_action('ann-thyroid', 'Mapped class', '1/2/3 -> normal/hyperfunction/subnormal')
        
        # 2. Validate binary columns (should be 0 or 1) - Fix BEFORE removing duplicates
        binary_cols = ['sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
                      'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment',
                      'query_hypothyroid', 'query_hyperthyroid', 'lithium', 'goitre',
                      'tumor', 'hypopituitary', 'psych']
        
        for col in binary_cols:
            if col in df.columns:
                invalid = ~df[col].isin([0, 1])
                invalid_count = invalid.sum()
                if invalid_count > 0:
                    # Set invalid values to 0 (most common for binary features)
                    df.loc[invalid, col] = 0
                    self.log_action('ann-thyroid', 'Fixed invalid values',
                                  f'{col}: {invalid_count} values set to 0')
        
        # 3. Remove duplicates AFTER fixing invalid values (keep first occurrence)
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates(keep='first')
            self.log_action('ann-thyroid', 'Removed duplicates',
                          f'{duplicates} rows ({duplicates/initial_rows*100:.2f}%)')
        
        # 4. Handle outliers in continuous columns
        continuous_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
        for col in continuous_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    self.log_action('ann-thyroid', 'Capped outliers', 
                                  f'{col}: {outliers_count} values ({outliers_count/len(df)*100:.2f}%)')
        
        # 5. Validate data
        self._validate_dataset(df, 'ann-thyroid')
        
        # 6. Save cleaned data
        output_path = OUTPUT_DIR / 'ann_thyroid_cleaned.csv'
        df.to_csv(output_path, index=False)
        self.log_action('ann-thyroid', 'Saved', f'{len(df)} rows, {len(df.columns)} columns to {output_path.name}')
        
        print(f"\n[OK] ANN-Thyroid dataset cleaned successfully")
        print(f"   Original: {initial_rows} rows, {initial_cols} columns")
        print(f"   Cleaned: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def clean_new_thyroid_dataset(self):
        """Clean the new-thyroid dataset"""
        print("\n" + "="*80)
        print("CLEANING NEW-THYROID DATASET")
        print("="*80)
        
        # Column names
        columns = [
            'class', 'T3_resin_uptake', 'total_serum_thyroxin',
            'total_serum_triiodothyronine', 'basal_TSH', 'max_TSH_difference'
        ]
        
        # Load data
        df = pd.read_csv(DATA_DIR / 'new-thyroid.data', names=columns)
        initial_rows = len(df)
        initial_cols = len(df.columns)
        self.log_action('new-thyroid', 'Loaded', f'{initial_rows} rows, {initial_cols} columns')
        
        # 1. Map class labels
        class_map = {1: 'normal', 2: 'hyperthyroid', 3: 'hypothyroid'}
        df['class'] = df['class'].map(class_map)  # type: ignore
        self.log_action('new-thyroid', 'Mapped class', '1/2/3 -> normal/hyperthyroid/hypothyroid')
        
        # 2. Check for duplicates (none expected based on assessment)
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates()
            self.log_action('new-thyroid', 'Removed duplicates', f'{duplicates} rows')
        else:
            self.log_action('new-thyroid', 'No duplicates', 'Dataset is clean')
        
        # 3. Handle negative values in max_TSH_difference
        # Note: Negative values might be valid (representing decrease)
        # We'll keep them but log the count
        negative_count = (df['max_TSH_difference'] < 0).sum()
        if negative_count > 0:
            self.log_action('new-thyroid', 'Negative values found', 
                          f'max_TSH_difference: {negative_count} negative values (kept as valid)')
        
        # 4. Handle outliers
        numerical_cols = ['T3_resin_uptake', 'total_serum_thyroxin', 
                         'total_serum_triiodothyronine', 'basal_TSH', 'max_TSH_difference']
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers_count > 0:
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                self.log_action('new-thyroid', 'Capped outliers', 
                              f'{col}: {outliers_count} values ({outliers_count/len(df)*100:.2f}%)')
        
        # 5. Validate data
        self._validate_dataset(df, 'new-thyroid')
        
        # 6. Save cleaned data
        output_path = OUTPUT_DIR / 'new_thyroid_cleaned.csv'
        df.to_csv(output_path, index=False)
        self.log_action('new-thyroid', 'Saved', f'{len(df)} rows, {len(df.columns)} columns to {output_path.name}')
        
        print(f"\n[OK] New-Thyroid dataset cleaned successfully")
        print(f"   Original: {initial_rows} rows, {initial_cols} columns")
        print(f"   Cleaned: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def _validate_dataset(self, df, dataset_name):
        """Validate cleaned dataset"""
        print(f"\n   Validating {dataset_name} dataset...")
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        if missing > 0:
            print(f"   [WARNING] {missing} missing values found")
        else:
            print(f"   [OK] No missing values")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"   [WARNING] {duplicates} duplicate rows found")
        else:
            print(f"   [OK] No duplicate rows")
        
        # Check class distribution
        if 'class' in df.columns:
            class_dist = df['class'].value_counts().sort_index()
            print(f"   [OK] Class distribution:")
            for cls, count in class_dist.items():
                print(f"      {cls}: {count} ({count/len(df)*100:.2f}%)")
        else:
            print(f"   [INFO] No 'class' column found in dataset")
    
    def generate_cleaning_report(self):
        """Generate comprehensive cleaning report"""
        print("\n" + "="*80)
        print("GENERATING CLEANING REPORT")
        print("="*80)
        
        if not self.cleaning_report:
            print("   No cleaning actions recorded")
            return
        
        # Create report DataFrame
        report_df = pd.DataFrame(self.cleaning_report)
        
        # Save detailed report
        report_path = REPORTS_DIR / 'data_cleaning_report.csv'
        report_df.to_csv(report_path, index=False)
        print(f"\n[OK] Detailed cleaning report saved to {report_path.name}")
        
        # Print summary
        print("\nCleaning Summary:")
        for dataset in report_df['Dataset'].unique():
            dataset_actions = report_df[report_df['Dataset'] == dataset]
            print(f"\n{dataset.upper()}:")
            for _, row in dataset_actions.iterrows():
                print(f"   - {row['Action']}: {row['Details']}")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("THYROID DISEASE DATASET - DATA CLEANING PIPELINE")
    print("="*80)
    
    # Initialize cleaner
    cleaner = ThyroidDataCleaner()
    
    # Clean all datasets
    try:
        df_hypothyroid = cleaner.clean_hypothyroid_dataset()
        df_ann = cleaner.clean_ann_thyroid_dataset()
        df_new = cleaner.clean_new_thyroid_dataset()
        
        # Generate cleaning report
        cleaner.generate_cleaning_report()
        
        print("\n" + "="*80)
        print("DATA CLEANING COMPLETE")
        print("="*80)
        print(f"\nCleaned datasets saved to: {OUTPUT_DIR}")
        print(f"Cleaning reports saved to: {REPORTS_DIR}")
        print("\nNext steps:")
        print("1. Review cleaned datasets in the 'output' directory")
        print("2. Review cleaning report in the 'reports' directory")
        print("3. Proceed with exploratory data analysis")
        print("4. Build machine learning models")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Data cleaning failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

