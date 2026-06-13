"""
Gallbladder Disease Dataset - Data Cleaning Script
This script implements comprehensive data cleaning pipeline for the gallbladder dataset.
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
DATA_DIR = BASE_DIR / 'datasets' / 'gallbladder'
OUTPUT_DIR = BASE_DIR / 'datasets' / 'gallbladder'
REPORTS_DIR = BASE_DIR / 'reports' / 'gallbladder'

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class GallbladderDataCleaner:
    """Class to handle cleaning of gallbladder disease dataset"""
    
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
        """Clean the gallbladder dataset"""
        print("\n" + "="*80)
        print("CLEANING GALLBLADDER DATASET")
        print("="*80)
        
        # Load data
        df = pd.read_csv(DATA_DIR / 'dataset-uci.csv')
        initial_rows = len(df)
        initial_cols = len(df.columns)
        self.log_action('Loaded', f'{initial_rows} rows, {initial_cols} columns')
        
        # 1. Standardize column names (remove special characters, lowercase, underscores)
        print("\n1. Standardizing Column Names")
        print("-" * 70)
        
        column_mapping = {
            'Gallstone Status': 'gallstone_status',
            'Age': 'age',
            'Gender': 'gender',
            'Comorbidity': 'comorbidity',
            'Coronary Artery Disease (CAD)': 'coronary_artery_disease',
            'Hypothyroidism': 'hypothyroidism',
            'Hyperlipidemia': 'hyperlipidemia',
            'Diabetes Mellitus (DM)': 'diabetes_mellitus',
            'Height': 'height',
            'Weight': 'weight',
            'Body Mass Index (BMI)': 'body_mass_index',
            'Total Body Water (TBW)': 'total_body_water',
            'Extracellular Water (ECW)': 'extracellular_water',
            'Intracellular Water (ICW)': 'intracellular_water',
            'Extracellular Fluid/Total Body Water (ECF/TBW)': 'extracellular_fluid_total_body_water',
            'Total Body Fat Ratio (TBFR) (%)': 'total_body_fat_ratio',
            'Lean Mass (LM) (%)': 'lean_mass',
            'Body Protein Content (Protein) (%)': 'body_protein_content',
            'Visceral Fat Rating (VFR)': 'visceral_fat_rating',
            'Bone Mass (BM)': 'bone_mass',
            'Muscle Mass (MM)': 'muscle_mass',
            'Obesity (%)': 'obesity',
            'Total Fat Content (TFC)': 'total_fat_content',
            'Visceral Fat Area (VFA)': 'visceral_fat_area',
            'Visceral Muscle Area (VMA) (Kg)': 'visceral_muscle_area',
            'Hepatic Fat Accumulation (HFA)': 'hepatic_fat_accumulation',
            'Glucose': 'glucose',
            'Total Cholesterol (TC)': 'total_cholesterol',
            'Low Density Lipoprotein (LDL)': 'low_density_lipoprotein',
            'High Density Lipoprotein (HDL)': 'high_density_lipoprotein',
            'Triglyceride': 'triglyceride',
            'Aspartat Aminotransferaz (AST)': 'aspartat_aminotransferaz',
            'Alanin Aminotransferaz (ALT)': 'alanin_aminotransferaz',
            'Alkaline Phosphatase (ALP)': 'alkaline_phosphatase',
            'Creatinine': 'creatinine',
            'Glomerular Filtration Rate (GFR)': 'glomerular_filtration_rate',
            'C-Reactive Protein (CRP)': 'c_reactive_protein',
            'Hemoglobin (HGB)': 'hemoglobin',
            'Vitamin D': 'vitamin_d'
        }
        
        df = df.rename(columns=column_mapping)
        self.log_action('Standardized column names', f'{len(column_mapping)} columns renamed')
        
        # 2. Check for missing values
        print("\n2. Handling Missing Values")
        print("-" * 70)
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            self.log_action('Missing values found', f'{missing_count} total missing values')
            # Handle missing values (median for numeric, mode for categorical)
            for col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    if df[col].dtype in ['float64', 'int64']:
                        median_val = df[col].median()
                        df[col] = df[col].fillna(median_val)
                        self.log_action('Imputed missing', f'{col}: {missing} values with median ({median_val:.2f})')
                    else:
                        mode_series = df[col].mode()
                        if len(mode_series) > 0:
                            mode_val = mode_series.iloc[0]
                            df[col] = df[col].fillna(mode_val)
                            self.log_action('Imputed missing', f'{col}: {missing} values with mode ({mode_val})')
        else:
            self.log_action('No missing values', 'Dataset is complete')
        
        # 3. Remove duplicates
        print("\n3. Removing Duplicates")
        print("-" * 70)
        
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            df = df.drop_duplicates(keep='first')
            self.log_action('Removed duplicates', f'{duplicates} rows ({duplicates/initial_rows*100:.2f}%)')
        else:
            self.log_action('No duplicates', 'Dataset is clean')
        
        # 4. Validate data types
        print("\n4. Validating Data Types")
        print("-" * 70)
        
        # Ensure binary columns are 0/1
        binary_cols = ['gallstone_status', 'gender', 'comorbidity', 'coronary_artery_disease',
                      'hypothyroidism', 'hyperlipidemia', 'diabetes_mellitus', 'hepatic_fat_accumulation']
        
        for col in binary_cols:
            if col in df.columns:
                unique_vals = df[col].unique()
                if not all(val in [0, 1] for val in unique_vals):
                    self.log_action('Invalid binary values', f'{col}: {unique_vals}')
        
        # 5. Handle outliers (cap at 1.5*IQR)
        print("\n5. Handling Outliers")
        print("-" * 70)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['gallstone_status', 'gender', 'comorbidity', 'coronary_artery_disease',
                       'hypothyroidism', 'hyperlipidemia', 'diabetes_mellitus', 'hepatic_fat_accumulation']
        
        for col in numeric_cols:
            if col not in exclude_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                if outliers_count > 0:
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    self.log_action('Capped outliers', 
                                  f'{col}: {outliers_count} values ({outliers_count/len(df)*100:.2f}%)')
        
        # 6. Validate data ranges
        print("\n6. Validating Data Ranges")
        print("-" * 70)
        
        # Check for negative values where they shouldn't exist
        for col in numeric_cols:
            if col not in exclude_cols:
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    df[col] = df[col].abs()
                    self.log_action('Fixed negative values', f'{col}: {neg_count} values converted to absolute')
        
        # 7. Final validation
        print("\n7. Final Validation")
        print("-" * 70)
        
        self._validate_dataset(df)
        
        # 8. Save cleaned data
        output_path = OUTPUT_DIR / 'gallstone_cleaned.csv'
        df.to_csv(output_path, index=False)
        self.log_action('Saved', f'{len(df)} rows, {len(df.columns)} columns to {output_path.name}')
        
        print(f"\n[OK] Gallbladder dataset cleaned successfully")
        print(f"   Original: {initial_rows} rows, {initial_cols} columns")
        print(f"   Cleaned: {len(df)} rows, {len(df.columns)} columns")
        
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
        
        # Check data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 30:  # Should have at least 30 numeric columns
            issues.append(f"Expected more numeric columns, found {len(numeric_cols)}")
        
        if issues:
            print("   ⚠ Validation issues:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print("   ✓ All validation checks passed")
    
    def save_cleaning_report(self):
        """Save the cleaning report"""
        report_df = pd.DataFrame(self.cleaning_report)
        report_path = REPORTS_DIR / 'data_cleaning_report.csv'
        report_df.to_csv(report_path, index=False)
        print(f"\n[OK] Cleaning report saved to {report_path}")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("GALLBLADDER DISEASE DATASET - DATA CLEANING")
    print("="*80)
    
    cleaner = GallbladderDataCleaner()
    df = cleaner.clean_dataset()
    cleaner.save_cleaning_report()
    
    print("\n" + "="*80)
    print("DATA CLEANING COMPLETE")
    print("="*80)
    print(f"[OK] Cleaned dataset: {OUTPUT_DIR / 'gallstone_cleaned.csv'}")
    print(f"[OK] Cleaning report: {REPORTS_DIR / 'data_cleaning_report.csv'}")
    print("\nNext steps:")
    print("1. Review the cleaned dataset")
    print("2. Run exploratory_analysis_gallbladder.py for EDA")
    print("3. Run feature_engineering_gallbladder.py to create features")
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    main()

