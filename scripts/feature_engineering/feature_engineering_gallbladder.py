"""
Gallbladder Disease Dataset - Feature Engineering Script
This script creates advanced features for the gallbladder dataset based on medical domain knowledge.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
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


print("="*80)
print("GALLBLADDER DISEASE DATASET - FEATURE ENGINEERING")
print("="*80)


def create_body_composition_features(df):
    """Create features based on body composition relationships"""
    features_created = []
    
    # Fat-lean ratios
    if 'total_fat_content' in df.columns and 'lean_mass' in df.columns:
        df['fat_lean_ratio'] = df['total_fat_content'] / (df['lean_mass'] + 1)
        features_created.append('fat_lean_ratio')
    
    # Visceral-subcutaneous fat ratio
    if 'visceral_fat_area' in df.columns and 'total_fat_content' in df.columns:
        df['visceral_subcutaneous_ratio'] = df['visceral_fat_area'] / (df['total_fat_content'] + 1)
        features_created.append('visceral_subcutaneous_ratio')
    
    # Water distribution
    if 'extracellular_water' in df.columns and 'intracellular_water' in df.columns:
        df['water_distribution_ratio'] = df['extracellular_water'] / (df['intracellular_water'] + 1)
        features_created.append('water_distribution_ratio')
    
    # Muscle-fat ratio
    if 'muscle_mass' in df.columns and 'total_fat_content' in df.columns:
        df['muscle_fat_ratio'] = df['muscle_mass'] / (df['total_fat_content'] + 1)
        features_created.append('muscle_fat_ratio')
    
    # Body water percentage
    if 'total_body_water' in df.columns and 'weight' in df.columns:
        df['body_water_percentage'] = (df['total_body_water'] / df['weight']) * 100
        features_created.append('body_water_percentage')
    
    # Lean BMI
    if 'lean_mass' in df.columns and 'height' in df.columns and 'weight' in df.columns:
        lean_weight = (df['lean_mass'] / 100) * df['weight']
        df['lean_bmi'] = lean_weight / ((df['height'] / 100) ** 2)
        features_created.append('lean_bmi')
    
    return df, features_created


def create_metabolic_features(df):
    """Create features related to lipid metabolism and metabolic syndrome"""
    features_created = []
    
    # Atherogenic index
    if 'total_cholesterol' in df.columns and 'high_density_lipoprotein' in df.columns:
        df['atherogenic_index'] = df['total_cholesterol'] / (df['high_density_lipoprotein'] + 1)
        features_created.append('atherogenic_index')
    
    # LDL/HDL ratio
    if 'low_density_lipoprotein' in df.columns and 'high_density_lipoprotein' in df.columns:
        df['ldl_hdl_ratio'] = df['low_density_lipoprotein'] / (df['high_density_lipoprotein'] + 1)
        features_created.append('ldl_hdl_ratio')
    
    # Non-HDL cholesterol
    if 'total_cholesterol' in df.columns and 'high_density_lipoprotein' in df.columns:
        df['non_hdl_cholesterol'] = df['total_cholesterol'] - df['high_density_lipoprotein']
        features_created.append('non_hdl_cholesterol')
    
    # Triglyceride/HDL ratio (insulin resistance marker)
    if 'triglyceride' in df.columns and 'high_density_lipoprotein' in df.columns:
        df['tg_hdl_ratio'] = df['triglyceride'] / (df['high_density_lipoprotein'] + 1)
        features_created.append('tg_hdl_ratio')
    
    # Metabolic syndrome score
    score = 0
    if 'body_mass_index' in df.columns:
        score += (df['body_mass_index'] >= 30).astype(int)
    if 'triglyceride' in df.columns:
        score += (df['triglyceride'] >= 150).astype(int)
    if 'high_density_lipoprotein' in df.columns:
        score += (df['high_density_lipoprotein'] < 40).astype(int)
    if 'glucose' in df.columns:
        score += (df['glucose'] >= 100).astype(int)
    df['metabolic_syndrome_score'] = score
    features_created.append('metabolic_syndrome_score')
    
    return df, features_created


def create_liver_function_features(df):
    """Create features related to liver function"""
    features_created = []
    
    # AST/ALT ratio (De Ritis ratio)
    if 'aspartat_aminotransferaz' in df.columns and 'alanin_aminotransferaz' in df.columns:
        df['ast_alt_ratio'] = df['aspartat_aminotransferaz'] / (df['alanin_aminotransferaz'] + 1)
        features_created.append('ast_alt_ratio')
    
    # Liver enzyme score
    enzyme_score = 0
    if 'aspartat_aminotransferaz' in df.columns:
        enzyme_score += (df['aspartat_aminotransferaz'] > 40).astype(int)
    if 'alanin_aminotransferaz' in df.columns:
        enzyme_score += (df['alanin_aminotransferaz'] > 56).astype(int)
    if 'alkaline_phosphatase' in df.columns:
        enzyme_score += (df['alkaline_phosphatase'] > 120).astype(int)
    df['liver_enzyme_score'] = enzyme_score
    features_created.append('liver_enzyme_score')
    
    # Hepatic fat-BMI interaction
    if 'hepatic_fat_accumulation' in df.columns and 'body_mass_index' in df.columns:
        df['hepatic_fat_bmi_interaction'] = df['hepatic_fat_accumulation'] * df['body_mass_index']
        features_created.append('hepatic_fat_bmi_interaction')
    
    return df, features_created


def create_inflammatory_features(df):
    """Create features related to inflammation"""
    features_created = []
    
    # Visceral inflammation proxy
    if 'visceral_fat_rating' in df.columns and 'body_mass_index' in df.columns:
        df['visceral_inflammation_proxy'] = df['visceral_fat_rating'] * df['body_mass_index']
        features_created.append('visceral_inflammation_proxy')
    
    # Visceral adiposity index (simplified)
    if 'visceral_fat_area' in df.columns and 'triglyceride' in df.columns:
        df['visceral_adiposity_index'] = df['visceral_fat_area'] * df['triglyceride'] / 100
        features_created.append('visceral_adiposity_index')
    
    return df, features_created


def create_age_related_features(df):
    """Create age-related interaction features"""
    features_created = []
    
    # Age groups
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 100], labels=[0, 1, 2, 3])
        df['age_group'] = df['age_group'].astype(int)
        features_created.append('age_group')
    
    # Age interactions
    if 'age' in df.columns and 'body_mass_index' in df.columns:
        df['age_bmi_interaction'] = df['age'] * df['body_mass_index']
        features_created.append('age_bmi_interaction')
    
    if 'age' in df.columns and 'visceral_fat_rating' in df.columns:
        df['age_visceral_interaction'] = df['age'] * df['visceral_fat_rating']
        features_created.append('age_visceral_interaction')
    
    if 'age' in df.columns and 'total_cholesterol' in df.columns:
        df['age_cholesterol_interaction'] = df['age'] * df['total_cholesterol']
        features_created.append('age_cholesterol_interaction')
    
    return df, features_created


def create_comorbidity_features(df):
    """Create comorbidity-related features"""
    features_created = []
    
    # Total comorbidities
    comorbidity_cols = ['comorbidity', 'coronary_artery_disease', 'hypothyroidism', 
                       'hyperlipidemia', 'diabetes_mellitus']
    available_cols = [col for col in comorbidity_cols if col in df.columns]
    if available_cols:
        df['total_comorbidities'] = df[available_cols].sum(axis=1)
        features_created.append('total_comorbidities')
    
    # Metabolic disease burden
    metabolic_cols = ['diabetes_mellitus', 'hyperlipidemia']
    available_metabolic = [col for col in metabolic_cols if col in df.columns]
    if available_metabolic:
        df['metabolic_disease_burden'] = df[available_metabolic].sum(axis=1)
        features_created.append('metabolic_disease_burden')
    
    return df, features_created


def create_interaction_features(df):
    """Create two-way interaction features"""
    features_created = []
    
    # BMI-lipid interactions
    if 'body_mass_index' in df.columns and 'triglyceride' in df.columns:
        df['bmi_triglyceride_interaction'] = df['body_mass_index'] * df['triglyceride']
        features_created.append('bmi_triglyceride_interaction')
    
    if 'body_mass_index' in df.columns and 'total_cholesterol' in df.columns:
        df['bmi_cholesterol_interaction'] = df['body_mass_index'] * df['total_cholesterol']
        features_created.append('bmi_cholesterol_interaction')
    
    # Visceral fat interactions
    if 'visceral_fat_rating' in df.columns and 'glucose' in df.columns:
        df['visceral_glucose_interaction'] = df['visceral_fat_rating'] * df['glucose']
        features_created.append('visceral_glucose_interaction')
    
    if 'visceral_fat_rating' in df.columns and 'triglyceride' in df.columns:
        df['visceral_triglyceride_interaction'] = df['visceral_fat_rating'] * df['triglyceride']
        features_created.append('visceral_triglyceride_interaction')
    
    return df, features_created


def create_polynomial_features(df):
    """Create polynomial features for key predictors"""
    features_created = []
    
    key_features = ['body_mass_index', 'age', 'visceral_fat_rating', 
                   'total_cholesterol', 'triglyceride', 'glucose']
    
    for feat in key_features:
        if feat in df.columns:
            df[f'{feat}_squared'] = df[feat] ** 2
            df[f'{feat}_cubed'] = df[feat] ** 3
            features_created.extend([f'{feat}_squared', f'{feat}_cubed'])
    
    return df, features_created


def create_risk_scores(df):
    """Create composite risk scores"""
    features_created = []
    
    # Gallstone risk score (weighted composite)
    risk_score = 0
    if 'age' in df.columns:
        risk_score += (df['age'] / 100) * 2  # Age weight: 2
    if 'body_mass_index' in df.columns:
        risk_score += (df['body_mass_index'] / 50) * 3  # BMI weight: 3
    if 'total_cholesterol' in df.columns:
        risk_score += (df['total_cholesterol'] / 300) * 2  # Cholesterol weight: 2
    if 'triglyceride' in df.columns:
        risk_score += (df['triglyceride'] / 200) * 2  # Triglyceride weight: 2
    if 'diabetes_mellitus' in df.columns:
        risk_score += df['diabetes_mellitus'] * 1.5  # Diabetes weight: 1.5
    
    df['gallstone_risk_score'] = risk_score
    features_created.append('gallstone_risk_score')
    
    return df, features_created


def main():
    """Main execution function"""
    print("\nLoading cleaned dataset...")
    
    # Load cleaned data
    input_path = DATA_DIR / 'gallstone_cleaned.csv'
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        print("Please run data_cleaning_gallbladder.py first")
        return None
    
    df = pd.read_csv(input_path)
    initial_features = len(df.columns)
    print(f"[OK] Loaded: {len(df)} rows, {initial_features} columns")
    
    # Apply feature engineering
    all_features_created = []
    
    print("\n" + "="*80)
    print("CREATING ENGINEERED FEATURES")
    print("="*80)
    
    print("\n1. Body Composition Features...")
    df, features = create_body_composition_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n2. Metabolic Features...")
    df, features = create_metabolic_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n3. Liver Function Features...")
    df, features = create_liver_function_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n4. Inflammatory Features...")
    df, features = create_inflammatory_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n5. Age-Related Features...")
    df, features = create_age_related_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n6. Comorbidity Features...")
    df, features = create_comorbidity_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n7. Interaction Features...")
    df, features = create_interaction_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n8. Polynomial Features...")
    df, features = create_polynomial_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n9. Risk Scores...")
    df, features = create_risk_scores(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill any remaining NaN values with median
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
    
    # Save engineered dataset
    output_path = OUTPUT_DIR / 'gallstone_engineered.csv'
    df.to_csv(output_path, index=False)
    
    # Generate report
    final_features = len(df.columns)
    new_features = final_features - initial_features
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"Original features: {initial_features}")
    print(f"New features created: {new_features}")
    print(f"Total features: {final_features}")
    print(f"\n[OK] Engineered dataset saved: {output_path}")
    
    # Save feature list
    feature_report = pd.DataFrame({
        'Feature_Name': all_features_created,
        'Feature_Number': range(1, len(all_features_created) + 1)
    })
    feature_report.to_csv(REPORTS_DIR / 'engineered_features_list.csv', index=False)
    print(f"[OK] Feature list saved: {REPORTS_DIR / 'engineered_features_list.csv'}")
    
    print("\nNext steps:")
    print("1. Review the engineered features")
    print("2. Perform feature selection if needed")
    print("3. Train machine learning models")
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    main()


