"""
Feature Engineering for Gallstone Disease Prediction
====================================================
This script creates domain-specific engineered features from the cleaned dataset.

Features Created:
1. Body Composition Ratios
2. Metabolic Health Indicators
3. Liver Function Markers
4. Cardiovascular Risk Factors
5. Interaction Features
6. Polynomial Features for key predictors
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'ml_ready'
REPORTS_DIR = BASE_DIR / 'reports' / 'gallstone'

# Create reports directory if it doesn't exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load the cleaned dataset"""
    print("Loading cleaned data...")
    df = pd.read_csv(DATA_DIR / 'gallstone_cleaned.csv')
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    return df

def create_body_composition_features(df):
    """Create body composition-related features"""
    print("\nCreating body composition features...")
    
    # Fat to lean mass ratio
    df['fat_lean_ratio'] = df['total_fat_content'] / (df['lean_mass'] + 1e-6)
    
    # Visceral to subcutaneous fat ratio (proxy)
    df['visceral_subcutaneous_ratio'] = df['visceral_fat_area'] / (df['total_fat_content'] - df['visceral_fat_area'] + 1e-6)
    
    # Body water distribution
    df['water_distribution_ratio'] = df['extracellular_water'] / (df['intracellular_water'] + 1e-6)
    
    # Muscle to fat ratio
    df['muscle_fat_ratio'] = df['muscle_mass'] / (df['total_fat_content'] + 1e-6)
    
    # Protein to weight ratio
    df['protein_weight_ratio'] = df['body_protein_content'] / (df['weight'] + 1e-6)
    
    # Body water percentage
    df['body_water_percentage'] = (df['total_body_water'] / df['weight']) * 100
    
    # Lean body mass index
    df['lean_bmi'] = df['lean_mass'] / ((df['height'] / 100) ** 2)
    
    # Fat mass index
    df['fat_mass_index'] = df['total_fat_content'] / ((df['height'] / 100) ** 2)
    
    return df

def create_metabolic_features(df):
    """Create metabolic health indicators"""
    print("Creating metabolic features...")
    
    # Atherogenic index (TC/HDL ratio)
    df['atherogenic_index'] = df['total_cholesterol'] / (df['high_density_lipoprotein'] + 1e-6)
    
    # LDL/HDL ratio
    df['ldl_hdl_ratio'] = df['low_density_lipoprotein'] / (df['high_density_lipoprotein'] + 1e-6)
    
    # Non-HDL cholesterol
    df['non_hdl_cholesterol'] = df['total_cholesterol'] - df['high_density_lipoprotein']
    
    # Triglyceride to HDL ratio (insulin resistance marker)
    df['tg_hdl_ratio'] = df['triglyceride'] / (df['high_density_lipoprotein'] + 1e-6)
    
    # Cholesterol saturation index (proxy)
    df['cholesterol_saturation'] = (df['total_cholesterol'] * df['triglyceride']) / (df['high_density_lipoprotein'] + 1e-6)
    
    # Metabolic syndrome score (composite)
    df['metabolic_syndrome_score'] = (
        (df['body_mass_index'] > 25).astype(int) +
        (df['triglyceride'] > 150).astype(int) +
        (df['high_density_lipoprotein'] < 40).astype(int) +
        (df['glucose'] > 100).astype(int)
    )
    
    # Lipid burden
    df['lipid_burden'] = df['total_cholesterol'] + df['triglyceride'] + df['low_density_lipoprotein']
    
    return df

def create_liver_function_features(df):
    """Create liver function markers"""
    print("Creating liver function features...")
    
    # AST/ALT ratio (De Ritis ratio)
    df['ast_alt_ratio'] = df['aspartat_aminotransferaz'] / (df['alanin_aminotransferaz'] + 1e-6)
    
    # Liver enzyme elevation score
    df['liver_enzyme_score'] = (
        (df['aspartat_aminotransferaz'] > 40).astype(int) +
        (df['alanin_aminotransferaz'] > 40).astype(int) +
        (df['alkaline_phosphatase'] > 120).astype(int)
    )
    
    # Combined liver enzyme index
    df['combined_liver_enzyme'] = df['aspartat_aminotransferaz'] + df['alanin_aminotransferaz'] + df['alkaline_phosphatase']
    
    # Hepatic fat accumulation interaction with BMI
    df['hepatic_fat_bmi_interaction'] = df['hepatic_fat_accumulation'] * df['body_mass_index']
    
    # Liver stress indicator
    df['liver_stress_indicator'] = (df['aspartat_aminotransferaz'] * df['alanin_aminotransferaz']) / (df['alkaline_phosphatase'] + 1e-6)
    
    return df

def create_cardiovascular_features(df):
    """Create cardiovascular risk factors"""
    print("Creating cardiovascular features...")
    
    # Framingham risk score components
    df['age_cholesterol_interaction'] = df['age'] * df['total_cholesterol']
    df['age_hdl_interaction'] = df['age'] * df['high_density_lipoprotein']
    
    # Cardiovascular risk score
    df['cv_risk_score'] = (
        (df['age'] > 50).astype(int) +
        (df['body_mass_index'] > 30).astype(int) +
        (df['total_cholesterol'] > 200).astype(int) +
        (df['high_density_lipoprotein'] < 40).astype(int) +
        df['coronary_artery_disease'] +
        df['diabetes_mellitus'] +
        df['hyperlipidemia']
    )
    
    # Visceral adiposity index (VAI) - simplified
    df['visceral_adiposity_index'] = (
        (df['visceral_fat_area'] / df['body_mass_index']) * 
        (df['triglyceride'] / df['high_density_lipoprotein'])
    )
    
    return df

def create_renal_features(df):
    """Create kidney function features"""
    print("Creating renal function features...")
    
    # Creatinine to GFR ratio
    df['creatinine_gfr_ratio'] = df['creatinine'] / (df['glomerular_filtration_rate'] + 1e-6)
    
    # Kidney function category (2=impaired, 1=mildly reduced, 0=normal)
    df['kidney_function_category'] = 0  # default normal
    df.loc[df['glomerular_filtration_rate'] < 60, 'kidney_function_category'] = 2  # impaired
    df.loc[(df['glomerular_filtration_rate'] >= 60) & (df['glomerular_filtration_rate'] < 90), 'kidney_function_category'] = 1  # mildly reduced
    
    return df

def create_inflammatory_features(df):
    """Create inflammation markers"""
    print("Creating inflammatory features...")
    
    # Inflammation score (if CRP is available)
    if df['c_reactive_protein'].notna().any():
        df['inflammation_score'] = (
            (df['c_reactive_protein'] > 3).astype(int) +
            (df['visceral_fat_rating'] > 10).astype(int)
        )
    else:
        df['inflammation_score'] = (df['visceral_fat_rating'] > 10).astype(int)
    
    # Visceral fat inflammation proxy
    df['visceral_inflammation_proxy'] = df['visceral_fat_area'] * df['body_mass_index']
    
    return df

def create_nutritional_features(df):
    """Create nutritional status features"""
    print("Creating nutritional features...")
    
    # Vitamin D deficiency indicator
    df['vitamin_d_deficiency'] = (df['vitamin_d'] < 30).astype(int)
    
    # Anemia indicator
    df['anemia_indicator'] = (df['hemoglobin'] < 13).astype(int)
    
    # Nutritional risk score
    df['nutritional_risk_score'] = (
        df['vitamin_d_deficiency'] +
        df['anemia_indicator'] +
        (df['body_protein_content'] < df['body_protein_content'].median()).astype(int)
    )
    
    return df

def create_age_related_features(df):
    """Create age-related features"""
    print("Creating age-related features...")
    
    # Age groups (0: <40, 1: 40-50, 2: 50-60, 3: >60)
    df['age_group'] = 0  # default <40
    df.loc[df['age'] >= 40, 'age_group'] = 1
    df.loc[df['age'] >= 50, 'age_group'] = 2
    df.loc[df['age'] >= 60, 'age_group'] = 3
    
    # Age-BMI interaction
    df['age_bmi_interaction'] = df['age'] * df['body_mass_index']
    
    # Age-visceral fat interaction
    df['age_visceral_interaction'] = df['age'] * df['visceral_fat_rating']
    
    return df

def create_comorbidity_features(df):
    """Create comorbidity-related features"""
    print("Creating comorbidity features...")
    
    # Total comorbidity count
    df['total_comorbidities'] = (
        df['comorbidity'] +
        df['coronary_artery_disease'] +
        df['hypothyroidism'] +
        df['hyperlipidemia'] +
        df['diabetes_mellitus']
    )
    
    # Metabolic disease burden
    df['metabolic_disease_burden'] = (
        df['diabetes_mellitus'] +
        df['hyperlipidemia'] +
        (df['body_mass_index'] > 30).astype(int)
    )
    
    return df

def create_interaction_features(df):
    """Create important interaction features based on domain knowledge"""
    print("Creating interaction features...")
    
    # BMI and lipid interactions
    df['bmi_triglyceride_interaction'] = df['body_mass_index'] * df['triglyceride']
    df['bmi_cholesterol_interaction'] = df['body_mass_index'] * df['total_cholesterol']
    
    # Visceral fat and metabolic interactions
    df['visceral_glucose_interaction'] = df['visceral_fat_rating'] * df['glucose']
    df['visceral_triglyceride_interaction'] = df['visceral_fat_rating'] * df['triglyceride']
    
    # Body composition and metabolic interactions
    df['fat_ratio_glucose_interaction'] = df['total_body_fat_ratio'] * df['glucose']
    df['fat_ratio_cholesterol_interaction'] = df['total_body_fat_ratio'] * df['total_cholesterol']
    
    return df

def create_polynomial_features(df):
    """Create polynomial features for key predictors"""
    print("Creating polynomial features...")
    
    # Squared features for non-linear relationships
    key_features = [
        'body_mass_index', 'age', 'visceral_fat_rating',
        'total_cholesterol', 'triglyceride', 'glucose'
    ]
    
    for feature in key_features:
        df[f'{feature}_squared'] = df[feature] ** 2
        df[f'{feature}_cubed'] = df[feature] ** 3
    
    return df

def create_risk_scores(df):
    """Create composite risk scores"""
    print("Creating composite risk scores...")
    
    # Gallstone risk score (based on known risk factors)
    df['gallstone_risk_score'] = (
        (df['age'] > 40).astype(int) * 2 +
        (df['gender'] == 1).astype(int) * 1.5 +  # Female
        (df['body_mass_index'] > 30).astype(int) * 2 +
        (df['total_cholesterol'] > 200).astype(int) * 1 +
        (df['triglyceride'] > 150).astype(int) * 1.5 +
        df['diabetes_mellitus'] * 1.5 +
        df['hepatic_fat_accumulation'] * 2
    )
    
    # Metabolic health score (inverse - lower is better)
    df['metabolic_health_score'] = (
        df['body_mass_index'] / 25 +
        df['total_cholesterol'] / 200 +
        df['triglyceride'] / 150 +
        df['glucose'] / 100 +
        (200 - df['high_density_lipoprotein']) / 50
    ) / 5
    
    return df

def create_feature_engineering_report(df_original, df_engineered):
    """Create a report of engineered features"""
    print("\nCreating feature engineering report...")
    
    # Get new features
    original_cols = set(df_original.columns)
    new_cols = [col for col in df_engineered.columns if col not in original_cols]
    
    # Create report
    report_data = []
    
    for col in new_cols:
        report_data.append({
            'Feature': col,
            'Type': 'Engineered',
            'Mean': df_engineered[col].mean(),
            'Std': df_engineered[col].std(),
            'Min': df_engineered[col].min(),
            'Max': df_engineered[col].max(),
            'Missing': df_engineered[col].isna().sum(),
            'Unique_Values': df_engineered[col].nunique()
        })
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(REPORTS_DIR / 'engineered_features_report.csv', index=False)
    print(f"Created {len(new_cols)} new features")
    
    # Feature categories summary
    categories = {
        'Body Composition': [col for col in new_cols if any(x in col.lower() for x in ['fat', 'lean', 'muscle', 'water', 'protein', 'bmi'])],
        'Metabolic': [col for col in new_cols if any(x in col.lower() for x in ['cholesterol', 'lipid', 'glucose', 'metabolic', 'hdl', 'ldl', 'triglyceride'])],
        'Liver Function': [col for col in new_cols if any(x in col.lower() for x in ['liver', 'ast', 'alt', 'hepatic'])],
        'Cardiovascular': [col for col in new_cols if any(x in col.lower() for x in ['cv', 'cardiovascular', 'atherogenic'])],
        'Renal': [col for col in new_cols if any(x in col.lower() for x in ['kidney', 'creatinine', 'gfr', 'renal'])],
        'Inflammatory': [col for col in new_cols if any(x in col.lower() for x in ['inflammation', 'visceral'])],
        'Nutritional': [col for col in new_cols if any(x in col.lower() for x in ['vitamin', 'anemia', 'nutritional'])],
        'Age-Related': [col for col in new_cols if 'age' in col.lower() and col != 'age'],
        'Comorbidity': [col for col in new_cols if 'comorbidity' in col.lower() or 'disease' in col.lower()],
        'Interaction': [col for col in new_cols if 'interaction' in col.lower()],
        'Polynomial': [col for col in new_cols if any(x in col.lower() for x in ['squared', 'cubed'])],
        'Risk Scores': [col for col in new_cols if 'score' in col.lower() or 'risk' in col.lower()]
    }
    
    category_summary = []
    for category, features in categories.items():
        category_summary.append({
            'Category': category,
            'Feature_Count': len(features),
            'Features': ', '.join(features[:5]) + ('...' if len(features) > 5 else '')
        })
    
    category_df = pd.DataFrame(category_summary)
    category_df.to_csv(REPORTS_DIR / 'feature_categories_summary.csv', index=False)
    
    return report_df, category_df

def main():
    """Main execution function"""
    print("=" * 80)
    print("GALLSTONE DISEASE - FEATURE ENGINEERING")
    print("=" * 80)
    
    # Load data
    df = load_data()
    df_original = df.copy()
    
    # Apply feature engineering
    df = create_body_composition_features(df)
    df = create_metabolic_features(df)
    df = create_liver_function_features(df)
    df = create_cardiovascular_features(df)
    df = create_renal_features(df)
    df = create_inflammatory_features(df)
    df = create_nutritional_features(df)
    df = create_age_related_features(df)
    df = create_comorbidity_features(df)
    df = create_interaction_features(df)
    df = create_polynomial_features(df)
    df = create_risk_scores(df)
    
    # Handle any infinite or NaN values
    print("\nHandling infinite and NaN values...")
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with median for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # Create reports
    report_df, category_df = create_feature_engineering_report(df_original, df)
    
    # Save engineered dataset
    output_path = DATA_DIR / 'gallstone_engineered.csv'
    df.to_csv(output_path, index=False)
    print(f"\nEngineered dataset saved to: {output_path}")
    print(f"Total features: {len(df.columns)}")
    print(f"Original features: {len(df_original.columns)}")
    print(f"New features: {len(df.columns) - len(df_original.columns)}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 80)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFeature categories:")
    print(category_df.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Feature engineering completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()

# Made with Bob
