"""
Heart Disease Dataset - Feature Engineering Script
This script creates advanced features for the heart dataset based on clinical domain knowledge.
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
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'datasets' / 'heart'
OUTPUT_DIR = BASE_DIR / 'datasets' / 'heart'
REPORTS_DIR = BASE_DIR / 'reports' / 'heart'

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


print("="*80)
print("HEART DISEASE DATASET - FEATURE ENGINEERING")
print("="*80)


def create_age_features(df):
    """Create age-based features"""
    features_created = []
    
    if 'age' not in df.columns:
        return df, features_created
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], 
                             bins=[0, 40, 50, 60, 70, 100],
                             labels=['<40', '40-50', '50-60', '60-70', '70+'])
    features_created.append('age_group')
    
    # Age risk categories
    df['age_high_risk'] = (df['age'] >= 55).astype(int)
    df['age_very_high_risk'] = (df['age'] >= 65).astype(int)
    features_created.extend(['age_high_risk', 'age_very_high_risk'])
    
    # Age squared (non-linear relationship)
    df['age_squared'] = df['age'] ** 2
    features_created.append('age_squared')
    
    return df, features_created


def create_blood_pressure_features(df):
    """Create blood pressure-based features"""
    features_created = []
    
    if 'resting_bp' not in df.columns:
        return df, features_created
    
    # BP categories (based on AHA guidelines)
    df['bp_normal'] = (df['resting_bp'] < 120).astype(int)
    df['bp_elevated'] = ((df['resting_bp'] >= 120) & (df['resting_bp'] < 130)).astype(int)
    df['bp_hypertension_stage1'] = ((df['resting_bp'] >= 130) & (df['resting_bp'] < 140)).astype(int)
    df['bp_hypertension_stage2'] = (df['resting_bp'] >= 140).astype(int)
    features_created.extend(['bp_normal', 'bp_elevated', 'bp_hypertension_stage1', 'bp_hypertension_stage2'])
    
    # BP risk indicator
    df['bp_high_risk'] = (df['resting_bp'] >= 140).astype(int)
    features_created.append('bp_high_risk')
    
    # BP deviation from normal
    df['bp_deviation'] = df['resting_bp'] - 120
    features_created.append('bp_deviation')
    
    return df, features_created


def create_cholesterol_features(df):
    """Create cholesterol-based features"""
    features_created = []
    
    if 'cholesterol' not in df.columns:
        return df, features_created
    
    # Cholesterol categories (based on AHA guidelines)
    df['chol_desirable'] = (df['cholesterol'] < 200).astype(int)
    df['chol_borderline_high'] = ((df['cholesterol'] >= 200) & (df['cholesterol'] < 240)).astype(int)
    df['chol_high'] = (df['cholesterol'] >= 240).astype(int)
    features_created.extend(['chol_desirable', 'chol_borderline_high', 'chol_high'])
    
    # Cholesterol risk indicator
    df['chol_high_risk'] = (df['cholesterol'] >= 240).astype(int)
    features_created.append('chol_high_risk')
    
    # Cholesterol deviation from optimal
    df['chol_deviation'] = df['cholesterol'] - 200
    features_created.append('chol_deviation')
    
    return df, features_created


def create_heart_rate_features(df):
    """Create heart rate-based features"""
    features_created = []
    
    if 'max_heart_rate' not in df.columns:
        return df, features_created
    
    # Maximum heart rate categories
    df['hr_low'] = (df['max_heart_rate'] < 100).astype(int)
    df['hr_normal'] = ((df['max_heart_rate'] >= 100) & (df['max_heart_rate'] <= 170)).astype(int)
    df['hr_high'] = (df['max_heart_rate'] > 170).astype(int)
    features_created.extend(['hr_low', 'hr_normal', 'hr_high'])
    
    # Age-predicted max heart rate (220 - age)
    if 'age' in df.columns:
        df['predicted_max_hr'] = 220 - df['age']
        df['hr_reserve'] = df['max_heart_rate'] - df['predicted_max_hr']
        df['hr_percentage'] = (df['max_heart_rate'] / df['predicted_max_hr']) * 100
        features_created.extend(['predicted_max_hr', 'hr_reserve', 'hr_percentage'])
    
    return df, features_created


def create_st_depression_features(df):
    """Create ST depression-based features"""
    features_created = []
    
    if 'st_depression' not in df.columns:
        return df, features_created
    
    # ST depression categories
    df['st_normal'] = (df['st_depression'] == 0).astype(int)
    df['st_mild'] = ((df['st_depression'] > 0) & (df['st_depression'] <= 1)).astype(int)
    df['st_moderate'] = ((df['st_depression'] > 1) & (df['st_depression'] <= 2)).astype(int)
    df['st_severe'] = (df['st_depression'] > 2).astype(int)
    features_created.extend(['st_normal', 'st_mild', 'st_moderate', 'st_severe'])
    
    # ST depression risk
    df['st_high_risk'] = (df['st_depression'] > 1).astype(int)
    features_created.append('st_high_risk')
    
    return df, features_created


def create_interaction_features(df):
    """Create interaction features between important variables"""
    features_created = []
    
    # Age and cholesterol interaction
    if 'age' in df.columns and 'cholesterol' in df.columns:
        df['age_chol_interaction'] = df['age'] * df['cholesterol']
        df['age_chol_ratio'] = df['age'] / (df['cholesterol'] + 1)
        features_created.extend(['age_chol_interaction', 'age_chol_ratio'])
    
    # Age and blood pressure interaction
    if 'age' in df.columns and 'resting_bp' in df.columns:
        df['age_bp_interaction'] = df['age'] * df['resting_bp']
        df['age_bp_ratio'] = df['age'] / (df['resting_bp'] + 1)
        features_created.extend(['age_bp_interaction', 'age_bp_ratio'])
    
    # Cholesterol and blood pressure interaction
    if 'cholesterol' in df.columns and 'resting_bp' in df.columns:
        df['chol_bp_interaction'] = df['cholesterol'] * df['resting_bp']
        df['chol_bp_ratio'] = df['cholesterol'] / (df['resting_bp'] + 1)
        features_created.extend(['chol_bp_interaction', 'chol_bp_ratio'])
    
    # Heart rate and ST depression interaction
    if 'max_heart_rate' in df.columns and 'st_depression' in df.columns:
        df['hr_st_interaction'] = df['max_heart_rate'] * df['st_depression']
        features_created.append('hr_st_interaction')
    
    # Exercise angina and ST depression
    if 'exercise_induced_angina' in df.columns and 'st_depression' in df.columns:
        df['angina_st_interaction'] = df['exercise_induced_angina'] * df['st_depression']
        features_created.append('angina_st_interaction')
    
    return df, features_created


def create_risk_score_features(df):
    """Create composite risk score features"""
    features_created = []
    
    # Cardiovascular risk score (simple additive model)
    risk_components = []
    
    if 'age_high_risk' in df.columns:
        risk_components.append('age_high_risk')
    if 'bp_high_risk' in df.columns:
        risk_components.append('bp_high_risk')
    if 'chol_high_risk' in df.columns:
        risk_components.append('chol_high_risk')
    if 'st_high_risk' in df.columns:
        risk_components.append('st_high_risk')
    if 'exercise_induced_angina' in df.columns:
        risk_components.append('exercise_induced_angina')
    
    if risk_components:
        df['cv_risk_score'] = df[risk_components].sum(axis=1)
        df['cv_high_risk'] = (df['cv_risk_score'] >= 3).astype(int)
        features_created.extend(['cv_risk_score', 'cv_high_risk'])
    
    return df, features_created


def create_chest_pain_features(df):
    """Create chest pain type features"""
    features_created = []
    
    if 'chest_pain_type' not in df.columns:
        return df, features_created
    
    # One-hot encode chest pain types
    df['cp_typical_angina'] = (df['chest_pain_type'] == 1).astype(int)
    df['cp_atypical_angina'] = (df['chest_pain_type'] == 2).astype(int)
    df['cp_non_anginal'] = (df['chest_pain_type'] == 3).astype(int)
    df['cp_asymptomatic'] = (df['chest_pain_type'] == 4).astype(int)
    features_created.extend(['cp_typical_angina', 'cp_atypical_angina', 'cp_non_anginal', 'cp_asymptomatic'])
    
    # Angina indicator (typical or atypical)
    df['has_angina'] = ((df['chest_pain_type'] == 1) | (df['chest_pain_type'] == 2)).astype(int)
    features_created.append('has_angina')
    
    return df, features_created


def create_vessel_features(df):
    """Create major vessels features"""
    features_created = []
    
    if 'num_major_vessels' not in df.columns:
        return df, features_created
    
    # Vessel categories
    df['vessels_none'] = (df['num_major_vessels'] == 0).astype(int)
    df['vessels_one'] = (df['num_major_vessels'] == 1).astype(int)
    df['vessels_multiple'] = (df['num_major_vessels'] >= 2).astype(int)
    features_created.extend(['vessels_none', 'vessels_one', 'vessels_multiple'])
    
    # High risk vessel count
    df['vessels_high_risk'] = (df['num_major_vessels'] >= 2).astype(int)
    features_created.append('vessels_high_risk')
    
    return df, features_created


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    # Load cleaned dataset
    input_file = DATA_DIR / 'heart_cleaned.csv'
    if not input_file.exists():
        input_file = DATA_DIR / 'heart_master_dataset.csv'
        print(f"Using master dataset: {input_file}")
    else:
        print(f"Using cleaned dataset: {input_file}")
    
    df = pd.read_csv(input_file)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    initial_features = len(df.columns)
    all_features_created = []
    
    # Create features
    print("\n" + "="*80)
    print("CREATING FEATURES")
    print("="*80)
    
    print("\n1. Age-based features")
    df, features = create_age_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features: {', '.join(features)}")
    
    print("\n2. Blood pressure features")
    df, features = create_blood_pressure_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features: {', '.join(features)}")
    
    print("\n3. Cholesterol features")
    df, features = create_cholesterol_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features: {', '.join(features)}")
    
    print("\n4. Heart rate features")
    df, features = create_heart_rate_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features: {', '.join(features)}")
    
    print("\n5. ST depression features")
    df, features = create_st_depression_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features: {', '.join(features)}")
    
    print("\n6. Chest pain features")
    df, features = create_chest_pain_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features: {', '.join(features)}")
    
    print("\n7. Vessel features")
    df, features = create_vessel_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features: {', '.join(features)}")
    
    print("\n8. Interaction features")
    df, features = create_interaction_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features: {', '.join(features)}")
    
    print("\n9. Risk score features")
    df, features = create_risk_score_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features: {', '.join(features)}")
    
    # Save engineered dataset
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_file = OUTPUT_DIR / 'heart_engineered.csv'
    df.to_csv(output_file, index=False)
    print(f"\n[OK] Engineered dataset saved: {output_file}")
    print(f"     Total features: {len(df.columns)} (added {len(df.columns) - initial_features})")
    
    # Save feature engineering report
    report = []
    report.append("="*80)
    report.append("HEART DISEASE DATASET - FEATURE ENGINEERING REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"Initial features: {initial_features}")
    report.append(f"Features created: {len(all_features_created)}")
    report.append(f"Final features: {len(df.columns)}")
    report.append("")
    report.append("CREATED FEATURES:")
    report.append("-"*80)
    for i, feat in enumerate(all_features_created, 1):
        report.append(f"{i:3d}. {feat}")
    report.append("")
    report.append("="*80)
    
    report_file = REPORTS_DIR / 'FEATURE_ENGINEERING_SUMMARY.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f"[OK] Feature engineering report saved: {report_file}")
    
    print("\n" + "="*80)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nTotal features created: {len(all_features_created)}")
    print(f"Output saved to: {output_file}")
    print("\nNext steps:")
    print("1. Perform feature selection")
    print("2. Create train/test splits")
    print("3. Train machine learning models")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()

# Made with Bob
