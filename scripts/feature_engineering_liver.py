"""
BUPA Liver Disorders Dataset - Feature Engineering Script
This script creates advanced features for the liver dataset based on clinical domain knowledge.
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
DATA_DIR = BASE_DIR / 'datasets' / 'liver'
OUTPUT_DIR = BASE_DIR / 'datasets' / 'liver'
REPORTS_DIR = BASE_DIR / 'reports' / 'liver'

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


print("="*80)
print("BUPA LIVER DISORDERS DATASET - FEATURE ENGINEERING")
print("="*80)


def create_enzyme_features(df):
    """Create enzyme-based features"""
    features_created = []
    
    # Enzyme ratios
    if 'sgpt' in df.columns and 'sgot' in df.columns:
        df['sgpt_sgot_ratio'] = df['sgpt'] / (df['sgot'] + 1)
        df['sgot_sgpt_ratio'] = df['sgot'] / (df['sgpt'] + 1)
        features_created.extend(['sgpt_sgot_ratio', 'sgot_sgpt_ratio'])
    
    if 'gammagt' in df.columns and 'sgpt' in df.columns:
        df['gammagt_sgpt_ratio'] = df['gammagt'] / (df['sgpt'] + 1)
        features_created.append('gammagt_sgpt_ratio')
    
    if 'gammagt' in df.columns and 'sgot' in df.columns:
        df['gammagt_sgot_ratio'] = df['gammagt'] / (df['sgot'] + 1)
        features_created.append('gammagt_sgot_ratio')
    
    # Enzyme combinations
    if 'sgpt' in df.columns and 'sgot' in df.columns:
        df['sgpt_sgot_sum'] = df['sgpt'] + df['sgot']
        df['sgpt_sgot_product'] = df['sgpt'] * df['sgot']
        features_created.extend(['sgpt_sgot_sum', 'sgpt_sgot_product'])
    
    # All enzymes sum and mean
    enzyme_cols = ['sgpt', 'sgot', 'gammagt', 'alkphos']
    available_enzymes = [col for col in enzyme_cols if col in df.columns]
    if available_enzymes:
        df['all_enzymes_sum'] = df[available_enzymes].sum(axis=1)
        df['all_enzymes_mean'] = df[available_enzymes].mean(axis=1)
        features_created.extend(['all_enzymes_sum', 'all_enzymes_mean'])
    
    # Enzyme dominance indicators
    if 'sgpt' in df.columns and 'sgot' in df.columns:
        df['sgpt_dominant'] = (df['sgpt'] > df['sgot']).astype(int)
        df['sgot_dominant'] = (df['sgot'] > df['sgpt']).astype(int)
        features_created.extend(['sgpt_dominant', 'sgot_dominant'])
    
    if 'gammagt' in df.columns and 'sgpt' in df.columns:
        df['gammagt_dominant'] = (df['gammagt'] > df['sgpt']).astype(int)
        features_created.append('gammagt_dominant')
    
    # Elevated enzyme indicators
    if 'sgpt' in df.columns:
        df['sgpt_elevated'] = (df['sgpt'] > 56).astype(int)
        features_created.append('sgpt_elevated')
    
    if 'sgot' in df.columns:
        df['sgot_elevated'] = (df['sgot'] > 40).astype(int)
        features_created.append('sgot_elevated')
    
    if 'gammagt' in df.columns:
        df['gammagt_elevated'] = (df['gammagt'] > 48).astype(int)
        features_created.append('gammagt_elevated')
    
    # Count of elevated enzymes
    elevated_cols = [col for col in ['sgpt_elevated', 'sgot_elevated', 'gammagt_elevated'] 
                    if col in df.columns]
    if elevated_cols:
        df['elevated_enzyme_count'] = df[elevated_cols].sum(axis=1)
        features_created.append('elevated_enzyme_count')
    
    return df, features_created


def create_alkphos_features(df):
    """Create alkaline phosphatase features"""
    features_created = []
    
    if 'alkphos' not in df.columns:
        return df, features_created
    
    # Ratios with other enzymes
    if 'sgpt' in df.columns:
        df['alkphos_sgpt_ratio'] = df['alkphos'] / (df['sgpt'] + 1)
        features_created.append('alkphos_sgpt_ratio')
    
    if 'sgot' in df.columns:
        df['alkphos_sgot_ratio'] = df['alkphos'] / (df['sgot'] + 1)
        features_created.append('alkphos_sgot_ratio')
    
    if 'gammagt' in df.columns:
        df['alkphos_gammagt_ratio'] = df['alkphos'] / (df['gammagt'] + 1)
        features_created.append('alkphos_gammagt_ratio')
    
    # Elevation indicators
    df['alkphos_elevated'] = (df['alkphos'] > 120).astype(int)
    df['alkphos_low'] = (df['alkphos'] < 30).astype(int)
    features_created.extend(['alkphos_elevated', 'alkphos_low'])
    
    # Interaction with MCV
    if 'mcv' in df.columns:
        df['alkphos_mcv_ratio'] = df['alkphos'] / (df['mcv'] + 1)
        df['alkphos_mcv_product'] = df['alkphos'] * df['mcv']
        features_created.extend(['alkphos_mcv_ratio', 'alkphos_mcv_product'])
    
    return df, features_created


def create_mcv_features(df):
    """Create MCV (Mean Corpuscular Volume) features"""
    features_created = []
    
    if 'mcv' not in df.columns:
        return df, features_created
    
    # MCV categories (normal: 80-100)
    df['mcv_low'] = (df['mcv'] < 80).astype(int)  # Microcytic
    df['mcv_normal'] = ((df['mcv'] >= 80) & (df['mcv'] <= 100)).astype(int)
    df['mcv_high'] = (df['mcv'] > 100).astype(int)  # Macrocytic
    features_created.extend(['mcv_low', 'mcv_normal', 'mcv_high'])
    
    # MCV deviation from normal
    df['mcv_deviation'] = df['mcv'] - 90  # 90 is midpoint of normal range
    df['mcv_normalized'] = (df['mcv'] - 80) / 20  # Normalize to 0-1 range
    features_created.extend(['mcv_deviation', 'mcv_normalized'])
    
    # MCV-alcohol interaction
    if 'drinks' in df.columns:
        df['mcv_drinks_interaction'] = df['mcv'] * df['drinks']
        features_created.append('mcv_drinks_interaction')
    
    if 'alcohol_category' in df.columns:
        # Encode alcohol category for interaction
        alcohol_encoding = {'None': 0, 'Light': 1, 'Moderate': 2, 'Heavy': 3, 'Very Heavy': 4}
        df['alcohol_category_encoded'] = df['alcohol_category'].map(alcohol_encoding)
        df['mcv_alcohol_category_encoded'] = df['mcv'] * df['alcohol_category_encoded']
        features_created.extend(['alcohol_category_encoded', 'mcv_alcohol_category_encoded'])
    
    return df, features_created


def create_alcohol_features(df):
    """Create alcohol-related features"""
    features_created = []
    
    if 'drinks' not in df.columns:
        return df, features_created
    
    # Polynomial features
    df['drinks_squared'] = df['drinks'] ** 2
    df['drinks_log'] = np.log1p(df['drinks'])
    df['drinks_sqrt'] = np.sqrt(df['drinks'])
    features_created.extend(['drinks_squared', 'drinks_log', 'drinks_sqrt'])
    
    # Alcohol level encoding
    df['alcohol_level'] = pd.cut(df['drinks'],
                                 bins=[-0.1, 0, 1, 2, 4, float('inf')],
                                 labels=[0, 1, 2, 3, 4])
    df['alcohol_level'] = df['alcohol_level'].astype(int)
    features_created.append('alcohol_level')
    
    # Interactions with enzymes
    if 'sgpt' in df.columns:
        df['drinks_sgpt_interaction'] = df['drinks'] * df['sgpt']
        features_created.append('drinks_sgpt_interaction')
    
    if 'sgot' in df.columns:
        df['drinks_sgot_interaction'] = df['drinks'] * df['sgot']
        features_created.append('drinks_sgot_interaction')
    
    if 'gammagt' in df.columns:
        df['drinks_gammagt_interaction'] = df['drinks'] * df['gammagt']
        features_created.append('drinks_gammagt_interaction')
    
    if 'ast_alt_ratio' in df.columns:
        df['drinks_ast_alt_ratio_interaction'] = df['drinks'] * df['ast_alt_ratio']
        features_created.append('drinks_ast_alt_ratio_interaction')
    
    # Heavy drinker indicator
    df['heavy_drinker'] = (df['drinks'] > 2).astype(int)
    features_created.append('heavy_drinker')
    
    # Alcohol-enzyme risk score
    risk = df['drinks'] / 10  # Normalize drinks
    if 'gammagt' in df.columns:
        risk += (df['gammagt'] > 48).astype(int)
    if 'ast_alt_ratio' in df.columns:
        risk += (df['ast_alt_ratio'] > 2).astype(int)
    df['alcohol_enzyme_risk'] = risk
    features_created.append('alcohol_enzyme_risk')
    
    return df, features_created


def create_clinical_risk_scores(df):
    """Create clinical risk scores"""
    features_created = []
    
    # Liver damage score
    damage_score = 0
    if 'sgpt_elevated' in df.columns:
        damage_score += df['sgpt_elevated'] * 2
    if 'sgot_elevated' in df.columns:
        damage_score += df['sgot_elevated'] * 2
    if 'gammagt_elevated' in df.columns:
        damage_score += df['gammagt_elevated']
    if 'ast_alt_ratio' in df.columns:
        damage_score += (df['ast_alt_ratio'] > 2).astype(int) * 3
    df['liver_damage_score'] = damage_score
    features_created.append('liver_damage_score')
    
    # Alcoholic liver disease risk
    alc_risk = 0
    if 'drinks' in df.columns:
        alc_risk += (df['drinks'] > 2).astype(int) * 2
    if 'ast_alt_ratio' in df.columns:
        alc_risk += (df['ast_alt_ratio'] > 2).astype(int) * 3
    if 'gammagt' in df.columns:
        alc_risk += (df['gammagt'] > 48).astype(int)
    if 'mcv' in df.columns:
        alc_risk += (df['mcv'] > 100).astype(int)
    df['alcoholic_liver_risk'] = alc_risk
    features_created.append('alcoholic_liver_risk')
    
    # Hepatocellular vs cholestatic pattern
    if 'sgpt' in df.columns and 'alkphos' in df.columns:
        df['hepatocellular_score'] = df['sgpt'] / (df['alkphos'] + 1)
        features_created.append('hepatocellular_score')
    
    if 'alkphos' in df.columns and 'sgpt' in df.columns:
        df['cholestatic_score'] = df['alkphos'] / (df['sgpt'] + 1)
        features_created.append('cholestatic_score')
    
    # Overall liver health score (inverse of damage)
    if 'liver_damage_score' in df.columns:
        max_damage = df['liver_damage_score'].max()
        df['liver_health_score'] = max_damage - df['liver_damage_score']
        features_created.append('liver_health_score')
    
    return df, features_created


def create_statistical_features(df):
    """Create statistical features"""
    features_created = []
    
    # Z-scores for key features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    key_features = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']
    available_features = [f for f in key_features if f in df.columns]
    
    if available_features:
        z_scores = scaler.fit_transform(df[available_features])
        for i, feat in enumerate(available_features):
            df[f'{feat}_zscore'] = z_scores[:, i]
            features_created.append(f'{feat}_zscore')
    
    # Robust scaling (median-based)
    for feat in available_features:
        median = df[feat].median()
        mad = (df[feat] - median).abs().median()
        if mad > 0:
            df[f'{feat}_robust'] = (df[feat] - median) / mad
            features_created.append(f'{feat}_robust')
    
    # Outlier indicators
    for feat in available_features:
        mean = df[feat].mean()
        std = df[feat].std()
        df[f'{feat}_outlier'] = (np.abs(df[feat] - mean) > 3 * std).astype(int)
        features_created.append(f'{feat}_outlier')
    
    # Total outliers
    outlier_cols = [f'{feat}_outlier' for feat in available_features]
    if outlier_cols:
        df['total_outliers'] = df[outlier_cols].sum(axis=1)
        features_created.append('total_outliers')
    
    return df, features_created


def create_polynomial_features(df):
    """Create polynomial features"""
    features_created = []
    
    key_features = ['sgpt', 'sgot', 'gammagt', 'alkphos', 'drinks']
    
    for feat in key_features:
        if feat in df.columns:
            df[f'{feat}_squared'] = df[feat] ** 2
            features_created.append(f'{feat}_squared')
    
    # Cubed features for highly skewed variables
    if 'gammagt' in df.columns:
        df['gammagt_cubed'] = df['gammagt'] ** 3
        features_created.append('gammagt_cubed')
    
    if 'drinks' in df.columns:
        df['drinks_cubed'] = df['drinks'] ** 3
        features_created.append('drinks_cubed')
    
    return df, features_created


def create_binned_features(df):
    """Create binned/categorical versions of continuous features"""
    features_created = []
    
    # Enzyme bins
    if 'sgpt' in df.columns:
        df['sgpt_bin'] = pd.cut(df['sgpt'], bins=[0, 28, 56, 84, float('inf')],
                               labels=['Low', 'Normal', 'Elevated', 'High'])
        df['sgpt_bin_encoded'] = pd.cut(df['sgpt'], bins=[0, 28, 56, 84, float('inf')],
                                        labels=[0, 1, 2, 3])
        df['sgpt_bin_encoded'] = df['sgpt_bin_encoded'].astype(int)
        features_created.extend(['sgpt_bin', 'sgpt_bin_encoded'])
    
    if 'sgot' in df.columns:
        df['sgot_bin'] = pd.cut(df['sgot'], bins=[0, 20, 40, 60, float('inf')],
                               labels=['Low', 'Normal', 'Elevated', 'High'])
        df['sgot_bin_encoded'] = pd.cut(df['sgot'], bins=[0, 20, 40, 60, float('inf')],
                                        labels=[0, 1, 2, 3])
        df['sgot_bin_encoded'] = df['sgot_bin_encoded'].astype(int)
        features_created.extend(['sgot_bin', 'sgot_bin_encoded'])
    
    if 'gammagt' in df.columns:
        df['gammagt_bin'] = pd.cut(df['gammagt'], bins=[0, 24, 48, 96, float('inf')],
                                   labels=['Low', 'Normal', 'Elevated', 'High'])
        df['gammagt_bin_encoded'] = pd.cut(df['gammagt'], bins=[0, 24, 48, 96, float('inf')],
                                           labels=[0, 1, 2, 3])
        df['gammagt_bin_encoded'] = df['gammagt_bin_encoded'].astype(int)
        features_created.extend(['gammagt_bin', 'gammagt_bin_encoded'])
    
    if 'drinks' in df.columns:
        df['drinks_bin'] = pd.cut(df['drinks'], bins=[-0.1, 0, 1, 2, 4, float('inf')],
                                  labels=['None', 'Light', 'Moderate', 'Heavy', 'Very Heavy'])
        df['drinks_bin_encoded'] = pd.cut(df['drinks'], bins=[-0.1, 0, 1, 2, 4, float('inf')],
                                          labels=[0, 1, 2, 3, 4])
        df['drinks_bin_encoded'] = df['drinks_bin_encoded'].astype(int)
        features_created.extend(['drinks_bin', 'drinks_bin_encoded'])
    
    return df, features_created


def create_interaction_features(df):
    """Create multi-feature interactions"""
    features_created = []
    
    # Three-way interactions
    if all(col in df.columns for col in ['mcv', 'alkphos', 'sgpt']):
        df['mcv_alkphos_sgpt'] = df['mcv'] * df['alkphos'] * df['sgpt'] / 1000
        features_created.append('mcv_alkphos_sgpt')
    
    if all(col in df.columns for col in ['sgpt', 'sgot', 'gammagt']):
        df['sgpt_sgot_gammagt'] = (df['sgpt'] * df['sgot'] * df['gammagt']) ** (1/3)
        features_created.append('sgpt_sgot_gammagt')
    
    if all(col in df.columns for col in ['drinks', 'mcv', 'gammagt']):
        df['drinks_mcv_gammagt'] = df['drinks'] * df['mcv'] * df['gammagt'] / 100
        features_created.append('drinks_mcv_gammagt')
    
    # Ratio interactions
    if all(col in df.columns for col in ['ast_alt_ratio', 'drinks']):
        df['ast_alt_drinks'] = df['ast_alt_ratio'] * df['drinks']
        features_created.append('ast_alt_drinks')
    
    if all(col in df.columns for col in ['total_enzyme_score', 'drinks']):
        df['enzyme_score_drinks'] = df['total_enzyme_score'] * df['drinks']
        features_created.append('enzyme_score_drinks')
    
    return df, features_created


def create_aggregate_features(df):
    """Create aggregate/summary features"""
    features_created = []
    
    # Count of abnormal values
    abnormal_indicators = [col for col in df.columns if 'elevated' in col or 'outlier' in col]
    if abnormal_indicators:
        df['abnormality_count'] = df[abnormal_indicators].sum(axis=1)
        df['abnormality_ratio'] = df['abnormality_count'] / len(abnormal_indicators)
        features_created.extend(['abnormality_count', 'abnormality_ratio'])
    
    # Enzyme statistics
    enzyme_cols = ['sgpt', 'sgot', 'gammagt', 'alkphos']
    available_enzymes = [col for col in enzyme_cols if col in df.columns]
    if len(available_enzymes) > 1:
        df['enzyme_mean'] = df[available_enzymes].mean(axis=1)
        df['enzyme_std'] = df[available_enzymes].std(axis=1)
        df['enzyme_cv'] = df['enzyme_std'] / (df['enzyme_mean'] + 1)  # Coefficient of variation
        df['enzyme_range'] = df[available_enzymes].max(axis=1) - df[available_enzymes].min(axis=1)
        df['enzyme_max'] = df[available_enzymes].max(axis=1)
        df['enzyme_min'] = df[available_enzymes].min(axis=1)
        features_created.extend(['enzyme_mean', 'enzyme_std', 'enzyme_cv', 
                                'enzyme_range', 'enzyme_max', 'enzyme_min'])
    
    return df, features_created


def main():
    """Main execution function"""
    print("\nLoading cleaned dataset...")
    
    # Load cleaned data
    input_path = DATA_DIR / 'liver_cleaned.csv'
    if not input_path.exists():
        print(f"[ERROR] File not found: {input_path}")
        print("Please run data_cleaning_liver.py first")
        return None
    
    df = pd.read_csv(input_path)
    initial_features = len(df.columns)
    print(f"[OK] Loaded: {len(df)} rows, {initial_features} columns")
    
    # Apply feature engineering
    all_features_created = []
    
    print("\n" + "="*80)
    print("CREATING ENGINEERED FEATURES")
    print("="*80)
    
    print("\n1. Enzyme Features...")
    df, features = create_enzyme_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n2. Alkaline Phosphatase Features...")
    df, features = create_alkphos_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n3. MCV Features...")
    df, features = create_mcv_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n4. Alcohol Features...")
    df, features = create_alcohol_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n5. Clinical Risk Scores...")
    df, features = create_clinical_risk_scores(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n6. Statistical Features...")
    df, features = create_statistical_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n7. Polynomial Features...")
    df, features = create_polynomial_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n8. Binned Features...")
    df, features = create_binned_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n9. Interaction Features...")
    df, features = create_interaction_features(df)
    all_features_created.extend(features)
    print(f"   Created {len(features)} features")
    
    print("\n10. Aggregate Features...")
    df, features = create_aggregate_features(df)
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
    output_path = OUTPUT_DIR / 'liver_engineered.csv'
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

# Made with Bob
