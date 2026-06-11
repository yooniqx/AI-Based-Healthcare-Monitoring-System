"""
Feature Engineering Script for BUPA Liver Disorders Dataset
This script creates advanced features based on domain knowledge and statistical analysis.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import stats

print("="*80)
print("BUPA LIVER DISORDERS - FEATURE ENGINEERING")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load cleaned data
data_path = '../ml_ready/liver_cleaned.csv'
print(f"\nLoading cleaned data from: {data_path}")
df = pd.read_csv(data_path)
print(f"[OK] Loaded {len(df)} records with {len(df.columns)} features")

# Create output directories
os.makedirs('../ml_ready', exist_ok=True)
os.makedirs('../reports/liver', exist_ok=True)

# Store original features for comparison
original_features = df.columns.tolist()
print(f"\nOriginal features: {len(original_features)}")

# ============================================================================
# 1. ENZYME-BASED FEATURES
# ============================================================================
print("\n" + "="*80)
print("1. CREATING ENZYME-BASED FEATURES")
print("="*80)

# Liver enzyme ratios (clinical significance)
df['sgpt_sgot_ratio'] = df['sgpt'] / (df['sgot'] + 1e-6)
df['sgot_sgpt_ratio'] = df['sgot'] / (df['sgpt'] + 1e-6)
df['gammagt_sgpt_ratio'] = df['gammagt'] / (df['sgpt'] + 1e-6)
df['gammagt_sgot_ratio'] = df['gammagt'] / (df['sgot'] + 1e-6)

# Enzyme combinations
df['sgpt_sgot_sum'] = df['sgpt'] + df['sgot']
df['sgpt_sgot_product'] = df['sgpt'] * df['sgot']
df['all_enzymes_sum'] = df['sgpt'] + df['sgot'] + df['gammagt']
df['all_enzymes_mean'] = (df['sgpt'] + df['sgot'] + df['gammagt']) / 3

# Enzyme dominance indicators
df['sgpt_dominant'] = (df['sgpt'] > df['sgot']).astype(int)
df['sgot_dominant'] = (df['sgot'] > df['sgpt']).astype(int)
df['gammagt_dominant'] = ((df['gammagt'] > df['sgpt']) & (df['gammagt'] > df['sgot'])).astype(int)

# Enzyme elevation levels (based on typical reference ranges)
# SGPT normal: 7-56 U/L, SGOT normal: 10-40 U/L, GGT normal: 9-48 U/L
df['sgpt_elevated'] = (df['sgpt'] > 56).astype(int)
df['sgot_elevated'] = (df['sgot'] > 40).astype(int)
df['gammagt_elevated'] = (df['gammagt'] > 48).astype(int)
df['elevated_enzyme_count'] = df['sgpt_elevated'] + df['sgot_elevated'] + df['gammagt_elevated']

print(f"[OK] Created {14} enzyme-based features")

# ============================================================================
# 2. ALKALINE PHOSPHATASE FEATURES
# ============================================================================
print("\n" + "="*80)
print("2. CREATING ALKALINE PHOSPHATASE FEATURES")
print("="*80)

# Alkphos ratios with enzymes
df['alkphos_sgpt_ratio'] = df['alkphos'] / (df['sgpt'] + 1e-6)
df['alkphos_sgot_ratio'] = df['alkphos'] / (df['sgot'] + 1e-6)
df['alkphos_gammagt_ratio'] = df['alkphos'] / (df['gammagt'] + 1e-6)

# Alkphos elevation (normal range: 30-120 U/L)
df['alkphos_elevated'] = (df['alkphos'] > 120).astype(int)
df['alkphos_low'] = (df['alkphos'] < 30).astype(int)

# Alkphos interaction with MCV
df['alkphos_mcv_ratio'] = df['alkphos'] / (df['mcv'] + 1e-6)
df['alkphos_mcv_product'] = df['alkphos'] * df['mcv']

print(f"[OK] Created {7} alkaline phosphatase features")

# ============================================================================
# 3. MCV (MEAN CORPUSCULAR VOLUME) FEATURES
# ============================================================================
print("\n" + "="*80)
print("3. CREATING MCV FEATURES")
print("="*80)

# MCV categories (normal: 80-100 fL)
df['mcv_low'] = (df['mcv'] < 80).astype(int)  # Microcytic
df['mcv_normal'] = ((df['mcv'] >= 80) & (df['mcv'] <= 100)).astype(int)
df['mcv_high'] = (df['mcv'] > 100).astype(int)  # Macrocytic

# MCV deviation from normal
df['mcv_deviation'] = np.abs(df['mcv'] - 90)  # 90 is midpoint of normal range
df['mcv_normalized'] = (df['mcv'] - 90) / 10  # Standardized deviation

# MCV interaction with alcohol
df['mcv_drinks_interaction'] = df['mcv'] * df['drinks']
df['mcv_alcohol_category_encoded'] = df['mcv'] * df['alcohol_category'].replace({
    'None': 0, 'Light': 1, 'Moderate': 2, 'Heavy': 3, 'Very Heavy': 4
}).fillna(0)

print(f"[OK] Created {7} MCV features")

# ============================================================================
# 4. ALCOHOL-RELATED FEATURES
# ============================================================================
print("\n" + "="*80)
print("4. CREATING ALCOHOL-RELATED FEATURES")
print("="*80)

# Alcohol consumption levels
df['drinks_squared'] = df['drinks'] ** 2
df['drinks_log'] = np.log1p(df['drinks'])  # log(1 + drinks)
df['drinks_sqrt'] = np.sqrt(df['drinks'])

# Alcohol categories encoded
alcohol_mapping = {'None': 0, 'Light': 1, 'Moderate': 2, 'Heavy': 3, 'Very Heavy': 4}
df['alcohol_level'] = df['alcohol_category'].replace(alcohol_mapping).fillna(0)

# Alcohol interaction with enzymes
df['drinks_sgpt_interaction'] = df['drinks'] * df['sgpt']
df['drinks_sgot_interaction'] = df['drinks'] * df['sgot']
df['drinks_gammagt_interaction'] = df['drinks'] * df['gammagt']
df['drinks_ast_alt_ratio_interaction'] = df['drinks'] * df['ast_alt_ratio']

# Heavy drinker indicator (>2 drinks/day)
df['heavy_drinker'] = (df['drinks'] > 2).astype(int)

# Alcohol-enzyme risk score
df['alcohol_enzyme_risk'] = (
    df['drinks'] * 0.3 + 
    df['gammagt'] / 100 * 0.4 + 
    df['ast_alt_ratio'] * 0.3
)

print(f"[OK] Created {10} alcohol-related features")

# ============================================================================
# 5. CLINICAL RISK SCORES
# ============================================================================
print("\n" + "="*80)
print("5. CREATING CLINICAL RISK SCORES")
print("="*80)

# Liver damage risk score (weighted combination)
df['liver_damage_score'] = (
    (df['sgpt'] / 56) * 0.25 +  # Normalized by upper normal limit
    (df['sgot'] / 40) * 0.25 +
    (df['gammagt'] / 48) * 0.25 +
    (df['ast_alt_ratio'] > 2).astype(int) * 0.25
)

# Alcoholic liver disease risk
df['alcoholic_liver_risk'] = (
    (df['ast_alt_ratio'] > 2).astype(int) * 0.3 +
    (df['gammagt'] > 48).astype(int) * 0.3 +
    (df['drinks'] > 2).astype(int) * 0.2 +
    (df['mcv'] > 100).astype(int) * 0.2
)

# Hepatocellular injury score (SGPT/SGOT dominant)
df['hepatocellular_score'] = (df['sgpt'] + df['sgot']) / (df['alkphos'] + 1e-6)

# Cholestatic injury score (Alkphos dominant)
df['cholestatic_score'] = df['alkphos'] / (df['sgpt'] + df['sgot'] + 1e-6)

# Overall liver health score (inverse of damage)
df['liver_health_score'] = 1 / (1 + df['liver_damage_score'])

print(f"[OK] Created {5} clinical risk scores")

# ============================================================================
# 6. STATISTICAL FEATURES
# ============================================================================
print("\n" + "="*80)
print("6. CREATING STATISTICAL FEATURES")
print("="*80)

# Z-scores for key features
scaler = StandardScaler()
key_features = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks']
z_score_cols = [f'{col}_zscore' for col in key_features]
df[z_score_cols] = scaler.fit_transform(df[key_features])

# Robust scaling (less sensitive to outliers)
robust_scaler = RobustScaler()
robust_cols = [f'{col}_robust' for col in key_features]
df[robust_cols] = robust_scaler.fit_transform(df[key_features])

# Outlier indicators (beyond 3 standard deviations)
for col in key_features:
    df[f'{col}_outlier'] = (np.abs(df[f'{col}_zscore']) > 3).astype(int)

df['total_outliers'] = df[[f'{col}_outlier' for col in key_features]].sum(axis=1)

print(f"[OK] Created {19} statistical features")

# ============================================================================
# 7. POLYNOMIAL FEATURES (SELECTED)
# ============================================================================
print("\n" + "="*80)
print("7. CREATING POLYNOMIAL FEATURES")
print("="*80)

# Squared terms for key features
df['sgpt_squared'] = df['sgpt'] ** 2
df['sgot_squared'] = df['sgot'] ** 2
df['gammagt_squared'] = df['gammagt'] ** 2
df['alkphos_squared'] = df['alkphos'] ** 2

# Cubic terms for highly skewed features
df['gammagt_cubed'] = df['gammagt'] ** 3
df['drinks_cubed'] = df['drinks'] ** 3

print(f"[OK] Created {6} polynomial features")

# ============================================================================
# 8. BINNED FEATURES
# ============================================================================
print("\n" + "="*80)
print("8. CREATING BINNED FEATURES")
print("="*80)

# Bin continuous features into categories
df['sgpt_bin'] = pd.cut(df['sgpt'], bins=[0, 28, 56, 100, np.inf], 
                         labels=['Low', 'Normal', 'Elevated', 'High'])
df['sgot_bin'] = pd.cut(df['sgot'], bins=[0, 20, 40, 80, np.inf], 
                         labels=['Low', 'Normal', 'Elevated', 'High'])
df['gammagt_bin'] = pd.cut(df['gammagt'], bins=[0, 24, 48, 100, np.inf], 
                            labels=['Low', 'Normal', 'Elevated', 'High'])
df['drinks_bin'] = pd.cut(df['drinks'], bins=[-0.1, 0, 1, 2, 5, np.inf], 
                           labels=['None', 'Light', 'Moderate', 'Heavy', 'Very Heavy'])

# Encode bins as numeric
for col in ['sgpt_bin', 'sgot_bin', 'gammagt_bin', 'drinks_bin']:
    df[f'{col}_encoded'] = df[col].cat.codes

print(f"[OK] Created {8} binned features")

# ============================================================================
# 9. INTERACTION FEATURES
# ============================================================================
print("\n" + "="*80)
print("9. CREATING INTERACTION FEATURES")
print("="*80)

# Key interactions based on domain knowledge
df['mcv_alkphos_sgpt'] = df['mcv'] * df['alkphos'] * df['sgpt']
df['sgpt_sgot_gammagt'] = df['sgpt'] * df['sgot'] * df['gammagt']
df['drinks_mcv_gammagt'] = df['drinks'] * df['mcv'] * df['gammagt']

# Ratio-based interactions
df['ast_alt_drinks'] = df['ast_alt_ratio'] * df['drinks']
df['enzyme_score_drinks'] = df['total_enzyme_score'] * df['drinks']

print(f"[OK] Created {5} interaction features")

# ============================================================================
# 10. AGGREGATE FEATURES
# ============================================================================
print("\n" + "="*80)
print("10. CREATING AGGREGATE FEATURES")
print("="*80)

# Overall abnormality score
abnormal_features = [
    'sgpt_elevated', 'sgot_elevated', 'gammagt_elevated', 
    'alkphos_elevated', 'mcv_high', 'heavy_drinker'
]
df['abnormality_count'] = df[abnormal_features].sum(axis=1)
df['abnormality_ratio'] = df['abnormality_count'] / len(abnormal_features)

# Enzyme variability (coefficient of variation)
enzyme_cols = ['sgpt', 'sgot', 'gammagt']
df['enzyme_mean'] = df[enzyme_cols].mean(axis=1)
df['enzyme_std'] = df[enzyme_cols].std(axis=1)
df['enzyme_cv'] = df['enzyme_std'] / (df['enzyme_mean'] + 1e-6)

# Range features
df['enzyme_range'] = df[enzyme_cols].max(axis=1) - df[enzyme_cols].min(axis=1)
df['enzyme_max'] = df[enzyme_cols].max(axis=1)
df['enzyme_min'] = df[enzyme_cols].min(axis=1)

print(f"[OK] Created {8} aggregate features")

# ============================================================================
# SUMMARY AND SAVE
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING SUMMARY")
print("="*80)

new_features = [col for col in df.columns if col not in original_features]
print(f"\nOriginal features: {len(original_features)}")
print(f"New features created: {len(new_features)}")
print(f"Total features: {len(df.columns)}")

# Feature categories
feature_categories = {
    'Enzyme-based': 14,
    'Alkaline Phosphatase': 7,
    'MCV': 7,
    'Alcohol-related': 10,
    'Clinical Risk Scores': 5,
    'Statistical': 19,
    'Polynomial': 6,
    'Binned': 8,
    'Interaction': 5,
    'Aggregate': 8
}

print("\nFeature categories:")
for category, count in feature_categories.items():
    print(f"  {category}: {count} features")

# Save engineered dataset
output_path = '../ml_ready/liver_engineered.csv'
df.to_csv(output_path, index=False)
print(f"\n[OK] Saved engineered dataset to: {output_path}")

# Create feature engineering report
report_data = []
for feature in new_features:
    report_data.append({
        'Feature': feature,
        'Type': df[feature].dtype,
        'Non_Null_Count': df[feature].notna().sum(),
        'Null_Count': df[feature].isna().sum(),
        'Mean': df[feature].mean() if df[feature].dtype in ['int64', 'float64'] else 'N/A',
        'Std': df[feature].std() if df[feature].dtype in ['int64', 'float64'] else 'N/A',
        'Min': df[feature].min() if df[feature].dtype in ['int64', 'float64'] else 'N/A',
        'Max': df[feature].max() if df[feature].dtype in ['int64', 'float64'] else 'N/A'
    })

report_df = pd.DataFrame(report_data)
report_path = '../reports/liver/feature_engineering_report.csv'
report_df.to_csv(report_path, index=False)
print(f"[OK] Saved feature engineering report to: {report_path}")

# Save feature list
feature_list_path = '../reports/liver/engineered_features_list.csv'
pd.DataFrame({
    'Original_Features': original_features + [''] * (len(new_features) - len(original_features)),
    'New_Features': new_features
}).to_csv(feature_list_path, index=False)
print(f"[OK] Saved feature list to: {feature_list_path}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE")
print("="*80)

