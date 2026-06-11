"""
Feature Engineering Script for Thyroid Disease Datasets
This script creates advanced features for ann_thyroid, hypothyroid, and new_thyroid datasets.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler

print("="*80)
print("THYROID DISEASE DATASETS - FEATURE ENGINEERING")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create output directories
os.makedirs('../ml_ready', exist_ok=True)
os.makedirs('../reports/thyroid', exist_ok=True)

# ============================================================================
# FUNCTION DEFINITIONS
# ============================================================================

def create_thyroid_hormone_features(df):
    """Create features based on thyroid hormone relationships"""
    features_created = []
    
    # TSH-based features (if TSH exists)
    if 'TSH' in df.columns:
        df['TSH_log'] = np.log1p(df['TSH'])
        df['TSH_squared'] = df['TSH'] ** 2
        df['TSH_elevated'] = (df['TSH'] > 4.5).astype(int)  # Normal: 0.4-4.5
        df['TSH_suppressed'] = (df['TSH'] < 0.4).astype(int)
        features_created.extend(['TSH_log', 'TSH_squared', 'TSH_elevated', 'TSH_suppressed'])
    
    # T3-based features
    if 'T3' in df.columns:
        df['T3_log'] = np.log1p(df['T3'])
        df['T3_squared'] = df['T3'] ** 2
        df['T3_elevated'] = (df['T3'] > 0.2).astype(int)  # Approximate normal upper limit
        df['T3_low'] = (df['T3'] < 0.08).astype(int)
        features_created.extend(['T3_log', 'T3_squared', 'T3_elevated', 'T3_low'])
    
    # T4/TT4-based features
    t4_col = 'TT4' if 'TT4' in df.columns else 'T4' if 'T4' in df.columns else None
    if t4_col:
        df['T4_log'] = np.log1p(df[t4_col])
        df['T4_squared'] = df[t4_col] ** 2
        df['T4_elevated'] = (df[t4_col] > 0.15).astype(int)
        df['T4_low'] = (df[t4_col] < 0.05).astype(int)
        features_created.extend(['T4_log', 'T4_squared', 'T4_elevated', 'T4_low'])
    
    # T4U-based features
    if 'T4U' in df.columns:
        df['T4U_log'] = np.log1p(df['T4U'])
        df['T4U_squared'] = df['T4U'] ** 2
        features_created.extend(['T4U_log', 'T4U_squared'])
    
    # FTI (Free Thyroxine Index) features
    if 'FTI' in df.columns:
        df['FTI_log'] = np.log1p(df['FTI'])
        df['FTI_squared'] = df['FTI'] ** 2
        df['FTI_elevated'] = (df['FTI'] > 0.15).astype(int)
        df['FTI_low'] = (df['FTI'] < 0.05).astype(int)
        features_created.extend(['FTI_log', 'FTI_squared', 'FTI_elevated', 'FTI_low'])
    
    return df, features_created

def create_hormone_ratios(df):
    """Create ratio features between hormones"""
    features_created = []
    
    # TSH to T3 ratio
    if 'TSH' in df.columns and 'T3' in df.columns:
        df['TSH_T3_ratio'] = df['TSH'] / (df['T3'] + 1e-6)
        features_created.append('TSH_T3_ratio')
    
    # TSH to T4 ratio
    t4_col = 'TT4' if 'TT4' in df.columns else 'T4' if 'T4' in df.columns else None
    if 'TSH' in df.columns and t4_col:
        df['TSH_T4_ratio'] = df['TSH'] / (df[t4_col] + 1e-6)
        features_created.append('TSH_T4_ratio')
    
    # T3 to T4 ratio
    if 'T3' in df.columns and t4_col:
        df['T3_T4_ratio'] = df['T3'] / (df[t4_col] + 1e-6)
        df['T4_T3_ratio'] = df[t4_col] / (df['T3'] + 1e-6)
        features_created.extend(['T3_T4_ratio', 'T4_T3_ratio'])
    
    # FTI to T4U ratio
    if 'FTI' in df.columns and 'T4U' in df.columns:
        df['FTI_T4U_ratio'] = df['FTI'] / (df['T4U'] + 1e-6)
        features_created.append('FTI_T4U_ratio')
    
    # T4 to T4U ratio (should approximate FTI)
    if t4_col and 'T4U' in df.columns:
        df['T4_T4U_ratio'] = df[t4_col] / (df['T4U'] + 1e-6)
        features_created.append('T4_T4U_ratio')
    
    return df, features_created

def create_clinical_scores(df):
    """Create clinical risk and diagnostic scores"""
    features_created = []
    
    # Hypothyroid risk score
    score_components = []
    if 'TSH' in df.columns:
        score_components.append((df['TSH'] > 4.5).astype(int) * 0.4)
    
    t4_col = 'TT4' if 'TT4' in df.columns else 'T4' if 'T4' in df.columns else None
    if t4_col:
        score_components.append((df[t4_col] < 0.05).astype(int) * 0.3)
    
    if 'T3' in df.columns:
        score_components.append((df['T3'] < 0.08).astype(int) * 0.3)
    
    if score_components:
        df['hypothyroid_risk_score'] = sum(score_components)
        features_created.append('hypothyroid_risk_score')
    
    # Hyperthyroid risk score
    score_components = []
    if 'TSH' in df.columns:
        score_components.append((df['TSH'] < 0.4).astype(int) * 0.4)
    
    if t4_col:
        score_components.append((df[t4_col] > 0.15).astype(int) * 0.3)
    
    if 'T3' in df.columns:
        score_components.append((df['T3'] > 0.2).astype(int) * 0.3)
    
    if score_components:
        df['hyperthyroid_risk_score'] = sum(score_components)
        features_created.append('hyperthyroid_risk_score')
    
    # Thyroid function abnormality count
    abnormal_count = 0
    if 'TSH' in df.columns:
        abnormal_count += ((df['TSH'] < 0.4) | (df['TSH'] > 4.5)).astype(int)
    if t4_col:
        abnormal_count += ((df[t4_col] < 0.05) | (df[t4_col] > 0.15)).astype(int)
    if 'T3' in df.columns:
        abnormal_count += ((df['T3'] < 0.08) | (df['T3'] > 0.2)).astype(int)
    
    if isinstance(abnormal_count, pd.Series):
        df['abnormal_hormone_count'] = abnormal_count
        features_created.append('abnormal_hormone_count')
    
    return df, features_created

def create_demographic_features(df):
    """Create features from demographic information"""
    features_created = []
    
    # Age-based features
    if 'age' in df.columns:
        df['age_squared'] = df['age'] ** 2
        df['age_log'] = np.log1p(df['age'])
        df['age_group'] = pd.cut(df['age'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                 labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly'])
        df['age_group_encoded'] = df['age_group'].cat.codes
        df['elderly'] = (df['age'] > 0.7).astype(int)
        df['young'] = (df['age'] < 0.3).astype(int)
        features_created.extend(['age_squared', 'age_log', 'age_group', 
                                'age_group_encoded', 'elderly', 'young'])
    
    # Sex-based interactions
    if 'sex' in df.columns and 'age' in df.columns:
        df['sex_age_interaction'] = df['sex'] * df['age']
        features_created.append('sex_age_interaction')
    
    return df, features_created

def create_medication_features(df):
    """Create features from medication and treatment history"""
    features_created = []
    
    # Treatment count
    treatment_cols = ['on_thyroxine', 'on_antithyroid_medication', 'I131_treatment', 
                     'thyroid_surgery']
    existing_treatment_cols = [col for col in treatment_cols if col in df.columns]
    
    if existing_treatment_cols:
        df['treatment_count'] = df[existing_treatment_cols].sum(axis=1)
        df['on_any_treatment'] = (df['treatment_count'] > 0).astype(int)
        features_created.extend(['treatment_count', 'on_any_treatment'])
    
    # Query count (diagnostic uncertainty)
    query_cols = ['query_on_thyroxine', 'query_hypothyroid', 'query_hyperthyroid']
    existing_query_cols = [col for col in query_cols if col in df.columns]
    
    if existing_query_cols:
        df['query_count'] = df[existing_query_cols].sum(axis=1)
        df['diagnostic_uncertainty'] = (df['query_count'] > 0).astype(int)
        features_created.extend(['query_count', 'diagnostic_uncertainty'])
    
    # Condition count
    condition_cols = ['sick', 'pregnant', 'lithium', 'goitre', 'tumor', 
                     'hypopituitary', 'psych']
    existing_condition_cols = [col for col in condition_cols if col in df.columns]
    
    if existing_condition_cols:
        df['condition_count'] = df[existing_condition_cols].sum(axis=1)
        df['has_comorbidity'] = (df['condition_count'] > 0).astype(int)
        features_created.extend(['condition_count', 'has_comorbidity'])
    
    # Complex case indicator
    if 'treatment_count' in df.columns and 'condition_count' in df.columns:
        df['complex_case'] = ((df['treatment_count'] > 1) | 
                              (df['condition_count'] > 1)).astype(int)
        features_created.append('complex_case')
    
    return df, features_created

def create_statistical_features(df, numerical_cols):
    """Create statistical features"""
    features_created = []
    
    # Z-scores
    scaler = StandardScaler()
    z_score_cols = [f'{col}_zscore' for col in numerical_cols]
    df[z_score_cols] = scaler.fit_transform(df[numerical_cols])
    features_created.extend(z_score_cols)
    
    # Robust scaling
    robust_scaler = RobustScaler()
    robust_cols = [f'{col}_robust' for col in numerical_cols]
    df[robust_cols] = robust_scaler.fit_transform(df[numerical_cols])
    features_created.extend(robust_cols)
    
    # Outlier detection
    for col in numerical_cols:
        df[f'{col}_outlier'] = (np.abs(df[f'{col}_zscore']) > 3).astype(int)
        features_created.append(f'{col}_outlier')
    
    return df, features_created

def create_new_thyroid_specific_features(df):
    """Create features specific to new_thyroid dataset"""
    features_created = []
    
    if 'T3_resin_uptake' in df.columns:
        df['T3_resin_log'] = np.log1p(df['T3_resin_uptake'])
        df['T3_resin_squared'] = df['T3_resin_uptake'] ** 2
        features_created.extend(['T3_resin_log', 'T3_resin_squared'])
    
    if 'total_serum_thyroxin' in df.columns:
        df['serum_thyroxin_log'] = np.log1p(df['total_serum_thyroxin'])
        df['serum_thyroxin_squared'] = df['total_serum_thyroxin'] ** 2
        features_created.extend(['serum_thyroxin_log', 'serum_thyroxin_squared'])
    
    if 'total_serum_triiodothyronine' in df.columns:
        df['serum_t3_log'] = np.log1p(df['total_serum_triiodothyronine'])
        df['serum_t3_squared'] = df['total_serum_triiodothyronine'] ** 2
        features_created.extend(['serum_t3_log', 'serum_t3_squared'])
    
    if 'basal_TSH' in df.columns:
        df['basal_TSH_log'] = np.log1p(df['basal_TSH'])
        df['basal_TSH_squared'] = df['basal_TSH'] ** 2
        features_created.extend(['basal_TSH_log', 'basal_TSH_squared'])
    
    if 'max_TSH_difference' in df.columns:
        df['TSH_diff_log'] = np.log1p(np.abs(df['max_TSH_difference']))
        df['TSH_diff_squared'] = df['max_TSH_difference'] ** 2
        df['TSH_response_positive'] = (df['max_TSH_difference'] > 0).astype(int)
        features_created.extend(['TSH_diff_log', 'TSH_diff_squared', 'TSH_response_positive'])
    
    # Ratios
    if 'T3_resin_uptake' in df.columns and 'total_serum_thyroxin' in df.columns:
        df['resin_thyroxin_ratio'] = df['T3_resin_uptake'] / (df['total_serum_thyroxin'] + 1e-6)
        features_created.append('resin_thyroxin_ratio')
    
    if 'total_serum_thyroxin' in df.columns and 'total_serum_triiodothyronine' in df.columns:
        df['T4_T3_serum_ratio'] = df['total_serum_thyroxin'] / (df['total_serum_triiodothyronine'] + 1e-6)
        features_created.append('T4_T3_serum_ratio')
    
    return df, features_created

# ============================================================================
# INITIALIZE VARIABLES
# ============================================================================
# Initialize all dataframe variables to avoid type checker warnings
df_ann = pd.DataFrame()
df_hypo = pd.DataFrame()
df_new = pd.DataFrame()
original_features_ann: list = []
original_features_hypo: list = []
original_features_new: list = []

# ============================================================================
# PROCESS ANN_THYROID DATASET
# ============================================================================
print("\n" + "="*80)
print("PROCESSING ANN_THYROID DATASET")
print("="*80)

ann_path = '../ml_ready/ann_thyroid_cleaned.csv'
if os.path.exists(ann_path):
    print(f"\nLoading: {ann_path}")
    df_ann = pd.read_csv(ann_path)
    print(f"[OK] Loaded {len(df_ann)} records with {len(df_ann.columns)} features")
    
    original_features_ann = df_ann.columns.tolist()
    all_features_created = []
    
    # Apply feature engineering
    print("\nCreating thyroid hormone features...")
    df_ann, features = create_thyroid_hormone_features(df_ann)
    all_features_created.extend(features)
    
    print("Creating hormone ratios...")
    df_ann, features = create_hormone_ratios(df_ann)
    all_features_created.extend(features)
    
    print("Creating clinical scores...")
    df_ann, features = create_clinical_scores(df_ann)
    all_features_created.extend(features)
    
    print("Creating demographic features...")
    df_ann, features = create_demographic_features(df_ann)
    all_features_created.extend(features)
    
    print("Creating medication features...")
    df_ann, features = create_medication_features(df_ann)
    all_features_created.extend(features)
    
    # Statistical features for numerical columns
    numerical_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    numerical_cols = [col for col in numerical_cols if col in df_ann.columns]
    print("Creating statistical features...")
    df_ann, features = create_statistical_features(df_ann, numerical_cols)
    all_features_created.extend(features)
    
    # Save
    output_path = '../ml_ready/ann_thyroid_engineered.csv'
    df_ann.to_csv(output_path, index=False)
    print(f"\n[OK] Saved to: {output_path}")
    print(f"Original features: {len(original_features_ann)}")
    print(f"New features: {len(all_features_created)}")
    print(f"Total features: {len(df_ann.columns)}")
else:
    print(f"[SKIP] File not found: {ann_path}")

# ============================================================================
# PROCESS HYPOTHYROID DATASET
# ============================================================================
print("\n" + "="*80)
print("PROCESSING HYPOTHYROID DATASET")
print("="*80)

hypo_path = '../ml_ready/hypothyroid_cleaned.csv'
if os.path.exists(hypo_path):
    print(f"\nLoading: {hypo_path}")
    df_hypo = pd.read_csv(hypo_path)
    print(f"[OK] Loaded {len(df_hypo)} records with {len(df_hypo.columns)} features")
    
    original_features_hypo = df_hypo.columns.tolist()
    all_features_created = []
    
    # Apply feature engineering
    print("\nCreating thyroid hormone features...")
    df_hypo, features = create_thyroid_hormone_features(df_hypo)
    all_features_created.extend(features)
    
    print("Creating hormone ratios...")
    df_hypo, features = create_hormone_ratios(df_hypo)
    all_features_created.extend(features)
    
    print("Creating clinical scores...")
    df_hypo, features = create_clinical_scores(df_hypo)
    all_features_created.extend(features)
    
    print("Creating demographic features...")
    df_hypo, features = create_demographic_features(df_hypo)
    all_features_created.extend(features)
    
    print("Creating medication features...")
    df_hypo, features = create_medication_features(df_hypo)
    all_features_created.extend(features)
    
    # Statistical features
    numerical_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    numerical_cols = [col for col in numerical_cols if col in df_hypo.columns]
    print("Creating statistical features...")
    df_hypo, features = create_statistical_features(df_hypo, numerical_cols)
    all_features_created.extend(features)
    
    # Save
    output_path = '../ml_ready/hypothyroid_engineered.csv'
    df_hypo.to_csv(output_path, index=False)
    print(f"\n[OK] Saved to: {output_path}")
    print(f"Original features: {len(original_features_hypo)}")
    print(f"New features: {len(all_features_created)}")
    print(f"Total features: {len(df_hypo.columns)}")
else:
    print(f"[SKIP] File not found: {hypo_path}")

# ============================================================================
# PROCESS NEW_THYROID DATASET
# ============================================================================
print("\n" + "="*80)
print("PROCESSING NEW_THYROID DATASET")
print("="*80)

new_path = '../ml_ready/new_thyroid_cleaned.csv'
if os.path.exists(new_path):
    print(f"\nLoading: {new_path}")
    df_new = pd.read_csv(new_path)
    print(f"[OK] Loaded {len(df_new)} records with {len(df_new.columns)} features")
    
    original_features_new = df_new.columns.tolist()
    all_features_created = []
    
    # Apply feature engineering
    print("\nCreating new_thyroid specific features...")
    df_new, features = create_new_thyroid_specific_features(df_new)
    all_features_created.extend(features)
    
    # Statistical features
    numerical_cols = ['T3_resin_uptake', 'total_serum_thyroxin', 
                     'total_serum_triiodothyronine', 'basal_TSH', 'max_TSH_difference']
    numerical_cols = [col for col in numerical_cols if col in df_new.columns]
    print("Creating statistical features...")
    df_new, features = create_statistical_features(df_new, numerical_cols)
    all_features_created.extend(features)
    
    # Save
    output_path = '../ml_ready/new_thyroid_engineered.csv'
    df_new.to_csv(output_path, index=False)
    print(f"\n[OK] Saved to: {output_path}")
    print(f"Original features: {len(original_features_new)}")
    print(f"New features: {len(all_features_created)}")
    print(f"Total features: {len(df_new.columns)}")
else:
    print(f"[SKIP] File not found: {new_path}")

# ============================================================================
# GENERATE SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("GENERATING SUMMARY REPORT")
print("="*80)

summary_data = []

try:
    if os.path.exists(ann_path):
        summary_data.append({
            'Dataset': 'ann_thyroid',
            'Original_Features': len(original_features_ann),
            'Engineered_Features': len(df_ann.columns),
            'New_Features': len(df_ann.columns) - len(original_features_ann),
            'Records': len(df_ann)
        })
except NameError:
    pass

try:
    if os.path.exists(hypo_path):
        summary_data.append({
            'Dataset': 'hypothyroid',
            'Original_Features': len(original_features_hypo),
            'Engineered_Features': len(df_hypo.columns),
            'New_Features': len(df_hypo.columns) - len(original_features_hypo),
            'Records': len(df_hypo)
        })
except NameError:
    pass

try:
    if os.path.exists(new_path):
        summary_data.append({
            'Dataset': 'new_thyroid',
            'Original_Features': len(original_features_new),
            'Engineered_Features': len(df_new.columns),
            'New_Features': len(df_new.columns) - len(original_features_new),
            'Records': len(df_new)
        })
except NameError:
    pass

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    summary_path = '../reports/thyroid/feature_engineering_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[OK] Saved summary to: {summary_path}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETE FOR ALL THYROID DATASETS")
print("="*80)

