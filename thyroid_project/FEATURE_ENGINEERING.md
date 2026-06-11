# Thyroid Disease Datasets - Feature Engineering Documentation

## Overview
This document describes the comprehensive feature engineering performed on three thyroid disease datasets: ann_thyroid, hypothyroid, and new_thyroid. Each dataset received tailored feature engineering based on its specific characteristics and clinical context.

## Datasets Summary

| Dataset | Original Features | Engineered Features | New Features | Records |
|---------|------------------|---------------------|--------------|---------|
| ann_thyroid | 23 | 82 | 59 | 7,129 |
| hypothyroid | 24 | 83 | 59 | 3,002 |
| new_thyroid | 6 | 34 | 28 | 215 |

## Feature Engineering Categories

### 1. Thyroid Hormone Features
Transformations and indicators for key thyroid hormones:

#### TSH (Thyroid Stimulating Hormone)
- **Transformations**: `TSH_log`, `TSH_squared`
- **Clinical Indicators**: 
  - `TSH_elevated` (>4.5 mIU/L): Suggests hypothyroidism
  - `TSH_suppressed` (<0.4 mIU/L): Suggests hyperthyroidism

#### T3 (Triiodothyronine)
- **Transformations**: `T3_log`, `T3_squared`
- **Clinical Indicators**:
  - `T3_elevated` (>0.2): Hyperthyroidism indicator
  - `T3_low` (<0.08): Hypothyroidism indicator

#### T4/TT4 (Thyroxine)
- **Transformations**: `T4_log`, `T4_squared`
- **Clinical Indicators**:
  - `T4_elevated` (>0.15): Hyperthyroidism indicator
  - `T4_low` (<0.05): Hypothyroidism indicator

#### T4U (T4 Uptake)
- **Transformations**: `T4U_log`, `T4U_squared`

#### FTI (Free Thyroxine Index)
- **Transformations**: `FTI_log`, `FTI_squared`
- **Clinical Indicators**:
  - `FTI_elevated` (>0.15)
  - `FTI_low` (<0.05)

**Clinical Significance**: These features capture non-linear relationships and clinical thresholds critical for thyroid disease diagnosis.

### 2. Hormone Ratio Features
Clinically meaningful ratios between hormones:

- **`TSH_T3_ratio`**: TSH to T3 ratio
- **`TSH_T4_ratio`**: TSH to T4 ratio
- **`T3_T4_ratio`**: T3 to T4 ratio (conversion efficiency)
- **`T4_T3_ratio`**: Inverse ratio
- **`FTI_T4U_ratio`**: Free thyroxine index relationship
- **`T4_T4U_ratio`**: Should approximate FTI

**Clinical Significance**: Ratios help identify thyroid hormone conversion issues and peripheral resistance.

### 3. Clinical Risk Scores
Composite scores for diagnostic support:

#### Hypothyroid Risk Score
Weighted combination of:
- TSH elevation (40% weight)
- Low T4 (30% weight)
- Low T3 (30% weight)

#### Hyperthyroid Risk Score
Weighted combination of:
- TSH suppression (40% weight)
- High T4 (30% weight)
- High T3 (30% weight)

#### Abnormal Hormone Count
Count of hormones outside normal ranges

**Clinical Significance**: These scores provide interpretable risk assessments for thyroid dysfunction.

### 4. Demographic Features
Age and sex-based features:

- **Age Transformations**: `age_squared`, `age_log`
- **Age Categories**: `age_group` (Very Young/Young/Middle/Senior/Elderly)
- **Age Indicators**: `elderly` (>0.7), `young` (<0.3)
- **Interactions**: `sex_age_interaction`

**Clinical Significance**: Thyroid disease prevalence varies significantly by age and sex.

### 5. Medication and Treatment Features
Treatment history indicators:

#### Treatment Count
Sum of active treatments:
- Thyroxine medication
- Antithyroid medication
- I131 treatment
- Thyroid surgery

#### Query Count
Diagnostic uncertainty indicators:
- Query on thyroxine
- Query hypothyroid
- Query hyperthyroid

#### Condition Count
Comorbidities:
- Sick, Pregnant, Lithium use
- Goitre, Tumor, Hypopituitary
- Psychiatric conditions

#### Derived Features
- **`on_any_treatment`**: Binary indicator
- **`diagnostic_uncertainty`**: Has any queries
- **`has_comorbidity`**: Has any conditions
- **`complex_case`**: Multiple treatments or conditions

**Clinical Significance**: Treatment history affects hormone levels and diagnostic interpretation.

### 6. Statistical Features
Standardization and outlier detection:

- **Z-scores**: Standardized values for all numerical features
- **Robust Scaling**: Median-centered, MAD-scaled values
- **Outlier Detection**: Binary indicators for values >3 SD from mean

**Purpose**: Normalize features for machine learning and identify anomalous measurements.

### 7. New Thyroid Specific Features
Features unique to the new_thyroid dataset:

#### T3 Resin Uptake
- `T3_resin_log`, `T3_resin_squared`

#### Total Serum Thyroxin
- `serum_thyroxin_log`, `serum_thyroxin_squared`

#### Total Serum Triiodothyronine
- `serum_t3_log`, `serum_t3_squared`

#### Basal TSH
- `basal_TSH_log`, `basal_TSH_squared`

#### TSH Response
- `TSH_diff_log`, `TSH_diff_squared`
- `TSH_response_positive`: Positive TSH response indicator

#### Ratios
- `resin_thyroxin_ratio`
- `T4_T3_serum_ratio`

**Clinical Significance**: These features are specific to the TRH stimulation test protocol.

## Clinical Reference Ranges

### Thyroid Hormones (Normalized Values)
- **TSH**: Normal 0.4-4.5 mIU/L
- **T3**: Normal 0.08-0.2 ng/dL (normalized)
- **T4/TT4**: Normal 0.05-0.15 μg/dL (normalized)
- **FTI**: Normal 0.05-0.15

### Clinical Patterns
- **Primary Hypothyroidism**: High TSH, Low T4
- **Primary Hyperthyroidism**: Low TSH, High T4/T3
- **Subclinical Hypothyroidism**: High TSH, Normal T4
- **Subclinical Hyperthyroidism**: Low TSH, Normal T4

## Files Generated

### Data Files
- `ann_thyroid_engineered.csv`: 7,129 records, 82 features
- `hypothyroid_engineered.csv`: 3,002 records, 83 features
- `new_thyroid_engineered.csv`: 215 records, 34 features

### Reports
- `feature_engineering_summary.csv`: Overview of all datasets

## Usage Recommendations

### For Machine Learning

#### Classification Tasks
1. **Use clinical risk scores** for interpretable models
2. **Include hormone ratios** for capturing physiological relationships
3. **Add z-scored features** for distance-based algorithms
4. **Consider treatment features** for patient stratification

#### Feature Selection
- Start with **hormone features and ratios** (highest clinical relevance)
- Add **demographic features** for personalization
- Include **treatment features** for context
- Use **statistical features** for normalization

### For Clinical Analysis

#### Diagnostic Support
1. Focus on **clinical risk scores** for screening
2. Use **abnormal hormone count** for severity assessment
3. Analyze **hormone ratios** for differential diagnosis

#### Patient Stratification
1. Group by **treatment count** and **condition count**
2. Consider **age groups** for age-specific analysis
3. Use **complex case indicator** for resource allocation

### Model Recommendations

#### Ann_Thyroid Dataset
- **Best for**: Multi-class classification (3 classes)
- **Key features**: TSH, T3, T4, FTI, hormone ratios
- **Challenge**: Class imbalance (subnormal class)

#### Hypothyroid Dataset
- **Best for**: Binary classification (hypothyroid vs normal)
- **Key features**: TSH, T4, treatment history
- **Advantage**: Largest dataset, good for deep learning

#### New_Thyroid Dataset
- **Best for**: TRH stimulation test analysis
- **Key features**: TSH response, serum hormone levels
- **Challenge**: Smallest dataset, may need regularization

## Feature Selection Strategy

### Priority Levels

**High Priority** (Always include):
- TSH, T3, T4, FTI (core hormones)
- Clinical risk scores
- Age and sex

**Medium Priority** (Include for better performance):
- Hormone ratios
- Treatment indicators
- Abnormal hormone count

**Low Priority** (Include if needed):
- Polynomial features
- Statistical features
- Complex interactions

### Dimensionality Reduction
With 80+ features, consider:
1. **Correlation analysis**: Remove features with r > 0.95
2. **Feature importance**: Use Random Forest or XGBoost
3. **PCA**: For visualization and noise reduction
4. **Domain knowledge**: Prioritize clinically validated features

## Clinical Validation

### Recommended Checks
1. Verify risk scores align with clinical diagnoses
2. Validate hormone ratios against medical literature
3. Check treatment features for logical consistency
4. Ensure age/sex patterns match epidemiology

### Quality Metrics
- **Sensitivity**: Ability to detect thyroid disease
- **Specificity**: Ability to rule out disease
- **PPV/NPV**: Predictive values for clinical utility
- **AUC-ROC**: Overall discriminative ability

## Next Steps

1. **Feature Selection**: Reduce to 20-30 most important features
2. **Model Training**: Compare multiple algorithms
3. **Clinical Validation**: Test against medical guidelines
4. **Interpretability**: Use SHAP or LIME for explanations
5. **Deployment**: Create clinical decision support tool

## Script Location
`thyroid_project/scripts/feature_engineering.py`

## Date Created
2026-06-11

## References
- American Thyroid Association Guidelines
- Clinical Thyroid Function Tests
- UCI Machine Learning Repository - Thyroid Disease Dataset