# Feature Engineering Documentation - Gallstone Disease Prediction

## Overview
This document describes the feature engineering process applied to the gallstone disease dataset. A total of **56 new features** were created from the original 39 features, resulting in a final dataset with **95 features**.

## Feature Engineering Strategy

The feature engineering approach was guided by:
1. **Medical domain knowledge** about gallstone disease risk factors
2. **Statistical relationships** identified in exploratory data analysis
3. **Clinical biomarkers** relevant to metabolic and liver health
4. **Interaction effects** between key predictors

## Feature Categories

### 1. Body Composition Features (15 features)

These features capture the relationship between different body composition components:

- **fat_lean_ratio**: Ratio of total fat to lean mass (obesity indicator)
- **visceral_subcutaneous_ratio**: Ratio of visceral to subcutaneous fat
- **water_distribution_ratio**: Extracellular to intracellular water ratio
- **muscle_fat_ratio**: Muscle mass to fat content ratio
- **protein_weight_ratio**: Body protein content relative to weight
- **body_water_percentage**: Total body water as percentage of weight
- **lean_bmi**: BMI calculated using lean mass only
- **fat_mass_index**: Fat mass normalized by height squared

**Clinical Relevance**: Body composition, especially visceral fat distribution, is strongly associated with gallstone formation due to metabolic dysregulation.

### 2. Metabolic Features (22 features)

Features related to lipid metabolism and metabolic syndrome:

- **atherogenic_index**: Total cholesterol to HDL ratio (cardiovascular risk)
- **ldl_hdl_ratio**: LDL to HDL cholesterol ratio
- **non_hdl_cholesterol**: Total cholesterol minus HDL
- **tg_hdl_ratio**: Triglyceride to HDL ratio (insulin resistance marker)
- **cholesterol_saturation**: Composite measure of cholesterol burden
- **metabolic_syndrome_score**: Count of metabolic syndrome criteria met
- **lipid_burden**: Sum of total cholesterol, triglycerides, and LDL

**Clinical Relevance**: Dyslipidemia and metabolic syndrome are major risk factors for cholesterol gallstone formation.

### 3. Liver Function Features (6 features)

Markers of hepatic health and function:

- **ast_alt_ratio**: De Ritis ratio (AST/ALT) for liver damage assessment
- **liver_enzyme_score**: Count of elevated liver enzymes
- **combined_liver_enzyme**: Sum of AST, ALT, and alkaline phosphatase
- **hepatic_fat_bmi_interaction**: Interaction between hepatic fat and BMI
- **liver_stress_indicator**: Composite liver enzyme ratio

**Clinical Relevance**: Liver function is directly related to bile composition and gallstone formation.

### 4. Cardiovascular Features (2 features)

Risk factors for cardiovascular disease:

- **atherogenic_index**: TC/HDL ratio
- **cv_risk_score**: Composite cardiovascular risk score

**Clinical Relevance**: Cardiovascular risk factors often co-occur with gallstone disease.

### 5. Renal Function Features (2 features)

Kidney function indicators:

- **creatinine_gfr_ratio**: Creatinine to GFR ratio
- **kidney_function_category**: Categorical kidney function status (0=normal, 1=mildly reduced, 2=impaired)

**Clinical Relevance**: Renal function affects metabolic homeostasis and may influence gallstone risk.

### 6. Inflammatory Features (9 features)

Markers of systemic inflammation:

- **inflammation_score**: Composite inflammation indicator
- **visceral_inflammation_proxy**: Visceral fat × BMI interaction
- **visceral_adiposity_index**: Simplified VAI calculation

**Clinical Relevance**: Chronic inflammation is associated with metabolic dysfunction and gallstone formation.

### 7. Nutritional Features (3 features)

Nutritional status indicators:

- **vitamin_d_deficiency**: Binary indicator (vitamin D < 30)
- **anemia_indicator**: Binary indicator (hemoglobin < 13)
- **nutritional_risk_score**: Composite nutritional risk

**Clinical Relevance**: Nutritional deficiencies may affect bile composition and gallstone risk.

### 8. Age-Related Features (8 features)

Age interactions with key risk factors:

- **age_group**: Categorical age groups (0: <40, 1: 40-50, 2: 50-60, 3: >60)
- **age_bmi_interaction**: Age × BMI
- **age_visceral_interaction**: Age × visceral fat rating
- **age_cholesterol_interaction**: Age × total cholesterol
- **age_hdl_interaction**: Age × HDL cholesterol

**Clinical Relevance**: Gallstone risk increases with age, and age modifies the effect of other risk factors.

### 9. Comorbidity Features (1 feature)

Disease burden indicators:

- **total_comorbidities**: Count of all comorbid conditions
- **metabolic_disease_burden**: Count of metabolic diseases

**Clinical Relevance**: Multiple comorbidities increase gallstone risk through various mechanisms.

### 10. Interaction Features (11 features)

Key two-way interactions:

- **bmi_triglyceride_interaction**: BMI × triglycerides
- **bmi_cholesterol_interaction**: BMI × total cholesterol
- **visceral_glucose_interaction**: Visceral fat × glucose
- **visceral_triglyceride_interaction**: Visceral fat × triglycerides
- **fat_ratio_glucose_interaction**: Body fat ratio × glucose
- **fat_ratio_cholesterol_interaction**: Body fat ratio × cholesterol

**Clinical Relevance**: Interaction effects capture synergistic relationships between risk factors.

### 11. Polynomial Features (12 features)

Non-linear transformations of key predictors:

For each of these features, squared and cubed terms were created:
- body_mass_index
- age
- visceral_fat_rating
- total_cholesterol
- triglyceride
- glucose

**Clinical Relevance**: Captures non-linear dose-response relationships between risk factors and disease.

### 12. Risk Scores (7 features)

Composite risk assessment scores:

- **gallstone_risk_score**: Weighted composite of known gallstone risk factors
- **metabolic_health_score**: Overall metabolic health indicator
- **metabolic_syndrome_score**: Count of metabolic syndrome criteria
- **liver_enzyme_score**: Liver function assessment
- **cv_risk_score**: Cardiovascular risk assessment
- **inflammation_score**: Inflammatory burden
- **nutritional_risk_score**: Nutritional status assessment

**Clinical Relevance**: Composite scores provide holistic risk assessment and may improve prediction.

## Feature Engineering Process

### Step 1: Data Loading
- Loaded cleaned dataset with 39 original features
- 5 records with complete data

### Step 2: Feature Creation
Applied 12 feature engineering functions sequentially:
1. Body composition features
2. Metabolic features
3. Liver function features
4. Cardiovascular features
5. Renal function features
6. Inflammatory features
7. Nutritional features
8. Age-related features
9. Comorbidity features
10. Interaction features
11. Polynomial features
12. Risk scores

### Step 3: Data Quality
- Handled infinite values (replaced with NaN)
- Imputed missing values with median
- Validated all features for consistency

### Step 4: Documentation
- Created feature engineering report
- Documented feature categories
- Generated summary statistics

## Key Statistics

| Metric | Value |
|--------|-------|
| Original Features | 39 |
| Engineered Features | 56 |
| Total Features | 95 |
| Records | 5 |
| Missing Values | 0 |
| Data Quality | 100% |

## Feature Importance Considerations

Based on domain knowledge, the following engineered features are expected to be highly predictive:

1. **gallstone_risk_score**: Composite of known risk factors
2. **metabolic_syndrome_score**: Strong association with gallstone disease
3. **atherogenic_index**: Lipid metabolism indicator
4. **visceral_adiposity_index**: Central obesity measure
5. **hepatic_fat_bmi_interaction**: Liver fat and obesity interaction
6. **tg_hdl_ratio**: Insulin resistance marker
7. **age_bmi_interaction**: Age-modified obesity effect

## Usage

### Loading the Engineered Dataset

```python
import pandas as pd

# Load engineered features
df = pd.read_csv('gallstone_project/ml_ready/gallstone_engineered.csv')

# Separate features and target
X = df.drop('gallstone_status', axis=1)
y = df['gallstone_status']
```

### Feature Selection Recommendations

Given the high dimensionality (95 features) relative to sample size (5 records), consider:

1. **Feature Selection Methods**:
   - Correlation-based selection
   - Recursive Feature Elimination (RFE)
   - LASSO regularization
   - Random Forest feature importance

2. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA)
   - Feature clustering
   - Domain-guided selection

3. **Model-Based Selection**:
   - Use regularized models (Ridge, LASSO, Elastic Net)
   - Tree-based models with built-in feature selection

## Files Generated

1. **gallstone_engineered.csv**: Complete dataset with all features
2. **engineered_features_report.csv**: Statistics for each engineered feature
3. **feature_categories_summary.csv**: Summary by feature category

## Next Steps

1. **Feature Selection**: Reduce dimensionality for modeling
2. **Feature Validation**: Validate features on larger dataset when available
3. **Model Training**: Train predictive models using engineered features
4. **Feature Importance Analysis**: Identify most predictive features
5. **Clinical Validation**: Validate feature relevance with domain experts

## References

- Metabolic syndrome and gallstone disease: A systematic review
- Body composition and gallstone formation
- Lipid metabolism in gallstone pathogenesis
- Age and gender effects on gallstone prevalence

## Notes

⚠️ **Important**: The current dataset has only 5 samples, which is insufficient for robust machine learning. The engineered features are designed to be used when a larger dataset becomes available.

---

*Generated: 2026-06-11*
*Script: feature_engineering.py*