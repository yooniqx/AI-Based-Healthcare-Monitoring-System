# BUPA Liver Disorders - Feature Engineering Documentation

## Overview
This document describes the comprehensive feature engineering performed on the BUPA Liver Disorders dataset. The feature engineering process transformed the original 10 features into 100 features, creating 90 new engineered features.

## Dataset Information
- **Original Features**: 10
- **Engineered Features**: 100
- **New Features Created**: 90
- **Total Records**: 341

## Feature Engineering Categories

### 1. Enzyme-Based Features (14 features)
Clinical liver enzyme ratios and combinations:
- **Ratios**: `sgpt_sgot_ratio`, `sgot_sgpt_ratio`, `gammagt_sgpt_ratio`, `gammagt_sgot_ratio`
- **Combinations**: `sgpt_sgot_sum`, `sgpt_sgot_product`, `all_enzymes_sum`, `all_enzymes_mean`
- **Dominance Indicators**: `sgpt_dominant`, `sgot_dominant`, `gammagt_dominant`
- **Elevation Indicators**: `sgpt_elevated`, `sgot_elevated`, `gammagt_elevated`, `elevated_enzyme_count`

**Clinical Significance**: Enzyme ratios help differentiate between types of liver damage. AST/ALT ratio > 2 suggests alcoholic liver disease.

### 2. Alkaline Phosphatase Features (7 features)
- **Ratios**: `alkphos_sgpt_ratio`, `alkphos_sgot_ratio`, `alkphos_gammagt_ratio`
- **Elevation Indicators**: `alkphos_elevated`, `alkphos_low`
- **Interactions**: `alkphos_mcv_ratio`, `alkphos_mcv_product`

**Clinical Significance**: Alkaline phosphatase helps distinguish between hepatocellular and cholestatic liver injury.

### 3. MCV (Mean Corpuscular Volume) Features (7 features)
- **Categories**: `mcv_low` (microcytic), `mcv_normal`, `mcv_high` (macrocytic)
- **Deviations**: `mcv_deviation`, `mcv_normalized`
- **Interactions**: `mcv_drinks_interaction`, `mcv_alcohol_category_encoded`

**Clinical Significance**: Elevated MCV is associated with chronic alcohol consumption and vitamin B12/folate deficiency.

### 4. Alcohol-Related Features (10 features)
- **Transformations**: `drinks_squared`, `drinks_log`, `drinks_sqrt`
- **Encoding**: `alcohol_level` (0-4 scale)
- **Interactions**: `drinks_sgpt_interaction`, `drinks_sgot_interaction`, `drinks_gammagt_interaction`, `drinks_ast_alt_ratio_interaction`
- **Indicators**: `heavy_drinker` (>2 drinks/day)
- **Risk Score**: `alcohol_enzyme_risk`

**Clinical Significance**: Alcohol consumption is a major risk factor for liver disease.

### 5. Clinical Risk Scores (5 features)
Composite scores based on clinical guidelines:
- **`liver_damage_score`**: Weighted combination of elevated enzymes and AST/ALT ratio
- **`alcoholic_liver_risk`**: Risk score for alcoholic liver disease
- **`hepatocellular_score`**: Indicates hepatocellular injury pattern
- **`cholestatic_score`**: Indicates cholestatic injury pattern
- **`liver_health_score`**: Overall liver health indicator (inverse of damage)

**Clinical Significance**: These scores provide interpretable risk assessments for different types of liver pathology.

### 6. Statistical Features (19 features)
- **Z-scores**: Standardized values for all 6 key features
- **Robust Scaling**: Median-centered scaling for outlier resistance
- **Outlier Detection**: Binary indicators for values >3 standard deviations
- **Aggregate**: `total_outliers` count

**Purpose**: Normalize features for machine learning and identify anomalous values.

### 7. Polynomial Features (6 features)
Non-linear transformations:
- **Squared**: `sgpt_squared`, `sgot_squared`, `gammagt_squared`, `alkphos_squared`
- **Cubed**: `gammagt_cubed`, `drinks_cubed`

**Purpose**: Capture non-linear relationships in the data.

### 8. Binned Features (8 features)
Categorical representations of continuous features:
- **Enzyme Bins**: `sgpt_bin`, `sgot_bin`, `gammagt_bin` (Low/Normal/Elevated/High)
- **Alcohol Bins**: `drinks_bin` (None/Light/Moderate/Heavy/Very Heavy)
- **Encoded Versions**: Numeric encoding of all bins

**Purpose**: Enable categorical analysis and decision tree algorithms.

### 9. Interaction Features (5 features)
Multi-feature interactions:
- `mcv_alkphos_sgpt`: Three-way interaction
- `sgpt_sgot_gammagt`: Enzyme interaction
- `drinks_mcv_gammagt`: Alcohol-related interaction
- `ast_alt_drinks`: Ratio-alcohol interaction
- `enzyme_score_drinks`: Score-alcohol interaction

**Purpose**: Capture complex relationships between features.

### 10. Aggregate Features (8 features)
Summary statistics across features:
- **Abnormality Metrics**: `abnormality_count`, `abnormality_ratio`
- **Enzyme Statistics**: `enzyme_mean`, `enzyme_std`, `enzyme_cv` (coefficient of variation)
- **Range Features**: `enzyme_range`, `enzyme_max`, `enzyme_min`

**Purpose**: Provide overall health indicators and variability measures.

## Clinical Reference Ranges

### Liver Enzymes
- **SGPT (ALT)**: Normal 7-56 U/L
- **SGOT (AST)**: Normal 10-40 U/L
- **Gamma-GT**: Normal 9-48 U/L
- **Alkaline Phosphatase**: Normal 30-120 U/L

### Blood Parameters
- **MCV**: Normal 80-100 fL
  - <80: Microcytic (iron deficiency)
  - >100: Macrocytic (B12/folate deficiency, alcohol)

### Clinical Thresholds
- **AST/ALT Ratio > 2**: Suggests alcoholic liver disease
- **Heavy Drinking**: >2 drinks per day
- **Enzyme Elevation**: >1.5x upper normal limit

## Files Generated

### Data Files
- `liver_engineered.csv`: Complete dataset with all 100 features

### Reports
- `feature_engineering_report.csv`: Statistics for all new features
- `engineered_features_list.csv`: Mapping of original to new features

## Usage Recommendations

### For Machine Learning
1. **Start with clinical risk scores** for interpretability
2. **Use z-scored features** for algorithms sensitive to scale (SVM, neural networks)
3. **Include interaction features** for ensemble methods (Random Forest, XGBoost)
4. **Consider binned features** for decision trees and rule-based models

### For Clinical Analysis
1. Focus on **clinical risk scores** and **enzyme ratios**
2. Use **abnormality counts** for patient stratification
3. Analyze **alcohol-related features** for lifestyle interventions

### Feature Selection
With 100 features, consider:
- **Correlation analysis**: Remove highly correlated features (>0.95)
- **Feature importance**: Use tree-based models to identify key features
- **Domain knowledge**: Prioritize clinically meaningful features
- **Regularization**: Use L1 (Lasso) for automatic feature selection

## Next Steps
1. Perform feature selection to reduce dimensionality
2. Train machine learning models on engineered features
3. Compare performance with original features
4. Validate clinical risk scores against medical outcomes

## Script Location
`liver_project/scripts/feature_engineering.py`

## Date Created
2026-06-11