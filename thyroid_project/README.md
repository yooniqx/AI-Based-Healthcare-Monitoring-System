# Thyroid Disease Dataset - Data Cleaning Project

## Project Overview
This project implements a comprehensive data cleaning pipeline for three thyroid disease datasets from the UCI Machine Learning Repository. The datasets are used for diagnosing thyroid conditions including hypothyroidism, hyperthyroidism, and normal thyroid function.

## Project Structure
```
thyroid_project/
├── data/                          # Original data files (extracted from zip)
├── output/                        # Cleaned datasets
│   ├── hypothyroid_cleaned.csv
│   ├── ann_thyroid_cleaned.csv
│   └── new_thyroid_cleaned.csv
├── reports/                       # Data quality and cleaning reports
│   ├── datasets_summary.csv
│   ├── data_cleaning_report.csv
│   ├── hypothyroid_missing_values.csv
│   ├── hypothyroid_numerical_stats.csv
│   ├── hypothyroid_categorical_info.csv
│   ├── hypothyroid_class_distribution.csv
│   ├── ann-thyroid_*.csv
│   └── new-thyroid_*.csv
├── scripts/                       # Python scripts
│   ├── initial_assessment.py     # Data exploration and quality assessment
│   └── data_cleaning.py          # Main data cleaning pipeline
├── data_cleaning_plan.md         # Detailed cleaning strategy
└── README.md                      # This file
```

## Datasets

### 1. Hypothyroid Dataset
- **Original Size**: 3,163 records, 26 features
- **Cleaned Size**: 3,011 records, 24 features
- **Target**: Binary classification (negative vs hypothyroid)
- **Class Distribution**: 95.32% negative, 4.68% hypothyroid
- **Key Changes**:
  - Dropped TBG column (91.8% missing)
  - Removed 152 duplicate records (4.81%)
  - Imputed missing values in 7 columns
  - Converted all categorical features to binary (0/1)
  - Capped outliers in 5 numerical columns

### 2. ANN-Thyroid Dataset
- **Original Size**: 7,200 records (3,772 train + 3,428 test), 23 features
- **Cleaned Size**: 6,981 records, 23 features
- **Target**: 3-class classification (normal, hyperfunction, subnormal)
- **Key Changes**:
  - Fixed class labels (mapped 1/2/3 to descriptive labels)
  - Removed 219 duplicate records (3.04%)
  - Fixed invalid values in binary features
  - Capped outliers in 5 continuous columns
  - Preserved train/test split information

### 3. New-Thyroid Dataset
- **Original Size**: 215 records, 6 features
- **Cleaned Size**: 215 records, 6 features
- **Target**: 3-class classification (normal, hyperthyroid, hypothyroid)
- **Class Distribution**: 69.77% normal, 16.28% hyperthyroid, 13.95% hypothyroid
- **Key Changes**:
  - Mapped class labels (1/2/3 to descriptive labels)
  - Capped outliers in all 5 numerical features
  - Kept negative values in max_TSH_difference (medically valid)
  - No missing values or duplicates

## Data Cleaning Pipeline

### Phase 1: Initial Assessment
Run `initial_assessment.py` to:
- Load and explore all three datasets
- Identify missing values, duplicates, and outliers
- Generate data quality reports
- Document class distributions

```bash
cd thyroid_project/scripts
python initial_assessment.py
```

### Phase 2: Data Cleaning
Run `data_cleaning.py` to:
- Handle missing values (imputation or column removal)
- Remove duplicate records
- Convert categorical features to numerical
- Cap outliers using IQR method
- Validate cleaned data
- Export cleaned datasets

```bash
cd thyroid_project/scripts
python data_cleaning.py
```

## Data Quality Issues Addressed

### Missing Values
- **Hypothyroid**: Imputed 2,073 missing values across 7 columns
- **ANN-Thyroid**: No missing values in original data
- **New-Thyroid**: No missing values

### Duplicates
- **Hypothyroid**: Removed 152 duplicates (4.81%)
- **ANN-Thyroid**: Removed 219 duplicates (3.04%)
- **New-Thyroid**: No duplicates found

### Outliers
- Applied IQR-based capping to all numerical features
- Preserved medical validity (e.g., negative TSH differences)
- Documented outlier counts and percentages

### Data Type Conversions
- Converted binary categorical features (f/t) to numerical (0/1)
- Mapped sex (M/F) to binary (1/0)
- Converted class labels to descriptive strings or binary

## Key Findings

### Hypothyroid Dataset
- Highly imbalanced dataset (95% negative cases)
- TBG column dropped due to 91.8% missing values
- Age, TSH, and T3 had significant missing values (14-22%)
- Multiple outliers in thyroid hormone measurements

### ANN-Thyroid Dataset
- Class labels were encoded as integers (1/2/3)
- Some binary features had invalid values (fixed)
- Relatively clean dataset with minimal missing data
- Preserved train/test split for modeling

### New-Thyroid Dataset
- Cleanest dataset with no missing values
- Small dataset (215 records) suitable for initial testing
- Balanced class distribution compared to other datasets
- Negative values in max_TSH_difference are medically valid

## Usage

### Loading Cleaned Data
```python
import pandas as pd

# Load cleaned datasets
df_hypothyroid = pd.read_csv('output/hypothyroid_cleaned.csv')
df_ann = pd.read_csv('output/ann_thyroid_cleaned.csv')
df_new = pd.read_csv('output/new_thyroid_cleaned.csv')

# Check data
print(df_hypothyroid.info())
print(df_hypothyroid['class'].value_counts())
```

### Feature Information

#### Hypothyroid Dataset Features (24)
- **Target**: class (0=negative, 1=hypothyroid)
- **Demographics**: age, sex
- **Medical History**: on_thyroxine, query_on_thyroxine, on_antithyroid_medication, thyroid_surgery, query_hypothyroid, query_hyperthyroid, pregnant, sick, tumor, lithium, goitre
- **Lab Measurements**: TSH, T3, TT4, T4U, FTI
- **Measurement Flags**: TSH_measured, T3_measured, TT4_measured, T4U_measured, FTI_measured

#### ANN-Thyroid Dataset Features (23)
- **Target**: class (normal, hyperfunction, subnormal)
- **Demographics**: age, sex
- **Medical History**: on_thyroxine, query_on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre, tumor, hypopituitary, psych
- **Lab Measurements**: TSH, T3, TT4, T4U, FTI
- **Metadata**: dataset_split (train/test)

#### New-Thyroid Dataset Features (6)
- **Target**: class (normal, hyperthyroid, hypothyroid)
- **Lab Measurements**: 
  - T3_resin_uptake
  - total_serum_thyroxin
  - total_serum_triiodothyronine
  - basal_TSH
  - max_TSH_difference

## Data Quality Metrics

### Before Cleaning
- **Total Records**: 10,578
- **Missing Values**: 4,176 (hypothyroid only)
- **Duplicates**: 371 (3.51%)
- **Outliers**: ~8-12% per numerical column

### After Cleaning
- **Total Records**: 10,207 (3.5% reduction)
- **Missing Values**: 0
- **Duplicates**: 0
- **Outliers**: Capped to valid ranges

## Next Steps

1. **Exploratory Data Analysis (EDA)**
   - Visualize feature distributions
   - Analyze correlations
   - Identify important features

2. **Feature Engineering**
   - Create age groups
   - Derive hormone ratios
   - Add interaction features

3. **Model Development**
   - Handle class imbalance (SMOTE, class weights)
   - Train classification models
   - Evaluate performance metrics

4. **Model Evaluation**
   - Cross-validation
   - Confusion matrices
   - ROC curves and AUC scores

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

## References

- UCI Machine Learning Repository: Thyroid Disease Dataset
- Quinlan, J.R. (1987). Simplifying Decision Trees. International Journal of Man-Machine Studies, 27, 221-234.
- Garavan Institute, Sydney, Australia (Data Source)

## License

This project uses publicly available datasets from the UCI Machine Learning Repository.

## Author

Data Cleaning Pipeline - 2026

---

**Last Updated**: 2026-06-08