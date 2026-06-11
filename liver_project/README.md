# BUPA Liver Disorders Dataset - Analysis Project

## Overview
This project contains a comprehensive analysis of the BUPA Liver Disorders dataset, including data cleaning, exploratory data analysis, and visualization.

## Dataset Information
- **Source**: BUPA Medical Research Ltd.
- **Original Records**: 345 male individuals
- **Cleaned Records**: 341 (after removing 4 duplicates)
- **Features**: 10 (7 original + 3 engineered)
- **Target Variable**: Selector (1 or 2)

## Features

### Original Features
1. **mcv** - Mean Corpuscular Volume
2. **alkphos** - Alkaline Phosphotase
3. **sgpt** - Alamine Aminotransferase (ALT)
4. **sgot** - Aspartate Aminotransferase (AST)
5. **gammagt** - Gamma-Glutamyl Transpeptidase
6. **drinks** - Number of half-pint equivalents of alcoholic beverages drunk per day
7. **selector** - Field used to split data into two sets (target variable)

### Engineered Features
1. **ast_alt_ratio** - AST/ALT ratio (clinically significant for liver disease)
2. **alcohol_category** - Categorical grouping of alcohol consumption (None, Light, Moderate, Heavy, Very Heavy)
3. **total_enzyme_score** - Composite score of normalized enzyme levels

## Project Structure

```
liver_project/
├── bupa.data                    # Original data file
├── bupa.names                   # Dataset metadata
├── noteDuplicates.txt          # Known duplicate records
├── data_cleaning_plan.md       # Comprehensive cleaning strategy
├── README.md                   # This file
├── scripts/
│   ├── initial_assessment.py   # Initial data exploration
│   ├── data_cleaning.py        # Data cleaning pipeline
│   └── exploratory_analysis.py # EDA with visualizations
├── output/
│   └── liver_cleaned.csv       # Cleaned dataset (341 records, 10 features)
├── reports/
│   ├── initial_assessment_summary.csv
│   ├── detailed_statistics.csv
│   ├── correlation_matrix.csv
│   ├── duplicate_rows.csv
│   ├── data_cleaning_report.csv
│   ├── removed_duplicates.csv
│   ├── outlier_analysis.csv
│   ├── validation_results.csv
│   ├── selector_comparison.csv
│   └── eda_summary.csv
└── plots/
    ├── selector_distribution.png
    ├── numerical_distributions.png
    ├── boxplots.png
    ├── correlation_heatmap.png
    ├── features_by_selector.png
    ├── alcohol_analysis.png
    ├── ast_alt_ratio_analysis.png
    └── pairplot.png
```

## Key Findings

### Data Quality
- **No missing values** in the original dataset
- **4 duplicate rows** identified and removed
- **Outliers present** but retained as valid medical values
- **Data quality score**: 98.84%

### Target Variable Distribution
- **Selector 1**: 142 records (41.6%)
- **Selector 2**: 199 records (58.4%)
- Slightly imbalanced but acceptable for analysis

### Significant Features (p < 0.05)
1. **AST/ALT Ratio** - Strongest correlation with selector (r = 0.26, p < 0.0001)
2. **SGOT (AST)** - Significant difference between groups (p = 0.003)
3. **GammaGT** - Significant difference between groups (p = 0.007)

### Clinical Insights
- **Mean alcohol consumption**: 3.43 drinks/day
- **High AST/ALT ratio (>2)**: 10 records (2.9%) - potential alcoholic liver disease indicator
- **Alcohol categories**: Light (36.7%), Moderate (30.8%), Heavy (23.8%), Very Heavy (6.2%)

### Correlation Analysis
- **Strongest positive correlations**:
  - SGPT & SGOT: 0.44
  - AlkPhos & MCV: 0.28
  - GammaGT & Drinks: 0.42
- **Selector correlations**:
  - AST/ALT ratio shows the strongest relationship (0.26)
  - Most other features show weak correlations

## Usage

### Running the Analysis

1. **Initial Assessment**:
```bash
cd liver_project/scripts
python initial_assessment.py
```

2. **Data Cleaning**:
```bash
python data_cleaning.py
```

3. **Exploratory Analysis**:
```bash
python exploratory_analysis.py
```

### Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scipy
```

## Data Cleaning Process

### Phase 1: Loading & Validation
- Loaded data with proper column names
- Verified data types
- Confirmed no missing values

### Phase 2: Duplicate Handling
- Identified 4 exact duplicate rows
- Removed duplicates (kept first occurrence)
- Documented all removed records

### Phase 3: Outlier Analysis
- Detected outliers using IQR method
- Kept all outliers (valid medical values)
- Documented outlier statistics

### Phase 4: Feature Engineering
- Created AST/ALT ratio (clinical significance)
- Categorized alcohol consumption
- Generated composite enzyme score

### Phase 5: Validation
- Verified no negative values (except normalized scores)
- Confirmed selector field integrity
- Validated all data types

## Exploratory Data Analysis

### Visualizations Generated
1. **Selector Distribution** - Bar and pie charts
2. **Numerical Distributions** - Histograms with mean/median lines
3. **Box Plots** - Outlier visualization for all features
4. **Correlation Heatmap** - Feature relationships
5. **Features by Selector** - Comparative histograms
6. **Alcohol Analysis** - Category distribution and selector comparison
7. **AST/ALT Ratio Analysis** - Clinical threshold visualization
8. **Pairplot** - Multivariate relationships

### Statistical Tests
- Independent t-tests for group comparisons
- Correlation analysis
- Outlier detection (IQR method)

## Clinical Relevance

### AST/ALT Ratio
- **Normal range**: 0.8 - 1.0
- **Ratio > 2**: Suggests alcoholic liver disease
- **Dataset**: 2.9% with ratio > 2

### Enzyme Levels
- **SGPT (ALT)**: Liver-specific enzyme
- **SGOT (AST)**: Found in liver and other tissues
- **GammaGT**: Sensitive to alcohol consumption
- **AlkPhos**: Indicates bile duct issues

### Alcohol Consumption
- Strong correlation with GammaGT (0.42)
- Weak correlation with selector (-0.01)
- Most subjects are light to moderate drinkers

## Future Work

### Potential Analyses
1. Machine learning classification (predict selector)
2. Clustering analysis (identify patient subgroups)
3. Time series analysis (if temporal data available)
4. Feature importance ranking
5. Model comparison (Logistic Regression, Random Forest, SVM, etc.)

### Data Enhancement
1. Collect additional demographic data
2. Include temporal measurements
3. Add clinical outcomes
4. Incorporate genetic markers

## References

- BUPA Medical Research Ltd. (Original data source)
- UCI Machine Learning Repository
- Clinical liver function test interpretation guidelines

## License
This analysis is for educational and research purposes.

## Contact
For questions or collaboration, please refer to the project repository.

---

**Last Updated**: June 10, 2026
**Analysis Version**: 1.0
**Dataset Version**: Cleaned (341 records)