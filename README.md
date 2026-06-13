# Healthcare Monitoring and Early Disease Prediction System

A comprehensive Python-based system for healthcare monitoring and early disease prediction across multiple health conditions. This project implements a complete machine learning pipeline from data collection to model-ready datasets.

## 🎯 Project Overview

This system aims to enable early detection of various health conditions through data-driven analysis and machine learning. The project currently supports prediction models for nine different health conditions.

## 🏥 Disease Modules

The system includes the following disease prediction modules:

1. **Heart Disease** - Cardiovascular health risk assessment
2. **Kidney Disease** - Renal function monitoring and prediction
3. **Lung Disease** - Respiratory health analysis
4. **Diabetes** - Blood glucose level monitoring and risk prediction
5. **Thyroid Disorders** - Thyroid function assessment
6. **Liver Disease** - Hepatic function evaluation
7. **Gallbladder Disease** - Gallstone risk prediction
8. **Mental Health** - Stress and mental health assessment
9. **Survey Data** - General health survey analysis

## 🛠️ Technology Stack

- **Python 3.x** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-Learn** - Machine learning and preprocessing
- **Matplotlib/Seaborn** - Data visualization
- **Streamlit** - Web application framework (planned)

## 📁 Project Structure

```
PROJECT/
│
├── datasets/                          # Disease-specific datasets
│   ├── heart/
│   │   ├── heart_master_dataset.csv
│   │   └── ml_ready/                  # Train/test splits
│   │       ├── X_train.csv
│   │       ├── X_test.csv
│   │       ├── y_train.csv
│   │       └── y_test.csv
│   │
│   ├── kidney/
│   ├── lung/
│   ├── diabetes/
│   ├── thyroid/
│   ├── liver/
│   ├── gallbladder/
│   ├── mental_health/
│   └── survey/
│
├── reports/                           # Analysis reports and visualizations
│   ├── heart/
│   │   ├── MODULE_SUMMARY.txt
│   │   └── eda/                       # Exploratory data analysis plots
│   │
│   ├── kidney/
│   ├── lung/
│   ├── diabetes/
│   ├── thyroid/
│   ├── liver/
│   ├── gallbladder/
│   ├── mental_health/
│   └── survey/
│
├── scripts/                           # Data processing scripts
│   ├── preprocessing/                 # Data cleaning scripts
│   ├── eda/                          # Exploratory analysis scripts
│   ├── feature_engineering/          # Feature creation scripts
│   ├── initial_assessment/           # Data assessment scripts
│   └── utilities/                    # Helper utilities
│
└── README.md
```

## 🔄 Machine Learning Pipeline

### ✅ Completed Stages

1. **Problem Definition** - Defined prediction tasks for each disease module
2. **Data Collection** - Gathered datasets from various healthcare sources
3. **Data Cleaning** - Handled missing values, duplicates, and outliers
4. **Exploratory Data Analysis (EDA)** - Statistical analysis and visualization
5. **Feature Engineering** - Created and transformed features for better predictions
6. **Train/Test Split** - Generated 80/20 stratified splits for all modules

### 🔜 Upcoming Stages

- Model Selection - Choose appropriate ML algorithms
- Model Training - Train models on prepared datasets
- Model Evaluation - Assess model performance
- Hyperparameter Tuning - Optimize model parameters
- Model Deployment - Deploy via Streamlit web application

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Dataset Structure

Each disease module follows a standardized structure:

- **Master Dataset**: `datasets/{module}/{module}_master_dataset.csv`
  - Contains all cleaned and engineered features
  - Ready for machine learning model training

- **ML-Ready Splits**: `datasets/{module}/ml_ready/`
  - `X_train.csv` - Training features (80%)
  - `X_test.csv` - Testing features (20%)
  - `y_train.csv` - Training labels
  - `y_test.csv` - Testing labels
  - Features are standardized using StandardScaler
  - Splits are stratified to maintain class distribution

### Usage

#### 1. Explore Datasets

```python
import pandas as pd

# Load master dataset
df = pd.read_csv('datasets/heart/heart_master_dataset.csv')
print(df.head())
```

#### 2. Load ML-Ready Data

```python
# Load train/test splits
X_train = pd.read_csv('datasets/heart/ml_ready/X_train.csv')
X_test = pd.read_csv('datasets/heart/ml_ready/X_test.csv')
y_train = pd.read_csv('datasets/heart/ml_ready/y_train.csv')
y_test = pd.read_csv('datasets/heart/ml_ready/y_test.csv')
```

#### 3. View Analysis Reports

Check the `reports/{module}/MODULE_SUMMARY.txt` for:
- Dataset statistics
- Data cleaning summary
- EDA findings
- Feature engineering details
- Recommended ML models

#### 4. Run Processing Scripts

```bash
# Data preprocessing
python scripts/preprocessing/data_cleaning.py

# Exploratory analysis
python scripts/eda/exploratory_analysis.py

# Feature engineering
python scripts/feature_engineering/feature_engineering.py

# Generate ML-ready splits
python scripts/utilities/create_ml_ready_splits.py
```

## 📊 Dataset Statistics

| Module | Samples | Features | Target Column | Classes |
|--------|---------|----------|---------------|---------|
| Heart | 297 | 13 | target | 2 |
| Kidney | 397 | 24 | classification | 2 |
| Lung | 32 | 56 | class | 3 |
| Diabetes | 70 | 15 | diabetes_status | 2 |
| Thyroid | 7,129 | 81 | class | 3 |
| Liver | 341 | 100 | selector | 2 |
| Gallbladder | 5 | 77 | gallstone_status | 1 |
| Mental Health | 15 | 0 | stress_label | 1 |
| Survey | 2,278 | 11 | glucose_risk | 3 |

## 📈 Exploratory Data Analysis

Each module includes comprehensive EDA visualizations:

- **Correlation Heatmaps** - Feature relationships
- **Distribution Plots** - Feature value distributions
- **Box Plots** - Outlier detection
- **Target Analysis** - Class distribution and balance
- **Feature Importance** - Key predictive features

All visualizations are saved in `reports/{module}/eda/` directories.

## 🤝 Team Workflow

### Git Workflow

```bash
# Pull latest changes
git pull origin main

# Create feature branch
git checkout -b feature/your-feature

# Make changes and commit
git add .
git commit -m "Description of changes"

# Push to remote
git push origin feature/your-feature

# Create pull request on GitHub
```

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to functions
- Comment complex logic
- Use meaningful variable names
- Test code before committing

## 📝 Module Summary Files

Each module has a `MODULE_SUMMARY.txt` in its reports folder containing:

- Dataset source and description
- Data cleaning steps performed
- EDA key findings
- Feature engineering summary
- Train/test split information
- Recommended ML algorithms for the specific condition

## 🔍 Data Quality

All datasets have undergone:

- Missing value imputation
- Duplicate removal
- Outlier detection and handling
- Feature scaling and normalization
- Stratified train/test splitting
- Data validation and integrity checks

## 📚 Documentation

- Each dataset folder may contain additional README files
- Processing scripts include inline documentation
- Module summaries provide detailed analysis reports
- EDA plots are labeled and self-explanatory

## 🎓 Academic Context

This project is developed as part of a university coursework focusing on:
- Machine learning applications in healthcare
- Data preprocessing and feature engineering
- Predictive modeling for disease detection
- Real-world data analysis and visualization

## 📧 Contact

For questions or collaboration opportunities, please reach out through the project repository.

## 📄 License

This project is developed for educational purposes.

---

**Note**: This system is designed for research and educational purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
