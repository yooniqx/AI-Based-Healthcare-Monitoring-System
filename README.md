# Healthcare Monitoring and Early Disease Prediction System

A comprehensive Python-based system for healthcare monitoring and early disease prediction across multiple health conditions. This project implements a complete machine learning pipeline from data collection to trained models ready for deployment.

## 🎯 Project Overview

This system enables early detection of various health conditions through data-driven analysis and machine learning. The project currently supports prediction models for nine different health conditions.

## 📸 Application Preview

Experience our intuitive healthcare monitoring interface designed for seamless user interaction and accurate disease prediction.

### 🌐 Live Demo

**[https://ai-based-healthcare-monitoring-system.onrender.com](https://ai-based-healthcare-monitoring-system.onrender.com)**

### Homepage & Dashboard
![Homepage](assests/homepage.jpeg)
*Clean, user-friendly interface providing easy access to all health monitoring modules and features*

### AI Health Assistant Chatbot
![Chatbot Interface](assests/chatbot.jpeg)
*Intelligent conversational AI assistant for symptom analysis and health guidance*

### Heart Disease Prediction Module
![Heart Disease Module](assests/heart-model.jpeg)
*Advanced cardiovascular risk assessment with real-time predictions using trained ML models*

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
- **Streamlit** - Interactive web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-Learn** - Machine learning and preprocessing
- **Matplotlib/Seaborn** - Data visualization
- **SQLite** - Local database for health records

## 🚀 Running the Application

### Prerequisites
```bash
pip install streamlit pandas numpy scikit-learn
```

### Launch the Web Application
```bash
cd development
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Features
- **9 Disease Prediction Modules**: Heart, Kidney, Diabetes, Lung, Thyroid, Liver, Survey, Gallbladder, Mental Health
- **ML-Based Predictions**: 7 trained machine learning models with high accuracy
- **Rule-Based Assessments**: Gallbladder and Mental Health screening engines
- **Interactive UI**: User-friendly forms for patient data entry
- **Real-time Results**: Instant risk assessment with recommendations
- **Health Chatbot**: General health guidance and symptom routing
- **Medication Reminders**: Track daily medications
- **Vitals Monitoring**: Monitor heart rate, SpO2, and temperature
- **Emergency SOS**: Simulated emergency alert system

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
├── models/                            # Trained ML models
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── kidney_model.pkl
│   ├── kidney_scaler.pkl
│   └── ...
│
├── reports/                           # Analysis reports and visualizations
│   ├── heart/
│   │   ├── MODULE_SUMMARY.txt
│   │   ├── MODEL_SUMMARY.txt
│   │   └── eda/                       # Exploratory data analysis plots
│   │
│   ├── kidney/
│   ├── lung/
│   ├── diabetes/
│   ├── thyroid/
│   ├── liver/
│   ├── gallbladder/
│   ├── mental_health/
│   ├── survey/
│   └── MODEL_TRAINING_SUMMARY.txt
│
├── scripts/                           # Data processing scripts
│   ├── preprocessing/                 # Data cleaning scripts
│   ├── eda/                          # Exploratory analysis scripts
│   ├── feature_engineering/          # Feature creation scripts
│   ├── model_training/               # Model training scripts
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
7. **Model Selection** - Evaluated Logistic Regression, Random Forest, and SVM
8. **Model Training** - Trained and evaluated models for all 9 modules
9. **Model Validation** - Tested and validated all prediction systems
10. **Hyperparameter Tuning** - Optimized model parameters using GridSearchCV for all 7 ML modules
11. **Streamlit Frontend** - Web-based prediction interface with 14 pages
12. **Module Integration** - Complete system integration via model_integration.py
13. **Healthcare Recommendations** - Personalised advice and specialist routing per prediction
14. **Chatbot + SOS Integration** - Symptom chatbot, emergency SOS alert, and vitals monitor

### 🎉 Project Complete!

## 📊 Model Training Status

### Machine Learning Models (7/9) ✅

| Module | Best Model | Accuracy | F1 Score | ROC-AUC | Status |
|--------|-----------|----------|----------|---------|--------|
| **Heart** | Random Forest | 86.67% | 86.62% | 94.14% | ✅ Tuned |
| **Kidney** | Logistic Regression | 98.75% | 98.75% | 99.87% | ✅ Tuned |
| **Lung** | SVM | 57.14% | 47.62% | - | ✅ Tuned* |
| **Diabetes** | Logistic Regression | 92.86% | 93.36% | - | ✅ Tuned |
| **Thyroid** | Random Forest | 98.88% | 98.90% | - | ✅ Tuned |
| **Liver** | Random Forest | 72.46% | 71.97% | 77.59% | ✅ Tuned |
| **Survey** | Random Forest | 100.00% | 100.00% | - | ✅ Tuned |

*Limited training data affects performance

**All models have been optimized using GridSearchCV hyperparameter tuning!**

### Rule-Based Screening Engines (2/9) ✅

| Module | Approach | Risk Levels | Validation | Status |
|--------|----------|-------------|------------|--------|
| **Gallbladder** | Symptom-based scoring | Low/Moderate/High | ✅ Tested | ✅ Ready |
| **Mental Health** | Questionnaire (10Q) | Low/Mild/Moderate/High | ✅ Tested | ✅ Ready |

**All 9 modules are now ready for production deployment!**

## 🎯 Hyperparameter Tuning

All 7 machine learning models have undergone comprehensive hyperparameter optimization using GridSearchCV with 5-fold cross-validation. The tuning process optimized the following parameters for each model type:

### Optimization Details

**Logistic Regression** (Diabetes, Kidney):
- Regularization strength (C): [0.001, 0.01, 0.1, 1, 10, 100]
- Penalty type: ['l1', 'l2']
- Solver: ['liblinear', 'saga']
- Max iterations: [100, 200, 500]

**Random Forest** (Heart, Thyroid, Liver, Survey):
- Number of estimators: [50, 100, 200, 300]
- Max depth: [None, 10, 20, 30]
- Min samples split: [2, 5, 10]
- Min samples leaf: [1, 2, 4]
- Max features: ['sqrt', 'log2']

**Support Vector Machine** (Lung):
- Kernel type: ['linear', 'rbf', 'poly']
- Regularization (C): [0.1, 1, 10, 100]
- Gamma: ['scale', 'auto', 0.001, 0.01, 0.1]
- Degree (for poly): [2, 3, 4]

### Tuned Model Files

Each module now has two sets of models:
- **Base Models**: `{module}_model.pkl` and `{module}_scaler.pkl`
- **Tuned Models**: `{module}_model_tuned.pkl` and `{module}_scaler_tuned.pkl`

The tuned models are recommended for production use as they provide optimized performance through systematic hyperparameter search.

### Running Hyperparameter Tuning

```bash
# Run comprehensive hyperparameter tuning for all modules
python scripts/model_training/hyperparameter_tuning.py

# Quick tuning with reduced parameter grid (faster)
python scripts/model_training/quick_hyperparameter_tuning.py
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
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

- **Trained Models**: `models/`
  - ML Models: `{module}_model.pkl` - Trained classifier
  - ML Models: `{module}_scaler.pkl` - Feature scaler
  - Rule-Based: `{module}_screening_engine.py` - Screening engine

### Usage

#### 1. Load and Use ML Models

```python
import pandas as pd
import pickle

# Load tuned model and scaler (recommended for production)
with open('models/heart_model_tuned.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/heart_scaler_tuned.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare input data
X_new = pd.DataFrame([...])  # User input features

# Scale and predict
X_scaled = scaler.transform(X_new)
prediction = model.predict(X_scaled)
probability = model.predict_proba(X_scaled)

print(f"Prediction: {prediction[0]}")
print(f"Probability: {probability[0]}")
```

**Note**: Use `{module}_model_tuned.pkl` for optimized performance. Base models (`{module}_model.pkl`) are also available.

#### 1b. Use Rule-Based Screening Engines

```python
from models.gallbladder_screening_engine import GallbladderScreeningEngine

# Initialize engine
engine = GallbladderScreeningEngine()

# Assess risk
symptoms = {
    'abdominal_pain': True,
    'nausea_vomiting': True,
    'fever': False,
    # ... other symptoms
}
result = engine.assess_risk(symptoms)

print(f"Risk Score: {result['risk_score']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}")
```

#### 2. Explore Datasets

```python
import pandas as pd

# Load master dataset
df = pd.read_csv('datasets/heart/heart_master_dataset.csv')
print(df.head())
```

#### 3. Load ML-Ready Data

```python
# Load train/test splits
X_train = pd.read_csv('datasets/heart/ml_ready/X_train.csv')
X_test = pd.read_csv('datasets/heart/ml_ready/X_test.csv')
y_train = pd.read_csv('datasets/heart/ml_ready/y_train.csv')
y_test = pd.read_csv('datasets/heart/ml_ready/y_test.csv')
```

#### 4. View Analysis Reports

Check the `reports/{module}/MODEL_SUMMARY.txt` for:
- Models trained and evaluated
- Performance metrics
- Best model selection criteria
- Model file locations
- Usage instructions

#### 5. Run Processing Scripts

```bash
# Data preprocessing
python scripts/preprocessing/data_cleaning.py

# Exploratory analysis
python scripts/eda/exploratory_analysis.py

# Feature engineering
python scripts/feature_engineering/feature_engineering.py

# Generate ML-ready splits
python scripts/utilities/create_ml_ready_splits.py

# Train base models
python scripts/model_training/train_models_robust.py

# Hyperparameter tuning (comprehensive)
python scripts/model_training/hyperparameter_tuning.py

# Quick hyperparameter tuning (faster, reduced grid)
python scripts/model_training/quick_hyperparameter_tuning.py
```

## 📈 Exploratory Data Analysis

Each module includes comprehensive EDA visualizations:

- **Correlation Heatmaps** - Feature relationships
- **Distribution Plots** - Feature value distributions
- **Box Plots** - Outlier detection
- **Target Analysis** - Class distribution and balance
- **Feature Importance** - Key predictive features

All visualizations are saved in `reports/{module}/eda/` directories.

## 🤝 Current Team Instructions

### Before Starting Work

1. **Pull Latest Changes**
   ```bash
   git pull origin main
   ```

2. **Do Not Create Duplicate Datasets**
   - Use existing datasets in `datasets/` folder
   - Do not regenerate ml_ready splits unless necessary

3. **Do Not Create Extra Reports**
   - Follow existing report structure
   - Update only `MODULE_SUMMARY.txt` and `MODEL_SUMMARY.txt`

4. **Follow Existing Folder Structure**
   - Place files in appropriate directories
   - Maintain naming conventions

5. **Update Only Assigned Module**
   - Work on your assigned disease module
   - Do not modify other modules without coordination

6. **Do Not Modify Trained Models Without Discussion**
   - Trained models in `models/` are production-ready
   - Discuss any changes with the team first

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

Each module has comprehensive documentation:

- `MODULE_SUMMARY.txt` - Data processing and EDA summary
- `MODEL_SUMMARY.txt` - Model training results and metrics
- `MODEL_TRAINING_SUMMARY.txt` - Overall project training status

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
- Model summaries include usage examples

## 🎓 Academic Context

This project is developed as part of a university coursework focusing on:
- Machine learning applications in healthcare
- Data preprocessing and feature engineering
- Predictive modeling for disease detection
- Real-world data analysis and visualization
- Model evaluation and deployment

## 📧 Contact

For questions or collaboration opportunities, please reach out through the project repository.

## 📄 License

**Copyright © 2026 AI-Based-Healthcare-Project-Team. All Rights Reserved.**

This repository and all its contents are proprietary and confidential. No permission is granted to copy, reproduce, distribute, modify, publish, sublicense, sell, or commercially use any part of this repository without prior written permission from the AI-Based-Healthcare-Project-Team organization.

**Prohibited Actions:**
- Unauthorized copying, reproduction, or distribution of code, datasets, documentation, trained models, reports, or any project assets
- Modification or creation of derivative works
- Commercial use or exploitation
- Public sharing or publishing of repository contents

**Intended Use:** This repository is for academic and project-development purposes by authorized organization members only.

For permissions or inquiries, contact the AI-Based-Healthcare-Project-Team organization administrators.

See the [LICENSE](LICENSE) file for complete terms and conditions.

---

**Note**: This system is designed for research and educational purposes. It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
