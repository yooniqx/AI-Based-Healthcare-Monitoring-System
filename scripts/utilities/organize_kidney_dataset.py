import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

# Create directory structure
directories = [
    'datasets/kidney/ml_ready',
    'reports/kidney/eda',
    'scripts'
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Load the kidney master dataset
print("\nLoading kidney_master_dataset.csv...")
df = pd.read_csv('datasets/kidney/kidney_master_dataset.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Separate features (X) and target (y)
X = df.drop('ckd_status', axis=1)
y = df['ckd_status']

# Split into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Save the split datasets
print("\nSaving train/test splits...")
X_train.to_csv('datasets/kidney/ml_ready/X_train.csv', index=False)
X_test.to_csv('datasets/kidney/ml_ready/X_test.csv', index=False)
y_train.to_csv('datasets/kidney/ml_ready/y_train.csv', index=False)
y_test.to_csv('datasets/kidney/ml_ready/y_test.csv', index=False)

print("[OK] X_train.csv saved")
print("[OK] X_test.csv saved")
print("[OK] y_train.csv saved")
print("[OK] y_test.csv saved")

# Generate EDA visualizations
print("\nGenerating EDA visualizations...")

# 1. Correlation Heatmap
plt.figure(figsize=(16, 12))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap - Chronic Kidney Disease Dataset', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('reports/kidney/eda/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] correlation_heatmap.png saved")

# 2. Feature Correlation with Target
plt.figure(figsize=(12, 8))
target_corr = correlation_matrix['ckd_status'].sort_values(ascending=False)
target_corr = target_corr[target_corr.index != 'ckd_status']
colors = ['green' if x > 0 else 'red' for x in target_corr.values]
plt.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
plt.yticks(range(len(target_corr)), target_corr.index)
plt.xlabel('Correlation with CKD Status', fontsize=12)
plt.title('Feature Correlation with Target (CKD Status)', fontsize=14, pad=20)
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('reports/kidney/eda/feature_correlation_with_target.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] feature_correlation_with_target.png saved")

# 3. Feature Distributions
numeric_features = X.select_dtypes(include=[np.number]).columns[:12]  # First 12 numeric features
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.ravel()

for idx, col in enumerate(numeric_features):
    axes[idx].hist(df[col].dropna(), bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[idx].set_xlabel('Value', fontsize=9)
    axes[idx].set_ylabel('Frequency', fontsize=9)
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Feature Distributions - Chronic Kidney Disease Dataset', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig('reports/kidney/eda/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] feature_distributions.png saved")

# 4. Outlier Analysis (Box plots)
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
axes = axes.ravel()

for idx, col in enumerate(numeric_features):
    axes[idx].boxplot(df[col].dropna(), vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    axes[idx].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[idx].set_ylabel('Value', fontsize=9)
    axes[idx].grid(axis='y', alpha=0.3)

plt.suptitle('Outlier Analysis - Chronic Kidney Disease Dataset', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig('reports/kidney/eda/outlier_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] outlier_analysis.png saved")

# 5. Target Distribution
plt.figure(figsize=(10, 6))
target_counts = y.value_counts()
colors_target = ['#2ecc71', '#e74c3c']
plt.bar(target_counts.index, target_counts.values, color=colors_target, alpha=0.7, edgecolor='black')
plt.xlabel('CKD Status (0=No, 1=Yes)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Target Distribution - CKD Status', fontsize=14, pad=20)
plt.xticks([0, 1], ['No CKD', 'CKD'])
for i, v in enumerate(target_counts.values):
    plt.text(target_counts.index[i], v + 5, str(v), ha='center', fontsize=12, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('reports/kidney/eda/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("[OK] target_distribution.png saved")

# Create KIDNEY_MODULE_SUMMARY
summary_content = f"""# CHRONIC KIDNEY DISEASE MODULE SUMMARY

## Dataset Overview
- **Total Samples**: {len(df)}
- **Total Features**: {len(df.columns) - 1}
- **Target Variable**: ckd_status (0=No CKD, 1=CKD)

## Data Split
- **Training Set**: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)
- **Test Set**: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)

## Target Distribution
- **No CKD (0)**: {(y == 0).sum()} samples ({(y == 0).sum()/len(y)*100:.1f}%)
- **CKD (1)**: {(y == 1).sum()} samples ({(y == 1).sum()/len(y)*100:.1f}%)

## Features ({len(X.columns)})
{chr(10).join([f"- {col}" for col in X.columns])}

## Directory Structure
```
datasets/kidney/
â”œâ”€â”€ kidney_master_dataset.csv
â””â”€â”€ ml_ready/
    â”œâ”€â”€ X_train.csv
    â”œâ”€â”€ X_test.csv
    â”œâ”€â”€ y_train.csv
    â””â”€â”€ y_test.csv

reports/kidney/
â”œâ”€â”€ kidney_DATASET_SUMMARY.txt
â””â”€â”€ eda/
    â”œâ”€â”€ correlation_heatmap.png
    â”œâ”€â”€ feature_correlation_with_target.png
    â”œâ”€â”€ feature_distributions.png
    â”œâ”€â”€ outlier_analysis.png
    â””â”€â”€ target_distribution.png

scripts/
â”œâ”€â”€ feature_engineering.py
â””â”€â”€ perform_eda.py
```

## Data Quality
- **Missing Values**: {df.isnull().sum().sum()} total
- **Duplicate Rows**: {df.duplicated().sum()}

## Top Correlated Features with Target
{chr(10).join([f"{i+1}. {feat}: {corr:.3f}" for i, (feat, corr) in enumerate(target_corr.head(10).items())])}

## Generated Files
- Train/Test splits saved in: datasets/kidney/ml_ready/
- EDA visualizations saved in: reports/kidney/eda/
- Summary document: reports/kidney/KIDNEY_MODULE_SUMMARY.txt

---
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

with open('reports/kidney/KIDNEY_MODULE_SUMMARY.txt', 'w', encoding='utf-8') as f:
    f.write(summary_content)

print("\n[OK] KIDNEY_MODULE_SUMMARY.txt saved")

print("\n" + "="*60)
print("ORGANIZATION COMPLETE!")
print("="*60)
print("\nAll files have been organized according to the structure:")
print("[OK] ML-ready datasets created")
print("[OK] EDA visualizations generated")
print("[OK] Summary document created")
print("\nYou can now proceed with feature engineering and model training.")


