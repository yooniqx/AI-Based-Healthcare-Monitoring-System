import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def create_ml_ready(dataset_path, output_dir, target_column, test_size=0.2, random_state=42, scale_features=True):
    """
    Create ML-ready train/test splits from a master dataset.
    
    Parameters:
    - dataset_path: Path to the master dataset CSV
    - output_dir: Directory to save ml_ready files
    - target_column: Name of the target column
    - test_size: Proportion of test set (default 0.2)
    - random_state: Random seed for reproducibility (default 42)
    - scale_features: Whether to apply StandardScaler (default True)
    """
    print(f"\nProcessing: {dataset_path}")
    
    # Load dataset
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {target_column}")
    
    # Separate features and target
    if target_column not in df.columns:
        print(f"ERROR: Target column '{target_column}' not found in dataset!")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Check if target is classification or regression
    is_classification = y.dtype == 'object' or y.nunique() < 20
    stratify = y if is_classification else None
    
    print(f"Task type: {'Classification' if is_classification else 'Regression'}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Target distribution:\n{y.value_counts() if is_classification else y.describe()}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale features if requested
    if scale_features and X_train.select_dtypes(include=[np.number]).shape[1] > 0:
        scaler = StandardScaler()
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
        print(f"Scaled {len(numeric_cols)} numeric features")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    print(f"[OK] ML-ready files saved to: {output_dir}")
    return True

# Module configurations
modules = [
    {
        'name': 'lung',
        'dataset': 'datasets/lung/lung_master_dataset.csv',
        'output': 'datasets/lung/ml_ready',
        'target': 'class',
        'scale': True
    },
    {
        'name': 'thyroid',
        'dataset': 'datasets/thyroid/thyroid_master_dataset.csv',
        'output': 'datasets/thyroid/ml_ready',
        'target': 'class',
        'scale': True
    },
    {
        'name': 'liver',
        'dataset': 'datasets/liver/liver_master_dataset.csv',
        'output': 'datasets/liver/ml_ready',
        'target': 'selector',
        'scale': True
    },
    {
        'name': 'gallbladder',
        'dataset': 'datasets/gallbladder/gallbladder_master_dataset.csv',
        'output': 'datasets/gallbladder/ml_ready',
        'target': 'gallstone_status',
        'scale': True
    },
    {
        'name': 'mental_health',
        'dataset': 'datasets/mental_health/mental_health_master_dataset.csv',
        'output': 'datasets/mental_health/ml_ready',
        'target': 'stress_label',
        'scale': True
    },
    {
        'name': 'survey',
        'dataset': 'datasets/survey/survey_master_dataset.csv',
        'output': 'datasets/survey/ml_ready',
        'target': 'glucose_risk',
        'scale': True
    }
]

print("="*70)
print("Creating ML-Ready Train/Test Splits")
print("="*70)

success_count = 0
failed_modules = []

for module in modules:
    try:
        success = create_ml_ready(
            dataset_path=module['dataset'],
            output_dir=module['output'],
            target_column=module['target'],
            scale_features=module['scale']
        )
        if success:
            success_count += 1
        else:
            failed_modules.append(module['name'])
    except Exception as e:
        print(f"ERROR processing {module['name']}: {str(e)}")
        failed_modules.append(module['name'])

print("\n" + "="*70)
print(f"Summary: {success_count}/{len(modules)} modules processed successfully")
if failed_modules:
    print(f"Failed modules: {', '.join(failed_modules)}")
print("="*70)


