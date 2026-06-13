"""
Feature Engineering Script for Chronic Kidney Disease Dataset
This script handles feature engineering tasks including:
- Feature scaling and normalization
- Feature selection
- Creating new features
- Handling categorical variables
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the train and test datasets"""
    X_train = pd.read_csv('datasets/kidney/ml_ready/X_train.csv')
    X_test = pd.read_csv('datasets/kidney/ml_ready/X_test.csv')
    y_train = pd.read_csv('datasets/kidney/ml_ready/y_train.csv')
    y_test = pd.read_csv('datasets/kidney/ml_ready/y_test.csv')
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, method='standard'):
    """
    Scale numerical features
    
    Parameters:
    - method: 'standard' for StandardScaler or 'minmax' for MinMaxScaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler

def select_features(X_train, y_train, k=15, method='f_classif'):
    """
    Select top k features based on statistical tests
    
    Parameters:
    - k: number of features to select
    - method: 'f_classif' or 'mutual_info'
    """
    if method == 'f_classif':
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    
    selector.fit(X_train, y_train.values.ravel())
    
    # Get selected feature names
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # Get feature scores
    scores = pd.DataFrame({
        'feature': X_train.columns,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    return selected_features, scores, selector

def create_interaction_features(X):
    """Create interaction features between important variables"""
    X_new = X.copy()
    
    # Example interactions (customize based on domain knowledge)
    if 'blood_urea' in X.columns and 'serum_creatinine' in X.columns:
        X_new['urea_creatinine_ratio'] = X['blood_urea'] / (X['serum_creatinine'] + 1e-6)
    
    if 'hemoglobin' in X.columns and 'packed_cell_volume' in X.columns:
        X_new['hb_pcv_ratio'] = X['hemoglobin'] / (X['packed_cell_volume'] + 1e-6)
    
    return X_new

if __name__ == "__main__":
    print("Feature Engineering for Chronic Kidney Disease Dataset")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Scale features
    print("\nScaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("Features scaled successfully")
    
    # Feature selection
    print("\nSelecting top features...")
    selected_features, feature_scores, selector = select_features(X_train, y_train, k=15)
    print(f"\nTop 15 features selected:")
    for i, feat in enumerate(selected_features, 1):
        print(f"{i}. {feat}")
    
    print("\nFeature engineering complete!")


