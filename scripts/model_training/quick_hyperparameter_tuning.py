"""
Quick Hyperparameter Tuning Script - Faster version using RandomizedSearchCV
Optimizes model performance with fewer iterations for faster completion
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(module_name):
    """Load and preprocess data"""
    base_path = Path(f'datasets/{module_name}/ml_ready')
    
    X_train = pd.read_csv(base_path / 'X_train.csv')
    X_test = pd.read_csv(base_path / 'X_test.csv')
    y_train = pd.read_csv(base_path / 'y_train.csv').values.ravel()
    y_test = pd.read_csv(base_path / 'y_test.csv').values.ravel()
    
    # Remove problematic columns
    for col in ['dataset_split', 'Unnamed: 0']:
        if col in X_train.columns:
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])
    
    # Handle categorical columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # Handle missing values
    if X_train.isnull().any().any():
        imputer = SimpleImputer(strategy='median')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_param_distributions():
    """Define parameter distributions for RandomizedSearchCV"""
    return {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000, 2000]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf', 'linear']
        }
    }


def tune_model(model_name, model, param_dist, X_train, y_train):
    """Perform randomized search"""
    print(f"\n{'='*70}")
    print(f"Tuning {model_name}...")
    print(f"{'='*70}")
    
    f1_scorer = make_scorer(f1_score, average='weighted')
    
    # Use RandomizedSearchCV with fewer iterations
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=10,  # Only 10 iterations instead of testing all combinations
        cv=3,  # 3-fold instead of 5-fold for speed
        scoring=f1_scorer,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    print(f"Testing 10 random parameter combinations with 3-fold CV...")
    random_search.fit(X_train, y_train)
    
    print(f"\n[+] Best parameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"\n[+] Best CV F1 score: {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'roc_auc': None
    }
    
    if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except:
            pass
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if metrics['roc_auc']:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def save_tuned_model(model, scaler, module_name):
    """Save tuned model"""
    models_dir = Path('models')
    model_path = models_dir / f'{module_name}_model_tuned.pkl'
    scaler_path = models_dir / f'{module_name}_scaler_tuned.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\n[+] Saved: {model_path}")
    print(f"[+] Saved: {scaler_path}")
    
    return model_path, scaler_path


def load_baseline_metrics(module_name):
    """Load baseline metrics"""
    registry_path = Path('models/model_registry.json')
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            if module_name in registry['models']:
                return registry['models'][module_name].get('metrics', {})
    return {}


def compare_performance(tuned_metrics, baseline_metrics, module_name):
    """Compare tuned vs baseline"""
    print(f"\n{'='*70}")
    print(f"COMPARISON: {module_name.upper()}")
    print(f"{'='*70}")
    
    improvements = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        if metric in baseline_metrics and baseline_metrics[metric] is not None:
            baseline = baseline_metrics[metric]
            tuned = tuned_metrics[metric]
            if tuned is not None:
                improvement = ((tuned - baseline) / baseline) * 100
                improvements[metric] = improvement
                symbol = "+" if improvement > 0 else ""
                print(f"{metric:12} | Baseline: {baseline:.4f} | Tuned: {tuned:.4f} | {symbol}{improvement:.2f}%")
    
    return improvements


def tune_module(module_name, model_type):
    """Complete tuning pipeline"""
    try:
        print(f"\n{'#'*70}")
        print(f"# {module_name.upper()} - {model_type}")
        print(f"{'#'*70}")
        
        # Load data
        X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(module_name)
        print(f"Data loaded: {len(X_train)} train, {len(X_test)} test samples")
        
        # Get parameters
        param_dists = get_param_distributions()
        
        # Initialize model
        if model_type == 'Logistic Regression':
            model = LogisticRegression(random_state=42)
        elif model_type == 'Random Forest':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_type == 'SVM':
            model = SVC(probability=True, random_state=42)
        
        # Tune
        best_model, best_params, cv_score = tune_model(
            model_type, model, param_dists[model_type], X_train, y_train
        )
        
        # Evaluate
        tuned_metrics = evaluate_model(best_model, X_test, y_test)
        
        # Compare
        baseline_metrics = load_baseline_metrics(module_name)
        improvements = compare_performance(tuned_metrics, baseline_metrics, module_name)
        
        # Save
        save_tuned_model(best_model, scaler, module_name)
        
        print(f"\n[+] {module_name.upper()} TUNING COMPLETE\n")
        return True, improvements
        
    except Exception as e:
        print(f"\n[ERROR] {module_name}: {str(e)}\n")
        return False, {}


def main():
    """Main function"""
    print("\n" + "="*70)
    print("QUICK HYPERPARAMETER TUNING - HEALTHCARE PREDICTION SYSTEM")
    print("="*70)
    print("Method: RandomizedSearchCV (10 iterations, 3-fold CV)")
    print("="*70)
    
    modules_to_tune = {
        'heart': 'Random Forest',
        'kidney': 'Logistic Regression',
        'lung': 'SVM',
        'diabetes': 'Logistic Regression',
        'thyroid': 'Random Forest',
        'liver': 'Random Forest',
        'survey': 'Random Forest'
    }
    
    successful = []
    failed = []
    all_improvements = {}
    
    for module, model_type in modules_to_tune.items():
        success, improvements = tune_module(module, model_type)
        if success:
            successful.append(module)
            all_improvements[module] = improvements
        else:
            failed.append(module)
    
    # Final summary
    print("\n" + "="*70)
    print("TUNING COMPLETE - SUMMARY")
    print("="*70)
    print(f"\nSuccessful: {len(successful)}/{len(modules_to_tune)}")
    if successful:
        print("  [+] " + ", ".join(successful))
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(modules_to_tune)}")
        print("  [-] " + ", ".join(failed))
    
    if all_improvements:
        print("\n" + "="*70)
        print("IMPROVEMENT SUMMARY")
        print("="*70)
        for module, improvements in all_improvements.items():
            if improvements:
                avg_imp = np.mean([v for v in improvements.values()])
                print(f"{module:15} | Avg Improvement: {avg_imp:+.2f}%")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

# Made with Bob
