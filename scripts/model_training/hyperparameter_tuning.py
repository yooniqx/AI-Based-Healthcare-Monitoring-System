"""
Hyperparameter Tuning Script for Healthcare Prediction System
Optimizes model performance through systematic parameter search
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, make_scorer
)
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """Hyperparameter tuning for healthcare prediction models"""
    
    def __init__(self, module_name):
        self.module_name = module_name
        self.base_path = Path(f'datasets/{module_name}/ml_ready')
        self.models_path = Path('models')
        self.reports_path = Path('reports') / module_name
        
    def load_data(self):
        """Load and preprocess data"""
        print(f"\n{'='*70}")
        print(f"Loading data for {self.module_name.upper()} module...")
        print(f"{'='*70}")
        
        X_train = pd.read_csv(self.base_path / 'X_train.csv')
        X_test = pd.read_csv(self.base_path / 'X_test.csv')
        y_train = pd.read_csv(self.base_path / 'y_train.csv').values.ravel()
        y_test = pd.read_csv(self.base_path / 'y_test.csv').values.ravel()
        
        # Remove problematic columns
        cols_to_drop = ['dataset_split', 'Unnamed: 0']
        for col in cols_to_drop:
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
            X_train = pd.DataFrame(
                imputer.fit_transform(X_train),
                columns=X_train.columns
            )
            X_test = pd.DataFrame(
                imputer.transform(X_test),
                columns=X_test.columns
            )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Features: {X_train.shape[1]}")
        print(f"Classes: {len(np.unique(y_train))}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns
    
    def get_param_grids(self):
        """Define parameter grids for each model type"""
        param_grids = {
            'Logistic Regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [1000, 2000]
            },
            'Random Forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        }
        return param_grids
    
    def tune_model(self, model_name, model, param_grid, X_train, y_train):
        """Perform grid search for a specific model"""
        print(f"\n{'='*70}")
        print(f"Tuning {model_name}...")
        print(f"{'='*70}")
        print(f"Parameter grid: {param_grid}")
        print(f"Total combinations: {np.prod([len(v) for v in param_grid.values()])}")
        
        # Custom scorer prioritizing F1 and Recall
        f1_scorer = make_scorer(f1_score, average='weighted')
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring=f1_scorer,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
        
        print("\nPerforming grid search with 5-fold cross-validation...")
        grid_search.fit(X_train, y_train)
        
        print(f"\n[+] Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        
        print(f"\n[+] Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate tuned model on test set"""
        print(f"\nEvaluating {model_name} on test set...")
        
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC for binary classification
        roc_auc = None
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
            except:
                pass
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        return metrics
    
    def load_baseline_metrics(self):
        """Load baseline model metrics from registry"""
        registry_path = Path('models/model_registry.json')
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry = json.load(f)
                if self.module_name in registry['models']:
                    return registry['models'][self.module_name].get('metrics', {})
        return {}
    
    def compare_with_baseline(self, tuned_metrics, baseline_metrics):
        """Compare tuned model with baseline"""
        print(f"\n{'='*70}")
        print("COMPARISON: Tuned vs Baseline")
        print(f"{'='*70}")
        
        improvements = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
            if metric in baseline_metrics and baseline_metrics[metric] is not None:
                baseline = baseline_metrics[metric]
                tuned = tuned_metrics[metric]
                if tuned is not None:
                    improvement = ((tuned - baseline) / baseline) * 100
                    improvements[metric] = improvement
                    
                    if improvement > 0:
                        symbol = "+"
                    elif improvement < 0:
                        symbol = "-"
                    else:
                        symbol = "="
                    print(f"{metric.upper():12} | Baseline: {baseline:.4f} | Tuned: {tuned:.4f} | {symbol} {abs(improvement):.2f}%")
        
        return improvements
    
    def save_tuned_model(self, model, scaler, best_params, metrics):
        """Save tuned model and update registry"""
        # Save model
        model_path = self.models_path / f'{self.module_name}_model_tuned.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save scaler
        scaler_path = self.models_path / f'{self.module_name}_scaler_tuned.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        print(f"\n[+] Tuned model saved: {model_path}")
        print(f"[+] Scaler saved: {scaler_path}")
        
        return model_path, scaler_path
    
    def generate_tuning_report(self, model_name, best_params, tuned_metrics, 
                              baseline_metrics, improvements, cv_score):
        """Generate comprehensive tuning report"""
        report_path = self.reports_path / 'HYPERPARAMETER_TUNING_REPORT.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"HYPERPARAMETER TUNING REPORT - {self.module_name.upper()}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Model Type: {model_name}\n")
            f.write(f"Tuning Method: GridSearchCV with 5-fold Cross-Validation\n")
            f.write(f"Scoring Metric: F1 Score (weighted)\n\n")
            
            f.write("="*70 + "\n")
            f.write("BEST HYPERPARAMETERS\n")
            f.write("="*70 + "\n\n")
            for param, value in best_params.items():
                f.write(f"  {param}: {value}\n")
            
            f.write(f"\nBest Cross-Validation F1 Score: {cv_score:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("PERFORMANCE COMPARISON\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"{'Metric':<15} {'Baseline':<12} {'Tuned':<12} {'Change':<12}\n")
            f.write("-"*70 + "\n")
            
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                if metric in baseline_metrics and baseline_metrics[metric] is not None:
                    baseline = baseline_metrics[metric]
                    tuned = tuned_metrics[metric]
                    if tuned is not None:
                        change = improvements.get(metric, 0)
                        symbol = "+" if change > 0 else "-" if change < 0 else "="
                        f.write(f"{metric:<15} {baseline:<12.4f} {tuned:<12.4f} {symbol} {abs(change):.2f}%\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("TUNED MODEL METRICS\n")
            f.write("="*70 + "\n\n")
            
            for metric, value in tuned_metrics.items():
                if value is not None:
                    f.write(f"  {metric}: {value:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("="*70 + "\n\n")
            
            avg_improvement = np.mean([v for v in improvements.values() if v is not None])
            
            if avg_improvement > 2:
                f.write("[+] SIGNIFICANT IMPROVEMENT: Tuned model shows substantial gains.\n")
                f.write("  Recommendation: Deploy tuned model to production.\n")
            elif avg_improvement > 0:
                f.write("[+] MODERATE IMPROVEMENT: Tuned model shows some gains.\n")
                f.write("  Recommendation: Consider deploying tuned model.\n")
            else:
                f.write("[!] MINIMAL/NO IMPROVEMENT: Baseline model performs similarly.\n")
                f.write("  Recommendation: Keep baseline model or explore other techniques.\n")
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"\n[+] Tuning report saved: {report_path}")
        return report_path


def tune_module(module_name, model_type):
    """Complete tuning pipeline for a module"""
    try:
        tuner = HyperparameterTuner(module_name)
        
        # Load data
        X_train, X_test, y_train, y_test, scaler, feature_names = tuner.load_data()
        
        # Get parameter grids
        param_grids = tuner.get_param_grids()
        
        # Initialize model
        if model_type == 'Logistic Regression':
            model = LogisticRegression(random_state=42)
        elif model_type == 'Random Forest':
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
        elif model_type == 'SVM':
            model = SVC(probability=True, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Tune model
        best_model, best_params, cv_score = tuner.tune_model(
            model_type, model, param_grids[model_type], X_train, y_train
        )
        
        # Evaluate on test set
        tuned_metrics = tuner.evaluate_model(best_model, X_test, y_test, model_type)
        
        # Load baseline metrics
        baseline_metrics = tuner.load_baseline_metrics()
        
        # Compare with baseline
        improvements = tuner.compare_with_baseline(tuned_metrics, baseline_metrics)
        
        # Save tuned model
        model_path, scaler_path = tuner.save_tuned_model(
            best_model, scaler, best_params, tuned_metrics
        )
        
        # Generate report
        report_path = tuner.generate_tuning_report(
            model_type, best_params, tuned_metrics, 
            baseline_metrics, improvements, cv_score
        )
        
        print(f"\n{'='*70}")
        print(f"[+] {module_name.upper()} TUNING COMPLETE")
        print(f"{'='*70}\n")
        
        return True, improvements
        
    except Exception as e:
        print(f"\n[ERROR] Failed to tune {module_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, {}


def main():
    """Main tuning function"""
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING - HEALTHCARE PREDICTION SYSTEM")
    print("="*70)
    
    # Define modules and their best baseline models
    modules_to_tune = {
        'heart': 'Random Forest',
        'kidney': 'Logistic Regression',
        'lung': 'SVM',
        'diabetes': 'Logistic Regression',
        'thyroid': 'Random Forest',
        'liver': 'Random Forest',
        'survey': 'Random Forest'
    }
    
    print(f"\nModules to tune: {len(modules_to_tune)}")
    print("Method: GridSearchCV with 5-fold Cross-Validation")
    print("Scoring: F1 Score (weighted)\n")
    
    successful = []
    failed = []
    all_improvements = {}
    
    for module, model_type in modules_to_tune.items():
        print(f"\n{'#'*70}")
        print(f"# TUNING {module.upper()} MODULE ({model_type})")
        print(f"{'#'*70}")
        
        success, improvements = tune_module(module, model_type)
        
        if success:
            successful.append(module)
            all_improvements[module] = improvements
        else:
            failed.append(module)
    
    # Final summary
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING COMPLETE - FINAL SUMMARY")
    print("="*70)
    
    print(f"\nSuccessful: {len(successful)}/{len(modules_to_tune)} modules")
    if successful:
        print("  [+] " + ", ".join(successful))
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(modules_to_tune)} modules")
        print("  [-] " + ", ".join(failed))
    
    # Overall improvement summary
    if all_improvements:
        print("\n" + "="*70)
        print("OVERALL IMPROVEMENT SUMMARY")
        print("="*70 + "\n")
        
        for module, improvements in all_improvements.items():
            if improvements:
                avg_improvement = np.mean([v for v in improvements.values()])
                print(f"{module.upper():15} | Avg Improvement: {avg_improvement:+.2f}%")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()

# Made with Bob