"""
Lung Cancer Dataset Processing Pipeline
Self-contained script for processing lung cancer data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class LungCancerProcessor:
    """Complete processor for lung cancer dataset"""
    
    def __init__(self):
        self.base_dir = Path('datasets/lung_cancer')
        self.reports_dir = Path('reports/lung_cancer')
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / 'eda').mkdir(exist_ok=True)
        
        self.raw_data_path = self.base_dir / 'lung-cancer.data'
        self.master_dataset_path = self.base_dir / 'lung_cancer_master_dataset.csv'
        
    def load_raw_data(self):
        """Load and parse the raw lung cancer data"""
        print("=" * 80)
        print("LOADING RAW LUNG CANCER DATA")
        print("=" * 80)
        
        # Define column names
        columns = ['class'] + [f'feature_{i}' for i in range(1, 57)]
        
        # Load data
        df = pd.read_csv(self.raw_data_path, header=None, names=columns)
        df = df.replace('?', np.nan)
        
        # Convert to numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"\nDataset Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"Rows: {len(df)}")
        print(f"\nClass Distribution:")
        print(df['class'].value_counts().sort_index())
        print(f"\nMissing Values:")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values")
        
        # Save master dataset
        df.to_csv(self.master_dataset_path, index=False)
        print(f"\n[OK] Master dataset saved to: {self.master_dataset_path}")
        
        return df
    
    def clean_dataset(self, df):
        """Clean the lung cancer dataset"""
        print("\n" + "=" * 80)
        print("CLEANING LUNG CANCER DATASET")
        print("=" * 80)
        
        df_clean = df.copy()
        initial_shape = df_clean.shape
        
        # Handle missing values
        print("\n1. Handling Missing Values...")
        missing_before = df_clean.isnull().sum().sum()
        
        # Impute with median for each feature
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"   Imputed {col} with median: {median_val}")
        
        missing_after = df_clean.isnull().sum().sum()
        print(f"   Missing values: {missing_before} -> {missing_after}")
        
        # Remove duplicates
        print("\n2. Removing Duplicates...")
        duplicates = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        print(f"   Removed {duplicates} duplicate rows")
        
        # Validate data ranges
        print("\n3. Validating Data Ranges...")
        feature_cols = [col for col in df_clean.columns if col.startswith('feature_')]
        outliers_fixed = 0
        for col in feature_cols:
            mask = (df_clean[col] < 0) | (df_clean[col] > 3)
            if mask.any():
                outliers_fixed += mask.sum()
                df_clean.loc[mask, col] = df_clean[col].median()
        print(f"   Fixed {outliers_fixed} out-of-range values")
        
        print(f"\nCleaning Summary:")
        print(f"   Initial shape: {initial_shape}")
        print(f"   Final shape: {df_clean.shape}")
        print(f"   Rows removed: {initial_shape[0] - df_clean.shape[0]}")
        
        # Save cleaning report
        report = self._generate_cleaning_report(df, df_clean)
        report_path = self.reports_dir / 'DATASET_SUMMARY.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n[OK] Cleaning report saved to: {report_path}")
        
        return df_clean
    
    def _generate_cleaning_report(self, df_before, df_after):
        """Generate cleaning report"""
        report = []
        report.append("=" * 80)
        report.append("LUNG CANCER DATASET CLEANING REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Original Shape: {df_before.shape}")
        report.append(f"Cleaned Shape: {df_after.shape}")
        report.append(f"Rows Removed: {df_before.shape[0] - df_after.shape[0]}")
        report.append("")
        report.append("Missing Values Handled:")
        missing_before = df_before.isnull().sum()
        for col in missing_before[missing_before > 0].index:
            report.append(f"  - {col}: {missing_before[col]} values imputed")
        report.append("")
        report.append("Data Quality Checks:")
        report.append("  [OK] All features in valid range (0-3)")
        report.append("  [OK] No duplicate rows")
        report.append("  [OK] No missing values")
        report.append("")
        report.append("=" * 80)
        return "\n".join(report)
    
    def perform_eda(self, df):
        """Perform exploratory data analysis"""
        print("\n" + "=" * 80)
        print("PERFORMING EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Basic statistics
        print("\n1. Generating Basic Statistics...")
        print(df.describe())
        
        # Class distribution
        print("\n2. Analyzing Class Distribution...")
        plt.figure(figsize=(10, 6))
        class_counts = df['class'].value_counts().sort_index()
        plt.bar(class_counts.index, class_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        plt.xlabel('Cancer Type', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title('Lung Cancer Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks([1, 2, 3], ['Type 1', 'Type 2', 'Type 3'])
        for i, v in enumerate(class_counts.values):
            plt.text(class_counts.index[i], v + 0.2, str(v), ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'eda' / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   [OK] Class distribution plot saved")
        
        # Feature distributions
        print("\n3. Analyzing Feature Distributions...")
        feature_cols = [f'feature_{i}' for i in range(1, 11)]
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.ravel()
        for idx, col in enumerate(feature_cols):
            df[col].value_counts().sort_index().plot(kind='bar', ax=axes[idx], color='steelblue')
            axes[idx].set_title(col, fontsize=10)
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'eda' / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   [OK] Feature distributions plot saved")
        
        # Correlation analysis
        print("\n4. Analyzing Correlations...")
        feature_cols_all = [col for col in df.columns if col.startswith('feature_')]
        corr_matrix = df[feature_cols_all + ['class']].corr()
        
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'eda' / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   [OK] Correlation heatmap saved")
        
        # Feature importance
        print("\n5. Analyzing Feature Importance...")
        correlations = df[feature_cols_all].corrwith(df['class']).abs().sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        correlations.head(20).plot(kind='barh', color='coral')
        plt.title('Top 20 Features by Correlation with Class', fontsize=14, fontweight='bold')
        plt.xlabel('Absolute Correlation', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.reports_dir / 'eda' / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   [OK] Feature importance plot saved")
        
        # Save EDA report
        report = self._generate_eda_report(df, correlations)
        report_path = self.reports_dir / 'EDA_SUMMARY.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n[OK] EDA report saved to: {report_path}")
        
        return df
    
    def _generate_eda_report(self, df, correlations):
        """Generate EDA report"""
        report = []
        report.append("=" * 80)
        report.append("LUNG CANCER EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        report.append("Dataset Overview:")
        report.append(f"  - Total Samples: {len(df)}")
        report.append(f"  - Total Features: {len(df.columns) - 1}")
        report.append(f"  - Classes: 3")
        report.append("")
        report.append("Class Distribution:")
        for cls, count in df['class'].value_counts().sort_index().items():
            pct = (count / len(df)) * 100
            report.append(f"  - Class {int(cls)}: {count} samples ({pct:.1f}%)")
        report.append("")
        report.append("Top 10 Most Correlated Features with Class:")
        for i, (feat, corr) in enumerate(correlations.head(10).items(), 1):
            report.append(f"  {i}. {feat}: {corr:.4f}")
        report.append("")
        report.append("Key Insights:")
        report.append("  â€¢ Small dataset (32 samples) - suitable for specialized techniques")
        report.append("  â€¢ Relatively balanced 3-class problem")
        report.append("  â€¢ All features are nominal (0-3 values)")
        report.append("  â€¢ Some features show strong correlation with cancer type")
        report.append("")
        report.append("=" * 80)
        return "\n".join(report)
    
    def engineer_features(self, df):
        """Engineer features for lung cancer dataset"""
        print("\n" + "=" * 80)
        print("ENGINEERING FEATURES")
        print("=" * 80)
        
        df_eng = df.copy()
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        
        # Feature groups
        print("\n1. Creating Feature Groups...")
        group1 = [f'feature_{i}' for i in range(1, 20)]
        group2 = [f'feature_{i}' for i in range(20, 40)]
        group3 = [f'feature_{i}' for i in range(40, 57)]
        
        df_eng['group1_mean'] = df_eng[group1].mean(axis=1)
        df_eng['group2_mean'] = df_eng[group2].mean(axis=1)
        df_eng['group3_mean'] = df_eng[group3].mean(axis=1)
        
        df_eng['group1_std'] = df_eng[group1].std(axis=1)
        df_eng['group2_std'] = df_eng[group2].std(axis=1)
        df_eng['group3_std'] = df_eng[group3].std(axis=1)
        print("   [OK] Created 6 group-based features")
        
        # Aggregation features
        print("\n2. Creating Aggregation Features...")
        df_eng['total_sum'] = df_eng[feature_cols].sum(axis=1)
        df_eng['total_mean'] = df_eng[feature_cols].mean(axis=1)
        df_eng['total_std'] = df_eng[feature_cols].std(axis=1)
        df_eng['total_max'] = df_eng[feature_cols].max(axis=1)
        df_eng['total_min'] = df_eng[feature_cols].min(axis=1)
        df_eng['total_range'] = df_eng['total_max'] - df_eng['total_min']
        print("   [OK] Created 6 aggregation features")
        
        # Count features
        print("\n3. Creating Count Features...")
        for value in [0, 1, 2, 3]:
            df_eng[f'count_value_{value}'] = (df_eng[feature_cols] == value).sum(axis=1)
        print("   [OK] Created 4 count features")
        
        # Ratio features
        print("\n4. Creating Ratio Features...")
        df_eng['high_value_ratio'] = (df_eng['count_value_2'] + df_eng['count_value_3']) / len(feature_cols)
        df_eng['low_value_ratio'] = (df_eng['count_value_0'] + df_eng['count_value_1']) / len(feature_cols)
        print("   [OK] Created 2 ratio features")
        
        print(f"\nFeature Engineering Summary:")
        print(f"   Original features: {len(feature_cols)}")
        print(f"   Engineered features: {len(df_eng.columns) - len(df.columns)}")
        print(f"   Total features: {len(df_eng.columns) - 1}")  # Exclude class
        
        # Save feature engineering report
        report = self._generate_feature_report(df, df_eng)
        report_path = self.reports_dir / 'FEATURE_ENGINEERING_SUMMARY.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n[OK] Feature engineering report saved to: {report_path}")
        
        return df_eng
    
    def _generate_feature_report(self, df_before, df_after):
        """Generate feature engineering report"""
        report = []
        report.append("=" * 80)
        report.append("LUNG CANCER FEATURE ENGINEERING REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Original Features: {len(df_before.columns) - 1}")
        report.append(f"Engineered Features: {len(df_after.columns) - len(df_before.columns)}")
        report.append(f"Total Features: {len(df_after.columns) - 1}")
        report.append("")
        report.append("Feature Categories Created:")
        report.append("  1. Group Statistics (6 features)")
        report.append("     - Mean and std for 3 feature groups")
        report.append("  2. Aggregation Features (6 features)")
        report.append("     - Sum, mean, std, max, min, range")
        report.append("  3. Count Features (4 features)")
        report.append("     - Count of each value (0, 1, 2, 3)")
        report.append("  4. Ratio Features (2 features)")
        report.append("     - High/low value ratios")
        report.append("")
        report.append("New Features List:")
        new_features = [col for col in df_after.columns if col not in df_before.columns]
        for feat in new_features:
            report.append(f"  - {feat}")
        report.append("")
        report.append("=" * 80)
        return "\n".join(report)
    
    def validate_and_split(self, df):
        """Validate and split dataset for ML"""
        print("\n" + "=" * 80)
        print("VALIDATING AND SPLITTING DATASET")
        print("=" * 80)
        
        # Separate features and target
        X = df.drop('class', axis=1)
        y = df['class']
        
        print("\n1. Data Quality Checks...")
        print(f"   [OK] No missing values: {X.isnull().sum().sum() == 0}")
        print(f"   [OK] No infinite values: {np.isinf(X.values).sum() == 0}")
        print(f"   [OK] All numeric: {X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]}")
        
        print("\n2. Class Balance Check...")
        class_dist = y.value_counts().sort_index()
        for cls, count in class_dist.items():
            pct = (count / len(y)) * 100
            print(f"   Class {int(cls)}: {count} samples ({pct:.1f}%)")
        
        print("\n3. Splitting Dataset (80-20 train-test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalize features
        print("\n4. Normalizing Features...")
        scaler = StandardScaler()
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
        
        # Save splits
        ml_ready_dir = self.base_dir / 'ml_ready'
        ml_ready_dir.mkdir(exist_ok=True)
        
        X_train_scaled.to_csv(ml_ready_dir / 'X_train.csv', index=False)
        X_test_scaled.to_csv(ml_ready_dir / 'X_test.csv', index=False)
        y_train.to_csv(ml_ready_dir / 'y_train.csv', index=False)
        y_test.to_csv(ml_ready_dir / 'y_test.csv', index=False)
        
        print(f"\n[OK] ML-ready datasets saved to: {ml_ready_dir}")
        print(f"   - X_train: {X_train_scaled.shape}")
        print(f"   - X_test: {X_test_scaled.shape}")
        print(f"   - y_train: {y_train.shape}")
        print(f"   - y_test: {y_test.shape}")
        
        # Save validation report
        report = self._generate_validation_report(X_train_scaled, X_test_scaled, y_train, y_test)
        report_path = self.reports_dir / 'VALIDATION_SUMMARY.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n[OK] Validation report saved to: {report_path}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _generate_validation_report(self, X_train, X_test, y_train, y_test):
        """Generate validation report"""
        report = []
        report.append("=" * 80)
        report.append("LUNG CANCER DATASET VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        report.append("Train-Test Split:")
        report.append(f"  - Training samples: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
        report.append(f"  - Testing samples: {len(X_test)} ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")
        report.append(f"  - Total features: {X_train.shape[1]}")
        report.append("")
        report.append("Training Set Class Distribution:")
        for cls, count in y_train.value_counts().sort_index().items():
            pct = (count / len(y_train)) * 100
            report.append(f"  - Class {int(cls)}: {count} samples ({pct:.1f}%)")
        report.append("")
        report.append("Testing Set Class Distribution:")
        for cls, count in y_test.value_counts().sort_index().items():
            pct = (count / len(y_test)) * 100
            report.append(f"  - Class {int(cls)}: {count} samples ({pct:.1f}%)")
        report.append("")
        report.append("Data Quality:")
        report.append("  [OK] Features normalized (StandardScaler)")
        report.append("  [OK] Stratified split maintains class balance")
        report.append("  [OK] No missing or infinite values")
        report.append("  [OK] Ready for ML model training")
        report.append("")
        report.append("=" * 80)
        return "\n".join(report)
    
    def generate_final_summary(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 80)
        print("GENERATING FINAL SUMMARY")
        print("=" * 80)
        
        summary = []
        summary.append("=" * 80)
        summary.append("LUNG CANCER DATASET PROCESSING - COMPREHENSIVE SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        summary.append("DATASET INFORMATION")
        summary.append("-" * 80)
        summary.append("Source: UCI Machine Learning Repository")
        summary.append("Title: Lung Cancer Data")
        summary.append("Description: 3 types of pathological lung cancers")
        summary.append("Samples: 32")
        summary.append("Original Features: 56 predictive + 1 class")
        summary.append("Classes: 3 (Type 1, Type 2, Type 3)")
        summary.append("")
        summary.append("PROCESSING PIPELINE")
        summary.append("-" * 80)
        summary.append("[OK] 1. Data Loading and Parsing")
        summary.append("[OK] 2. Data Cleaning (missing values, duplicates, validation)")
        summary.append("[OK] 3. Exploratory Data Analysis (distributions, correlations)")
        summary.append("[OK] 4. Feature Engineering (groups, aggregations, counts, ratios)")
        summary.append("[OK] 5. Data Validation and Train-Test Split (80-20)")
        summary.append("[OK] 6. Feature Normalization (StandardScaler)")
        summary.append("")
        summary.append("OUTPUT FILES")
        summary.append("-" * 80)
        summary.append("Master Dataset:")
        summary.append(f"  - {self.master_dataset_path}")
        summary.append("")
        summary.append("ML-Ready Datasets:")
        summary.append(f"  - {self.base_dir / 'ml_ready' / 'X_train.csv'}")
        summary.append(f"  - {self.base_dir / 'ml_ready' / 'X_test.csv'}")
        summary.append(f"  - {self.base_dir / 'ml_ready' / 'y_train.csv'}")
        summary.append(f"  - {self.base_dir / 'ml_ready' / 'y_test.csv'}")
        summary.append("")
        summary.append("Reports:")
        summary.append(f"  - {self.reports_dir / 'DATASET_SUMMARY.txt'}")
        summary.append(f"  - {self.reports_dir / 'EDA_SUMMARY.txt'}")
        summary.append(f"  - {self.reports_dir / 'FEATURE_ENGINEERING_SUMMARY.txt'}")
        summary.append(f"  - {self.reports_dir / 'VALIDATION_SUMMARY.txt'}")
        summary.append(f"  - {self.reports_dir / 'LUNG_CANCER_MODULE_SUMMARY.txt'}")
        summary.append("")
        summary.append("Visualizations:")
        summary.append(f"  - {self.reports_dir / 'eda' / 'class_distribution.png'}")
        summary.append(f"  - {self.reports_dir / 'eda' / 'feature_distributions.png'}")
        summary.append(f"  - {self.reports_dir / 'eda' / 'correlation_heatmap.png'}")
        summary.append(f"  - {self.reports_dir / 'eda' / 'feature_importance.png'}")
        summary.append("")
        summary.append("KEY INSIGHTS")
        summary.append("-" * 80)
        summary.append("- Small dataset (32 samples) - ideal for specialized ML techniques")
        summary.append("- Balanced 3-class classification problem")
        summary.append("- All features are nominal (0-3 values)")
        summary.append("- Missing values successfully imputed")
        summary.append("- 18 engineered features created from 56 original features")
        summary.append("- Features normalized for optimal ML performance")
        summary.append("- Dataset ready for classification modeling")
        summary.append("")
        summary.append("RECOMMENDED NEXT STEPS")
        summary.append("-" * 80)
        summary.append("1. Model Selection (SVM, Random Forest, Neural Networks)")
        summary.append("2. Cross-validation due to small sample size")
        summary.append("3. Hyperparameter tuning")
        summary.append("4. Model evaluation and comparison")
        summary.append("5. Feature selection/importance analysis")
        summary.append("6. Ensemble methods for improved accuracy")
        summary.append("")
        summary.append("=" * 80)
        summary.append("PROCESSING COMPLETE - DATASET READY FOR ML")
        summary.append("=" * 80)
        
        summary_text = "\n".join(summary)
        
        # Save summary
        summary_path = self.reports_dir / 'LUNG_CANCER_MODULE_SUMMARY.txt'
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"\n[OK] Final summary saved to: {summary_path}")
        
        return summary_text
    
    def run_full_pipeline(self):
        """Execute the complete processing pipeline"""
        print("\n" + "=" * 80)
        print("LUNG CANCER DATASET PROCESSING PIPELINE")
        print("=" * 80)
        print("\nStarting comprehensive data processing workflow...")
        
        try:
            # Step 1: Load raw data
            df = self.load_raw_data()
            
            # Step 2: Clean dataset
            df_cleaned = self.clean_dataset(df)
            
            # Step 3: Perform EDA
            df_eda = self.perform_eda(df_cleaned)
            
            # Step 4: Engineer features
            df_engineered = self.engineer_features(df_eda)
            
            # Step 5: Validate and split
            X_train, X_test, y_train, y_test = self.validate_and_split(df_engineered)
            
            # Step 6: Generate final summary
            self.generate_final_summary()
            
            print("\n" + "=" * 80)
            print("[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed with exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main execution function"""
    processor = LungCancerProcessor()
    success = processor.run_full_pipeline()
    
    if success:
        print("\n[SUCCESS] All lung cancer data processing tasks completed successfully!")
        print("[SUCCESS] Dataset is ready for machine learning model development.")
    else:
        print("\n[ERROR] Processing failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


