"""
Comprehensive EDA, Feature Engineering, and Feature Selection Script
Healthcare Monitoring and Early Disease Prediction System
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import json

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class EDAFeatureEngineer:
    """Comprehensive EDA and Feature Engineering Pipeline"""

    def __init__(self, dataset_name: str, input_path: str, output_dir: str):
        self.dataset_name = dataset_name
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.eda_dir = self.output_dir / 'eda'
        self.eda_dir.mkdir(parents=True, exist_ok=True)

        self.df: pd.DataFrame = pd.DataFrame()
        self.df_engineered: pd.DataFrame = pd.DataFrame()
        self.target_column: str = ""
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.feature_importance: dict = {}
        self.selected_features: list[str] = []

        self.eda_report: list[str] = []
        self.feature_engineering_report: list[str] = []
        self.feature_selection_report: list[str] = []

    def load_data(self) -> "EDAFeatureEngineer":
        print(f"Loading dataset from {self.input_path}...")
        self.df = pd.read_csv(self.input_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self

    def identify_features(self) -> "EDAFeatureEngineer":
        self.numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Numeric features: {len(self.numeric_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        return self

    def detect_target(self) -> "EDAFeatureEngineer":
        target_keywords = ['target', 'label', 'class', 'outcome', 'diagnosis',
                           'stress_label', 'age_group', 'disease', 'condition']
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                self.target_column = col
                print(f"Detected target column: {self.target_column}")
                return self
        print("Warning: No target column detected. Using last column as target.")
        self.target_column = str(self.df.columns[-1])
        return self

    def perform_eda(self) -> "EDAFeatureEngineer":
        print("\n" + "="*70)
        print("PERFORMING EXPLORATORY DATA ANALYSIS")
        print("="*70)

        self.eda_report.append("="*70)
        self.eda_report.append(f"EDA REPORT - {self.dataset_name.upper()}")
        self.eda_report.append("="*70)
        self.eda_report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.eda_report.append("")

        self._dataset_overview()
        self._statistical_analysis()
        self._categorical_analysis()
        self._correlation_analysis()
        self._generate_visualizations()

        print("EDA completed successfully!")
        return self

    def _dataset_overview(self):
        print("\n1. Dataset Overview")
        print("-" * 70)

        overview = f"""
DATASET OVERVIEW
{"-"*70}
Number of rows: {self.df.shape[0]}
Number of columns: {self.df.shape[1]}
Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

FEATURE LIST
{"-"*70}
"""
        self.eda_report.append(overview)

        for i, col in enumerate(self.df.columns, 1):
            dtype = str(self.df[col].dtype)
            self.eda_report.append(f"{i:3d}. {col:40s} ({dtype})")

        self.eda_report.append("")

        dtype_summary = f"""
DATA TYPES SUMMARY
{"-"*70}
Numeric features: {len(self.numeric_features)}
Categorical features: {len(self.categorical_features)}
"""
        self.eda_report.append(dtype_summary)
        print(dtype_summary)

    def _statistical_analysis(self):
        print("\n2. Statistical Analysis")
        print("-" * 70)

        if len(self.numeric_features) > 0:
            stats_summary = f"""
NUMERICAL FEATURES STATISTICS
{"-"*70}
"""
            self.eda_report.append(stats_summary)
            desc = self.df[self.numeric_features].describe()
            self.eda_report.append(desc.to_string())
            self.eda_report.append("")
            print(f"Statistical summary saved for {len(self.numeric_features)} numeric features")

    def _categorical_analysis(self):
        print("\n3. Categorical Analysis")
        print("-" * 70)

        if len(self.categorical_features) > 0:
            self.eda_report.append(f"\nCATEGORICAL FEATURES ANALYSIS\n{'-'*70}\n")
            for col in self.categorical_features:
                unique_count = self.df[col].nunique()
                value_counts = self.df[col].value_counts()
                self.eda_report.append(f"\nFeature: {col}")
                self.eda_report.append(f"Unique values: {unique_count}")
                self.eda_report.append("Value distribution:")
                self.eda_report.append(value_counts.to_string())
                self.eda_report.append("")
            print(f"Categorical analysis completed for {len(self.categorical_features)} features")

    def _correlation_analysis(self):
        print("\n4. Correlation Analysis")
        print("-" * 70)

        if len(self.numeric_features) > 1:
            # Use pandas corr() via select_dtypes to satisfy pyright's stub typing
            corr_matrix = self.df[self.numeric_features].select_dtypes(include=[np.number]).corr(min_periods=1)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append((
                            corr_matrix.columns[i],
                            corr_matrix.columns[j],
                            corr_matrix.iloc[i, j]
                        ))

            self.eda_report.append(f"\nCORRELATION ANALYSIS\n{'-'*70}\nHighly correlated pairs (|r| > 0.8): {len(high_corr_pairs)}\n")
            if high_corr_pairs:
                self.eda_report.append("Highly Correlated Pairs:")
                for feat1, feat2, corr_val in high_corr_pairs:
                    self.eda_report.append(f"  {feat1} <-> {feat2}: {corr_val:.3f}")

            print(f"Correlation analysis completed. Found {len(high_corr_pairs)} highly correlated pairs")

    def _generate_visualizations(self):
        print("\n5. Generating Visualizations")
        print("-" * 70)

        if len(self.numeric_features) > 1:
            self._plot_correlation_heatmap()
        if self.target_column:
            self._plot_target_distribution()
        if len(self.numeric_features) > 0:
            self._plot_numerical_distributions()
        if len(self.categorical_features) > 0:
            self._plot_categorical_distributions()
        if len(self.numeric_features) > 0:
            self._plot_boxplots()
        if self.target_column and len(self.numeric_features) > 0:
            self._plot_feature_relationships()

        print("All visualizations generated successfully!")

    def _plot_correlation_heatmap(self):
        plt.figure(figsize=(16, 14))
        features_to_plot = self.numeric_features[:50] if len(self.numeric_features) > 50 else self.numeric_features
        corr = self.df[features_to_plot].select_dtypes(include=[np.number]).corr(min_periods=1)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Heatmap - {self.dataset_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] Correlation heatmap saved")

    def _plot_target_distribution(self):
        plt.figure(figsize=(10, 6))
        if self.df[self.target_column].dtype in ['object', 'category'] or self.df[self.target_column].nunique() < 20:
            value_counts = self.df[self.target_column].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
            plt.xlabel('Class', fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.title(f'Target Distribution - {self.target_column}', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
        else:
            sns.histplot(data=self.df, x=self.target_column, kde=True, bins=30, color='steelblue')
            plt.xlabel(self.target_column, fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title(f'Target Distribution - {self.target_column}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] Target distribution saved")

    def _plot_numerical_distributions(self):
        n_features = min(len(self.numeric_features), 20)
        features_to_plot = self.numeric_features[:n_features]
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, col in enumerate(features_to_plot):
            if col != self.target_column:
                sns.histplot(data=self.df, x=col, kde=True, ax=axes[idx], color='skyblue')
                axes[idx].set_title(col, fontsize=10, fontweight='bold')
                axes[idx].set_xlabel('')

        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Numerical Feature Distributions - {self.dataset_name}',
                     fontsize=16, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] Numerical distributions saved")

    def _plot_categorical_distributions(self):
        if len(self.categorical_features) == 0:
            return
        n_features = len(self.categorical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, col in enumerate(self.categorical_features):
            value_counts = self.df[col].value_counts().head(10)
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[idx], palette='Set2')
            axes[idx].set_title(col, fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Count')

        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Categorical Feature Distributions - {self.dataset_name}',
                     fontsize=16, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'categorical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] Categorical distributions saved")

    def _plot_boxplots(self):
        n_features = min(len(self.numeric_features), 20)
        features_to_plot = [f for f in self.numeric_features[:n_features] if f != self.target_column]
        if len(features_to_plot) == 0:
            return

        n_cols = 4
        n_rows = (len(features_to_plot) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if len(features_to_plot) > 1 else [axes]

        for idx, col in enumerate(features_to_plot):
            sns.boxplot(y=self.df[col].dropna(), ax=axes[idx], color='lightcoral')
            axes[idx].set_title(col, fontsize=10, fontweight='bold')

        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Boxplots for Outlier Detection - {self.dataset_name}',
                     fontsize=16, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] Boxplots saved")

    def _plot_feature_relationships(self):
        n_features = min(len(self.numeric_features), 12)
        features_to_plot = [f for f in self.numeric_features[:n_features] if f != self.target_column]
        if len(features_to_plot) == 0:
            return

        n_cols = 3
        n_rows = (len(features_to_plot) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if len(features_to_plot) > 1 else [axes]

        for idx, col in enumerate(features_to_plot):
            if self.df[self.target_column].dtype in ['object', 'category'] or self.df[self.target_column].nunique() < 10:
                sns.boxplot(x=self.target_column, y=col, data=self.df, ax=axes[idx], palette='Set3')
            else:
                sns.scatterplot(x=col, y=self.target_column, data=self.df, ax=axes[idx], alpha=0.6)
            axes[idx].set_title(f'{col} vs {self.target_column}', fontsize=10, fontweight='bold')
            axes[idx].tick_params(axis='x', rotation=45)

        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Feature Relationships with Target - {self.dataset_name}',
                     fontsize=16, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.savefig(self.eda_dir / 'feature_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  [OK] Feature relationships saved")

    def perform_feature_engineering(self) -> "EDAFeatureEngineer":
        print("\n" + "="*70)
        print("PERFORMING FEATURE ENGINEERING")
        print("="*70)

        self.df_engineered = self.df.copy()

        self.feature_engineering_report.append("="*70)
        self.feature_engineering_report.append(f"FEATURE ENGINEERING REPORT - {self.dataset_name.upper()}")
        self.feature_engineering_report.append("="*70)
        self.feature_engineering_report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.feature_engineering_report.append("")

        self._encode_categorical()
        self._create_derived_features()
        self._scale_features()

        print("Feature engineering completed successfully!")
        return self

    def _encode_categorical(self):
        print("\n1. Encoding Categorical Variables")
        print("-" * 70)

        self.feature_engineering_report.append(f"\nCATEGORICAL ENCODING\n{'-'*70}\n")

        for col in self.categorical_features:
            if col == self.target_column:
                continue
            unique_count = self.df_engineered[col].nunique()
            if unique_count == 2:
                le = LabelEncoder()
                self.df_engineered[col] = le.fit_transform(self.df_engineered[col].astype(str))
                self.feature_engineering_report.append(f"  {col}: Label Encoding (binary)")
                print(f"  [OK] {col}: Label Encoding (binary)")
            elif unique_count <= 10:
                dummies = pd.get_dummies(self.df_engineered[col], prefix=col, drop_first=True)
                self.df_engineered = pd.concat([self.df_engineered, dummies], axis=1)
                self.df_engineered = self.df_engineered.drop(columns=[col])
                self.feature_engineering_report.append(f"  {col}: One-Hot Encoding ({unique_count} categories)")
                print(f"  [OK] {col}: One-Hot Encoding ({unique_count} categories)")
            else:
                le = LabelEncoder()
                self.df_engineered[col] = le.fit_transform(self.df_engineered[col].astype(str))
                self.feature_engineering_report.append(f"  {col}: Label Encoding ({unique_count} categories)")
                print(f"  [OK] {col}: Label Encoding ({unique_count} categories)")

        self.feature_engineering_report.append("")

    def _create_derived_features(self):
        print("\n2. Creating Derived Features")
        print("-" * 70)

        self.feature_engineering_report.append(f"\nDERIVED FEATURES\n{'-'*70}\n")
        created_any = False

        if 'bmxbmi' in self.df_engineered.columns:
            self.df_engineered['bmi_obese'] = (self.df_engineered['bmxbmi'] >= 30).astype(int)
            self.df_engineered['bmi_overweight'] = (
                (self.df_engineered['bmxbmi'] >= 25) & (self.df_engineered['bmxbmi'] < 30)
            ).astype(int)
            self.feature_engineering_report.append("  Created: bmi_obese, bmi_overweight")
            print("  [OK] Created BMI category features")
            created_any = True

        age_col = next((c for c in ['age', 'ridageyr'] if c in self.df_engineered.columns), None)
        if age_col is not None:
            self.df_engineered['age_squared'] = self.df_engineered[age_col] ** 2
            self.feature_engineering_report.append(f"  Created: age_squared")
            print("  [OK] Created age polynomial feature")
            created_any = True

        if 'ecg_mean' in self.df_engineered.columns and 'eda_mean' in self.df_engineered.columns:
            self.df_engineered['ecg_eda_interaction'] = (
                self.df_engineered['ecg_mean'] * self.df_engineered['eda_mean']
            )
            self.feature_engineering_report.append("  Created: ecg_eda_interaction")
            print("  [OK] Created ECG-EDA interaction feature")
            created_any = True

        if not created_any:
            self.feature_engineering_report.append("  No derived features created (not applicable for this dataset)")
            print("  No derived features created")

        self.feature_engineering_report.append("")

    def _scale_features(self):
        print("\n3. Scaling Numerical Features")
        print("-" * 70)

        self.feature_engineering_report.append(f"\nFEATURE SCALING\n{'-'*70}\n")

        numeric_cols = self.df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numeric_cols:
            numeric_cols.remove(self.target_column)

        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            self.df_engineered[numeric_cols] = scaler.fit_transform(self.df_engineered[numeric_cols])
            self.feature_engineering_report.append(f"  Applied StandardScaler to {len(numeric_cols)} features")
            print(f"  [OK] StandardScaler applied to {len(numeric_cols)} features")

        self.feature_engineering_report.append("")

        output_path = self.output_dir.parent / self.dataset_name / f"{self.dataset_name}_engineered.csv"
        self.df_engineered.to_csv(output_path, index=False)
        print(f"\n  [OK] Engineered dataset saved: {output_path}")

    def perform_feature_selection(self) -> "EDAFeatureEngineer":
        print("\n" + "="*70)
        print("PERFORMING FEATURE SELECTION")
        print("="*70)

        self.feature_selection_report.append("="*70)
        self.feature_selection_report.append(f"FEATURE SELECTION REPORT - {self.dataset_name.upper()}")
        self.feature_selection_report.append("="*70)
        self.feature_selection_report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.feature_selection_report.append("")

        from typing import cast
        X = self.df_engineered.drop(columns=[self.target_column])
        # cast required: pyright types df[str] as Series|DataFrame|Unknown due to overload ambiguity
        y: pd.Series = cast(pd.Series, self.df_engineered[self.target_column])

        self._variance_threshold_selection(X)
        self._correlation_based_selection(X)
        self._mutual_information_selection(X, y)
        self._model_based_selection(X, y)
        self._finalize_feature_selection(X)

        print("Feature selection completed successfully!")
        return self

    def _variance_threshold_selection(self, X: pd.DataFrame):
        print("\n1. Variance Threshold Selection")
        print("-" * 70)

        selector = VarianceThreshold(threshold=0.01)
        selector.fit(X)
        support_mask = selector.get_support()
        low_variance_features = X.columns[~np.array(support_mask)].tolist()

        self.feature_selection_report.append(
            f"\nVARIANCE THRESHOLD ANALYSIS\n{'-'*70}\nThreshold: 0.01\nLow variance features: {len(low_variance_features)}\n"
        )
        for feat in low_variance_features:
            self.feature_selection_report.append(f"  - {feat}")
        self.feature_selection_report.append("")
        print(f"  [OK] Identified {len(low_variance_features)} low-variance features")

    def _correlation_based_selection(self, X: pd.DataFrame):
        print("\n2. Correlation-Based Selection")
        print("-" * 70)

        corr_matrix = X.corr(numeric_only=True).abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]

        self.feature_selection_report.append(
            f"\nCORRELATION-BASED SELECTION\n{'-'*70}\nThreshold: 0.95\nHighly correlated features: {len(high_corr_features)}\n"
        )
        for feat in high_corr_features:
            self.feature_selection_report.append(f"  - {feat}")
        self.feature_selection_report.append("")
        print(f"  [OK] Identified {len(high_corr_features)} highly correlated features")

    def _mutual_information_selection(self, X: pd.DataFrame, y: pd.Series):
        print("\n3. Mutual Information Selection")
        print("-" * 70)

        is_classification = y.dtype in ['object', 'category'] or y.nunique() < 20

        if is_classification:
            y_encoded = LabelEncoder().fit_transform(y.astype(str)) if y.dtype == 'object' or str(y.dtype) == 'category' else y.to_numpy()
            mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)

        mi_scores_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)
        self.feature_importance['mutual_information'] = mi_scores_df

        self.feature_selection_report.append(
            f"\nMUTUAL INFORMATION SCORES\n{'-'*70}\nTask: {'Classification' if is_classification else 'Regression'}\nTop 20 features:\n"
        )
        for _, row in mi_scores_df.head(20).iterrows():
            self.feature_selection_report.append(f"  {str(row['feature']):40s} {row['mi_score']:.4f}")
        self.feature_selection_report.append("")
        print("  [OK] Mutual information scores calculated")

    def _model_based_selection(self, X: pd.DataFrame, y: pd.Series):
        print("\n4. Model-Based Feature Selection")
        print("-" * 70)

        is_classification = y.dtype in ['object', 'category'] or y.nunique() < 20

        if is_classification:
            y_encoded = LabelEncoder().fit_transform(y.astype(str)) if y.dtype == 'object' or str(y.dtype) == 'category' else y.to_numpy()
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y_encoded)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)

        importance_df = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        self.feature_importance['random_forest'] = importance_df

        self.feature_selection_report.append(
            f"\nRANDOM FOREST FEATURE IMPORTANCE\n{'-'*70}\nModel: {'Classifier' if is_classification else 'Regressor'}\nTop 20 features:\n"
        )
        for _, row in importance_df.head(20).iterrows():
            self.feature_selection_report.append(f"  {str(row['feature']):40s} {row['importance']:.4f}")
        self.feature_selection_report.append("")
        print("  [OK] Random Forest feature importance calculated")

    def _finalize_feature_selection(self, X: pd.DataFrame):
        print("\n5. Finalizing Feature Selection")
        print("-" * 70)

        if 'mutual_information' not in self.feature_importance or 'random_forest' not in self.feature_importance:
            return

        mi_df = self.feature_importance['mutual_information'].set_index('feature')
        rf_df = self.feature_importance['random_forest'].set_index('feature')

        mi_min, mi_max = mi_df['mi_score'].min(), mi_df['mi_score'].max()
        rf_min, rf_max = rf_df['importance'].min(), rf_df['importance'].max()

        mi_norm = (mi_df['mi_score'] - mi_min) / (mi_max - mi_min) if mi_max > mi_min else mi_df['mi_score']
        rf_norm = (rf_df['importance'] - rf_min) / (rf_max - rf_min) if rf_max > rf_min else rf_df['importance']

        combined_df = pd.DataFrame({
            'feature': X.columns,
            'combined_score': (mi_norm.reindex(X.columns).fillna(0) + rf_norm.reindex(X.columns).fillna(0)) / 2
        }).sort_values('combined_score', ascending=False)

        # Check if all scores are zero or very close to zero
        max_score = combined_df['combined_score'].max()
        if max_score < 1e-10:
            # If all scores are zero, select top 50% of features or at least 20 features
            n_features = max(20, len(X.columns) // 2)
            n_features = min(n_features, len(X.columns))
            self.selected_features = combined_df.head(n_features)['feature'].tolist()
            selection_method = f"Top {n_features} features (all importance scores near zero)"
            print(f"  [WARNING] All feature importance scores are near zero")
            print(f"  [INFO] Selecting top {n_features} features by default")
        else:
            # Normal selection based on cumulative importance
            cumsum = combined_df['combined_score'].cumsum() / combined_df['combined_score'].sum()
            n_features = int((cumsum <= 0.95).sum())
            # Ensure at least some features are selected
            if n_features == 0:
                n_features = min(20, len(X.columns))
            self.selected_features = combined_df.head(n_features)['feature'].tolist()
            selection_method = "95% cumulative importance"

        self.feature_selection_report.append(
            f"\nFINAL FEATURE SELECTION\n{'-'*70}\nSelection method: Combined MI + RF importance\nTotal features: {len(X.columns)}\nSelected features: {len(self.selected_features)}\nCumulative importance: 95%\n\nSelected features:\n"
        )
        for feat in self.selected_features:
            self.feature_selection_report.append(f"  - {feat}")

        selected_df = self.df_engineered[self.selected_features + [self.target_column]]
        output_path = self.output_dir.parent / self.dataset_name / f"{self.dataset_name}_selected_features.csv"
        selected_df.to_csv(output_path, index=False)

        print(f"  [OK] Selected {len(self.selected_features)} features ({selection_method})")
        print(f"  [OK] Selected features dataset saved: {output_path}")
        self.feature_selection_report.append("")

    def save_reports(self):
        print("\n" + "="*70)
        print("SAVING REPORTS")
        print("="*70)

        for path, report in [
            (self.output_dir / 'EDA_SUMMARY.txt', self.eda_report),
            (self.output_dir / 'FEATURE_ENGINEERING_SUMMARY.txt', self.feature_engineering_report),
            (self.output_dir / 'FEATURE_SELECTION_SUMMARY.txt', self.feature_selection_report),
        ]:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            print(f"  [OK] Saved: {path}")

        print("\nAll reports saved successfully!")

    def run_pipeline(self) -> "EDAFeatureEngineer":
        self.load_data()
        self.identify_features()
        self.detect_target()
        self.perform_eda()
        self.perform_feature_engineering()
        self.perform_feature_selection()
        self.save_reports()

        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        return self


def main():
    parser = argparse.ArgumentParser(description='EDA and Feature Engineering Pipeline')
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output_dir', required=True, help='Output directory for reports')

    args = parser.parse_args()
    pipeline = EDAFeatureEngineer(args.dataset, args.input, args.output_dir)
    pipeline.run_pipeline()


if __name__ == '__main__':
    main()
