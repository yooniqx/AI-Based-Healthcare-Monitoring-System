"""
Gallbladder Disease Dataset - Exploratory Data Analysis (EDA)
This script performs comprehensive EDA on the cleaned gallbladder dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except AttributeError:
        pass

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Define paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'datasets' / 'gallbladder'
REPORTS_DIR = BASE_DIR / 'reports' / 'gallbladder'
PLOTS_DIR = REPORTS_DIR / 'eda' / 'plots'

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class GallbladderEDA:
    """Class to perform EDA on gallbladder dataset"""
    
    def __init__(self):
        self.df: pd.DataFrame | None = None
        self.eda_findings: list = []
        
    def load_cleaned_dataset(self):
        """Load the cleaned dataset"""
        print("\n" + "="*80)
        print("LOADING CLEANED DATASET")
        print("="*80)
        
        filepath = DATA_DIR / 'gallstone_cleaned.csv'
        if filepath.exists():
            self.df = pd.read_csv(filepath)
            print(f"[OK] Loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
            return True
        else:
            print(f"[ERROR] File not found: {filepath}")
            print("Please run data_cleaning_gallbladder.py first")
            return False
    
    def perform_eda(self):
        """Perform comprehensive EDA"""
        if self.df is None:
            print("[ERROR] No dataset loaded")
            return
        
        # Type guard - after this check, self.df is guaranteed to be DataFrame
        assert self.df is not None
        
        print("\n" + "="*80)
        print("PERFORMING EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        self._dataset_overview()
        self._target_analysis()
        self._numerical_analysis()
        self._correlation_analysis()
        self._generate_visualizations()
        self._save_eda_summary()
        
        print("\n[OK] EDA completed successfully!")
    
    def _dataset_overview(self):
        """Generate dataset overview"""
        print("\n1. Dataset Overview")
        print("-" * 70)
        
        print(f"   Shape: {self.df.shape}")
        print(f"   Memory usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        print(f"   Missing values: {self.df.isnull().sum().sum()}")
        print(f"   Duplicates: {self.df.duplicated().sum()}")
        
        # Data types
        print(f"\n   Data types:")
        print(f"   - Numeric: {len(self.df.select_dtypes(include=[np.number]).columns)}")
        print(f"   - Categorical: {len(self.df.select_dtypes(include=['object']).columns)}")
    
    def _target_analysis(self):
        """Analyze target variable"""
        print("\n2. Target Variable Analysis")
        print("-" * 70)
        
        target_col = 'gallstone_status'
        if target_col in self.df.columns:
            dist = self.df[target_col].value_counts()
            print(f"\n   Gallstone Status Distribution:")
            for val, count in dist.items():
                pct = count / len(self.df) * 100
                print(f"   - Class {val}: {count} ({pct:.1f}%)")
            
            # Save distribution
            dist_df = pd.DataFrame({
                'Class': dist.index,
                'Count': dist.values,
                'Percentage': (dist.values / len(self.df) * 100).round(2)
            })
            dist_df.to_csv(REPORTS_DIR / 'target_distribution.csv', index=False)
    
    def _numerical_analysis(self):
        """Analyze numerical features"""
        print("\n3. Numerical Features Analysis")
        print("-" * 70)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"   Number of numerical features: {len(numeric_cols)}")
        
        # Descriptive statistics
        stats = self.df[numeric_cols].describe()
        stats.to_csv(REPORTS_DIR / 'detailed_statistics.csv')
        print(f"   [OK] Saved detailed statistics")
        
        # Show summary for key features
        key_features = ['age', 'body_mass_index', 'glucose', 'total_cholesterol', 'triglyceride']
        available_features = [f for f in key_features if f in self.df.columns]
        if available_features:
            print(f"\n   Key Features Summary:")
            print(self.df[available_features].describe().T[['mean', 'std', 'min', 'max']].to_string())
    
    def _correlation_analysis(self):
        """Analyze correlations"""
        print("\n4. Correlation Analysis")
        print("-" * 70)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        
        # Save correlation matrix
        corr_matrix.to_csv(REPORTS_DIR / 'correlation_matrix.csv')
        print(f"   [OK] Saved correlation matrix")
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    strong_corr.append({
                        'Feature_1': corr_matrix.columns[i],
                        'Feature_2': corr_matrix.columns[j],
                        'Correlation': round(corr_matrix.iloc[i, j], 3)
                    })
        
        if strong_corr:
            strong_corr_df = pd.DataFrame(strong_corr)
            strong_corr_df.to_csv(REPORTS_DIR / 'strong_correlations.csv', index=False)
            print(f"   [OK] Found {len(strong_corr)} strong correlations (|r| > 0.7)")
        
        # Target correlations
        if 'gallstone_status' in self.df.columns:
            target_corr = corr_matrix['gallstone_status'].sort_values(ascending=False)
            target_corr_df = pd.DataFrame({
                'Feature': target_corr.index,
                'Correlation': target_corr.values
            })
            target_corr_df.to_csv(REPORTS_DIR / 'target_correlations.csv', index=False)
            print(f"   [OK] Saved target correlations")
    
    def _generate_visualizations(self):
        """Generate all visualizations"""
        print("\n5. Generating Visualizations")
        print("-" * 70)
        
        self._plot_correlation_heatmap()
        self._plot_numerical_distributions()
        self._plot_boxplots()
        self._plot_features_by_target()
        self._plot_pairplot()
        
        print("   [OK] All visualizations generated")
    
    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Limit to first 30 features for readability
        cols_to_plot = numeric_cols[:30] if len(numeric_cols) > 30 else numeric_cols
        
        plt.figure(figsize=(16, 14))
        corr = self.df[cols_to_plot].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap - Gallbladder Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - Correlation heatmap saved")
    
    def _plot_numerical_distributions(self):
        """Plot distributions of numerical features"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Plot first 20 features
        features_to_plot = [col for col in numeric_cols if col != 'gallstone_status'][:20]
        
        n_cols = 4
        n_rows = (len(features_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if len(features_to_plot) > 1 else [axes]
        
        for idx, col in enumerate(features_to_plot):
            sns.histplot(data=self.df, x=col, kde=True, ax=axes[idx], color='skyblue')
            axes[idx].set_title(col, fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('')
        
        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Numerical Feature Distributions - Gallbladder Dataset',
                     fontsize=16, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - Numerical distributions saved")
    
    def _plot_boxplots(self):
        """Plot boxplots for outlier detection"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        features_to_plot = [col for col in numeric_cols if col != 'gallstone_status'][:20]
        
        n_cols = 4
        n_rows = (len(features_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten() if len(features_to_plot) > 1 else [axes]
        
        for idx, col in enumerate(features_to_plot):
            sns.boxplot(y=self.df[col], ax=axes[idx], color='lightcoral')
            axes[idx].set_title(col, fontsize=10, fontweight='bold')
        
        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Boxplots for Outlier Detection - Gallbladder Dataset',
                     fontsize=16, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - Boxplots saved")
    
    def _plot_features_by_target(self):
        """Plot features grouped by target variable"""
        if 'gallstone_status' not in self.df.columns:
            return
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        features_to_plot = [col for col in numeric_cols if col != 'gallstone_status'][:12]
        
        n_cols = 3
        n_rows = (len(features_to_plot) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if len(features_to_plot) > 1 else [axes]
        
        for idx, col in enumerate(features_to_plot):
            sns.boxplot(x='gallstone_status', y=col, data=self.df, ax=axes[idx], palette='Set3')
            axes[idx].set_title(f'{col} by Gallstone Status', fontsize=10, fontweight='bold')
        
        for idx in range(len(features_to_plot), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Features by Gallstone Status - Gallbladder Dataset',
                     fontsize=16, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'features_by_target.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - Features by target saved")
    
    def _plot_pairplot(self):
        """Plot pairplot for key features"""
        key_features = ['age', 'body_mass_index', 'glucose', 'total_cholesterol', 
                       'triglyceride', 'gallstone_status']
        available_features = [f for f in key_features if f in self.df.columns]
        
        if len(available_features) > 2:
            plt.figure(figsize=(15, 15))
            sns.pairplot(self.df[available_features], hue='gallstone_status', 
                        palette='Set1', diag_kind='kde', plot_kws={'alpha': 0.6})
            plt.suptitle('Pairplot of Key Features - Gallbladder Dataset',
                        fontsize=16, fontweight='bold', y=1.001)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'pairplot.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("      - Pairplot saved")
    
    def _save_eda_summary(self):
        """Save EDA summary report"""
        summary = {
            'Total_Records': len(self.df),
            'Total_Features': len(self.df.columns),
            'Numeric_Features': len(self.df.select_dtypes(include=[np.number]).columns),
            'Missing_Values': self.df.isnull().sum().sum(),
            'Duplicate_Rows': self.df.duplicated().sum()
        }
        
        if 'gallstone_status' in self.df.columns:
            summary['Class_0_Count'] = (self.df['gallstone_status'] == 0).sum()
            summary['Class_1_Count'] = (self.df['gallstone_status'] == 1).sum()
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(REPORTS_DIR / 'eda_summary.csv', index=False)
        print(f"\n[OK] EDA summary saved")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("GALLBLADDER DISEASE DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    eda = GallbladderEDA()
    
    if eda.load_cleaned_dataset():
        eda.perform_eda()
        
        print("\n" + "="*80)
        print("EDA COMPLETE")
        print("="*80)
        print(f"[OK] Reports saved to: {REPORTS_DIR}")
        print(f"[OK] Plots saved to: {PLOTS_DIR}")
        print("\nNext steps:")
        print("1. Review the EDA reports and visualizations")
        print("2. Run feature_engineering_gallbladder.py to create advanced features")
        print("="*80 + "\n")
    else:
        print("\n[ERROR] Failed to load dataset. Please run data_cleaning_gallbladder.py first.")


if __name__ == "__main__":
    main()

