"""
BUPA Liver Disorders Dataset - Exploratory Data Analysis (EDA)
This script performs comprehensive EDA on the cleaned liver dataset.
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
DATA_DIR = BASE_DIR / 'datasets' / 'liver'
REPORTS_DIR = BASE_DIR / 'reports' / 'liver'
PLOTS_DIR = REPORTS_DIR / 'eda' / 'plots'

# Ensure directories exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class LiverEDA:
    """Class to perform EDA on liver dataset"""
    
    def __init__(self):
        self.df = None
        self.eda_findings = []
        
    def load_cleaned_dataset(self):
        """Load the cleaned dataset"""
        print("\n" + "="*80)
        print("LOADING CLEANED DATASET")
        print("="*80)
        
        filepath = DATA_DIR / 'liver_cleaned.csv'
        if filepath.exists():
            self.df = pd.read_csv(filepath)
            print(f"[OK] Loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
            return True
        else:
            print(f"[ERROR] File not found: {filepath}")
            print("Please run data_cleaning_liver.py first")
            return False
    
    def perform_eda(self):
        """Perform comprehensive EDA"""
        if self.df is None:
            print("[ERROR] No dataset loaded")
            return
        
        print("\n" + "="*80)
        print("PERFORMING EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        self._dataset_overview()
        self._target_analysis()
        self._numerical_analysis()
        self._correlation_analysis()
        self._clinical_analysis()
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
        """Analyze target variable (selector)"""
        print("\n2. Target Variable Analysis (Selector)")
        print("-" * 70)
        
        if 'selector' in self.df.columns:
            dist = self.df['selector'].value_counts().sort_index()
            print(f"\n   Selector Distribution:")
            for val, count in dist.items():
                pct = count / len(self.df) * 100
                print(f"   - Selector {val}: {count} ({pct:.1f}%)")
            
            # Save distribution
            dist_df = pd.DataFrame({
                'Selector': dist.index,
                'Count': dist.values,
                'Percentage': (dist.values / len(self.df) * 100).round(2)
            })
            dist_df.to_csv(REPORTS_DIR / 'selector_distribution.csv', index=False)
    
    def _numerical_analysis(self):
        """Analyze numerical features"""
        print("\n3. Numerical Features Analysis")
        print("-" * 70)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"   Number of numerical features: {len(numeric_cols)}")
        
        # Descriptive statistics
        stats = self.df[numeric_cols].describe()
        stats.to_csv(REPORTS_DIR / 'detailed_statistics.csv')
        print(f"\n   Summary Statistics:")
        print(stats.to_string())
    
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
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    strong_corr.append({
                        'Feature_1': corr_matrix.columns[i],
                        'Feature_2': corr_matrix.columns[j],
                        'Correlation': round(corr_matrix.iloc[i, j], 3)
                    })
        
        if strong_corr:
            strong_corr_df = pd.DataFrame(strong_corr)
            strong_corr_df.to_csv(REPORTS_DIR / 'strong_correlations.csv', index=False)
            print(f"   [OK] Found {len(strong_corr)} strong correlations (|r| > 0.5)")
            print(f"\n   Top correlations:")
            for item in strong_corr[:5]:
                print(f"   - {item['Feature_1']} <-> {item['Feature_2']}: {item['Correlation']}")
        
        # Selector correlations
        if 'selector' in self.df.columns:
            selector_corr = corr_matrix['selector'].sort_values(ascending=False)
            selector_corr_df = pd.DataFrame({
                'Feature': selector_corr.index,
                'Correlation': selector_corr.values
            })
            selector_corr_df.to_csv(REPORTS_DIR / 'selector_correlations.csv', index=False)
            print(f"\n   Top features correlated with selector:")
            for feat, corr in selector_corr.head(5).items():
                if feat != 'selector':
                    print(f"   - {feat}: {corr:.3f}")
    
    def _clinical_analysis(self):
        """Perform clinical analysis"""
        print("\n5. Clinical Analysis")
        print("-" * 70)
        
        # AST/ALT ratio analysis
        if 'ast_alt_ratio' in self.df.columns:
            high_ratio = (self.df['ast_alt_ratio'] > 2).sum()
            print(f"   AST/ALT Ratio > 2: {high_ratio} ({high_ratio/len(self.df)*100:.1f}%)")
            print(f"   (May indicate alcoholic liver disease)")
        
        # Alcohol consumption analysis
        if 'drinks' in self.df.columns:
            print(f"\n   Alcohol Consumption:")
            print(f"   - Mean: {self.df['drinks'].mean():.2f} drinks/day")
            print(f"   - Median: {self.df['drinks'].median():.2f} drinks/day")
            print(f"   - Non-drinkers: {(self.df['drinks'] == 0).sum()} ({(self.df['drinks'] == 0).sum()/len(self.df)*100:.1f}%)")
            print(f"   - Heavy drinkers (>2/day): {(self.df['drinks'] > 2).sum()} ({(self.df['drinks'] > 2).sum()/len(self.df)*100:.1f}%)")
        
        # Enzyme elevation analysis
        enzyme_thresholds = {
            'sgpt': 56,  # ALT
            'sgot': 40,  # AST
            'gammagt': 48,
            'alkphos': 120
        }
        
        print(f"\n   Elevated Enzyme Levels:")
        for enzyme, threshold in enzyme_thresholds.items():
            if enzyme in self.df.columns:
                elevated = (self.df[enzyme] > threshold).sum()
                print(f"   - {enzyme} > {threshold}: {elevated} ({elevated/len(self.df)*100:.1f}%)")
    
    def _generate_visualizations(self):
        """Generate all visualizations"""
        print("\n6. Generating Visualizations")
        print("-" * 70)
        
        self._plot_selector_distribution()
        self._plot_correlation_heatmap()
        self._plot_numerical_distributions()
        self._plot_boxplots()
        self._plot_features_by_selector()
        self._plot_alcohol_analysis()
        self._plot_ast_alt_ratio_analysis()
        self._plot_pairplot()
        
        print("   [OK] All visualizations generated")
    
    def _plot_selector_distribution(self):
        """Plot selector distribution"""
        if 'selector' not in self.df.columns:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot
        dist = self.df['selector'].value_counts().sort_index()
        ax1.bar(dist.index, dist.values, color=['steelblue', 'coral'])
        ax1.set_xlabel('Selector', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.set_title('Selector Distribution - Bar Chart', fontsize=14, fontweight='bold')
        ax1.set_xticks([1, 2])
        
        # Pie chart
        ax2.pie(dist.values, labels=[f'Selector {i}' for i in dist.index], 
                autopct='%1.1f%%', colors=['steelblue', 'coral'], startangle=90)
        ax2.set_title('Selector Distribution - Pie Chart', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'selector_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - Selector distribution saved")
    
    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        plt.figure(figsize=(12, 10))
        corr = self.df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Correlation Heatmap - Liver Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - Correlation heatmap saved")
    
    def _plot_numerical_distributions(self):
        """Plot distributions of numerical features"""
        numeric_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                       if col != 'selector']
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            sns.histplot(data=self.df, x=col, kde=True, ax=axes[idx], color='skyblue')
            axes[idx].axvline(self.df[col].mean(), color='red', linestyle='--', label='Mean')
            axes[idx].axvline(self.df[col].median(), color='green', linestyle='--', label='Median')
            axes[idx].set_title(col, fontsize=12, fontweight='bold')
            axes[idx].legend()
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Numerical Feature Distributions - Liver Dataset',
                     fontsize=16, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - Numerical distributions saved")
    
    def _plot_boxplots(self):
        """Plot boxplots for outlier detection"""
        numeric_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                       if col != 'selector']
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            sns.boxplot(y=self.df[col], ax=axes[idx], color='lightcoral')
            axes[idx].set_title(col, fontsize=12, fontweight='bold')
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Boxplots for Outlier Detection - Liver Dataset',
                     fontsize=16, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - Boxplots saved")
    
    def _plot_features_by_selector(self):
        """Plot features grouped by selector"""
        if 'selector' not in self.df.columns:
            return
        
        numeric_cols = [col for col in self.df.select_dtypes(include=[np.number]).columns 
                       if col != 'selector']
        
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if len(numeric_cols) > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            sns.boxplot(x='selector', y=col, data=self.df, ax=axes[idx], palette='Set2')
            axes[idx].set_title(f'{col} by Selector', fontsize=10, fontweight='bold')
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Features by Selector - Liver Dataset',
                     fontsize=16, fontweight='bold', y=1.001)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'features_by_selector.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - Features by selector saved")
    
    def _plot_alcohol_analysis(self):
        """Plot alcohol consumption analysis"""
        if 'drinks' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Distribution
        sns.histplot(data=self.df, x='drinks', kde=True, ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('Alcohol Consumption Distribution', fontsize=12, fontweight='bold')
        
        # By selector
        if 'selector' in self.df.columns:
            sns.boxplot(x='selector', y='drinks', data=self.df, ax=axes[0, 1], palette='Set3')
            axes[0, 1].set_title('Alcohol Consumption by Selector', fontsize=12, fontweight='bold')
        
        # Category distribution
        if 'alcohol_category' in self.df.columns:
            cat_dist = self.df['alcohol_category'].value_counts()
            axes[1, 0].bar(range(len(cat_dist)), cat_dist.values, color='coral')
            axes[1, 0].set_xticks(range(len(cat_dist)))
            axes[1, 0].set_xticklabels(cat_dist.index, rotation=45)
            axes[1, 0].set_title('Alcohol Category Distribution', fontsize=12, fontweight='bold')
        
        # Correlation with enzymes
        if all(col in self.df.columns for col in ['drinks', 'gammagt']):
            axes[1, 1].scatter(self.df['drinks'], self.df['gammagt'], alpha=0.5, color='green')
            axes[1, 1].set_xlabel('Drinks per day')
            axes[1, 1].set_ylabel('Gamma-GT')
            axes[1, 1].set_title('Alcohol vs Gamma-GT', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'alcohol_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - Alcohol analysis saved")
    
    def _plot_ast_alt_ratio_analysis(self):
        """Plot AST/ALT ratio analysis"""
        if 'ast_alt_ratio' not in self.df.columns:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Distribution
        sns.histplot(data=self.df, x='ast_alt_ratio', kde=True, ax=axes[0], color='purple')
        axes[0].axvline(2, color='red', linestyle='--', label='Threshold (2.0)')
        axes[0].set_title('AST/ALT Ratio Distribution', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        # By selector
        if 'selector' in self.df.columns:
            sns.boxplot(x='selector', y='ast_alt_ratio', data=self.df, ax=axes[1], palette='Set1')
            axes[1].axhline(2, color='red', linestyle='--', label='Threshold (2.0)')
            axes[1].set_title('AST/ALT Ratio by Selector', fontsize=12, fontweight='bold')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / 'ast_alt_ratio_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("      - AST/ALT ratio analysis saved")
    
    def _plot_pairplot(self):
        """Plot pairplot for key features"""
        key_features = ['mcv', 'sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
        available_features = [f for f in key_features if f in self.df.columns]
        
        if len(available_features) > 2:
            plt.figure(figsize=(15, 15))
            sns.pairplot(self.df[available_features], hue='selector', 
                        palette='Set1', diag_kind='kde', plot_kws={'alpha': 0.6})
            plt.suptitle('Pairplot of Key Features - Liver Dataset',
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
        
        if 'selector' in self.df.columns:
            summary['Selector_1_Count'] = (self.df['selector'] == 1).sum()
            summary['Selector_2_Count'] = (self.df['selector'] == 2).sum()
        
        if 'drinks' in self.df.columns:
            summary['Mean_Drinks_Per_Day'] = round(self.df['drinks'].mean(), 2)
            summary['Non_Drinkers'] = (self.df['drinks'] == 0).sum()
        
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(REPORTS_DIR / 'eda_summary.csv', index=False)
        print(f"\n[OK] EDA summary saved")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("BUPA LIVER DISORDERS DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    eda = LiverEDA()
    
    if eda.load_cleaned_dataset():
        eda.perform_eda()
        
        print("\n" + "="*80)
        print("EDA COMPLETE")
        print("="*80)
        print(f"[OK] Reports saved to: {REPORTS_DIR}")
        print(f"[OK] Plots saved to: {PLOTS_DIR}")
        print("\nNext steps:")
        print("1. Review the EDA reports and visualizations")
        print("2. Run feature_engineering_liver.py to create advanced features")
        print("="*80 + "\n")
    else:
        print("\n[ERROR] Failed to load dataset. Please run data_cleaning_liver.py first.")


if __name__ == "__main__":
    main()


