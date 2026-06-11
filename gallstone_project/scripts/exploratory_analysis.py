"""
Exploratory Data Analysis (EDA) for Gallstone Dataset
=====================================================

This script performs comprehensive exploratory data analysis including:
- Statistical summaries
- Distribution analysis
- Correlation analysis
- Visualizations
- Feature relationships
- Data quality assessment

Author: Data Analysis Team
Date: June 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class GallstoneEDA:
    """Comprehensive EDA for Gallstone Dataset"""
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = Path(data_path)
        self.df: pd.DataFrame = None  # type: ignore
        self.numerical_cols = []
        self.categorical_cols = []
        self.target_col = 'gallstone_status'
        
        # Create output directories
        self.plots_dir = Path('gallstone_project/plots')
        self.reports_dir = Path('gallstone_project/reports')
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load the cleaned dataset"""
        print("=" * 80)
        print("LOADING GALLSTONE DATASET")
        print("=" * 80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\n[OK] Dataset loaded successfully")
        print(f"  - Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        
        # Identify column types
        self.numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from numerical if present
        if self.target_col in self.numerical_cols:
            self.numerical_cols.remove(self.target_col)
        
        print(f"  - Numerical features: {len(self.numerical_cols)}")
        print(f"  - Categorical features: {len(self.categorical_cols)}")
        print(f"  - Target variable: {self.target_col}")
        
        return self.df
    
    def basic_statistics(self):
        """Generate basic statistical summaries"""
        print("\n" + "=" * 80)
        print("BASIC STATISTICS")
        print("=" * 80)
        
        # Overall summary
        print("\n1. Dataset Overview:")
        print(f"   - Total records: {len(self.df)}")
        print(f"   - Total features: {len(self.df.columns)}")
        print(f"   - Memory usage: {self.df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Missing values check
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print(f"   - Missing values: None (100% complete)")
        else:
            print(f"   - Missing values: {missing.sum()} total")
        
        # Duplicates check
        duplicates = self.df.duplicated().sum()
        print(f"   - Duplicate rows: {duplicates}")
        
        # Target distribution
        print(f"\n2. Target Variable Distribution ({self.target_col}):")
        target_counts = self.df[self.target_col].value_counts()
        for value, count in target_counts.items():
            percentage = (count / len(self.df)) * 100
            status = "No Gallstones" if value == 0 else "Gallstones Present"
            print(f"   - {status} ({value}): {count} ({percentage:.1f}%)")
        
        # Numerical statistics
        print("\n3. Numerical Features Summary:")
        stats_df = self.df[self.numerical_cols].describe().T
        stats_df['range'] = stats_df['max'] - stats_df['min']
        stats_df['cv'] = (stats_df['std'] / stats_df['mean']) * 100  # Coefficient of variation
        
        # Save detailed statistics
        stats_df.to_csv(self.reports_dir / 'detailed_statistics.csv')
        print(f"   [OK] Detailed statistics saved to: {self.reports_dir / 'detailed_statistics.csv'}")
        
        # Display key statistics for first few features
        print("\n   Key Statistics (first 5 features):")
        display_cols = ['mean', 'std', 'min', 'max', 'range']
        print(stats_df[display_cols].head().to_string())
        
        return stats_df
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        # Calculate correlation matrix
        corr_matrix = self.df[self.numerical_cols].corr()
        
        # Save correlation matrix
        corr_matrix.to_csv(self.reports_dir / 'correlation_matrix.csv')
        print(f"\n[OK] Correlation matrix saved to: {self.reports_dir / 'correlation_matrix.csv'}")
        
        # Find strong correlations with target
        if self.target_col in self.df.columns:
            target_corr = self.df[self.numerical_cols + [self.target_col]].corr()[self.target_col].drop(self.target_col)
            target_corr_sorted = target_corr.abs().sort_values(ascending=False)
            
            print(f"\n1. Top 10 Features Correlated with {self.target_col}:")
            for i, (feature, corr_value) in enumerate(target_corr_sorted.head(10).items(), 1):
                actual_corr = target_corr[feature]
                print(f"   {i:2d}. {feature:40s}: {actual_corr:7.4f}")
            
            # Save target correlations
            target_corr_df = pd.DataFrame({
                'feature': target_corr.index,
                'correlation': target_corr.values,
                'abs_correlation': target_corr.abs().values
            }).sort_values('abs_correlation', ascending=False)
            target_corr_df.to_csv(self.reports_dir / 'target_correlations.csv', index=False)
        
        # Find strong inter-feature correlations
        print("\n2. Strong Inter-Feature Correlations (|r| > 0.7):")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    strong_corr.append({
                        'feature_1': corr_matrix.columns[i],
                        'feature_2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        if strong_corr:
            strong_corr_df = pd.DataFrame(strong_corr).sort_values('correlation', ascending=False)
            print(strong_corr_df.to_string(index=False))
            strong_corr_df.to_csv(self.reports_dir / 'strong_correlations.csv', index=False)
        else:
            print("   No strong correlations found (|r| > 0.7)")
        
        # Create correlation heatmap
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Correlation heatmap saved to: {self.plots_dir / 'correlation_heatmap.png'}")
        
        return corr_matrix
    
    def distribution_analysis(self):
        """Analyze distributions of numerical features"""
        print("\n" + "=" * 80)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        # Create distribution plots for all numerical features
        n_cols = 4
        n_rows = (len(self.numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(self.numerical_cols):
            ax = axes[idx]
            
            # Histogram only (skip KDE for small datasets or constant values)
            n_bins = min(10, len(self.df[col].unique()))
            self.df[col].hist(bins=n_bins if n_bins > 1 else 5, ax=ax, alpha=0.7, edgecolor='black', color='steelblue')
            
            # Try to add KDE only if data has variance
            if self.df[col].std() > 0 and len(self.df[col].unique()) > 2:
                try:
                    ax2 = ax.twinx()
                    self.df[col].plot(kind='kde', ax=ax2, color='red', linewidth=2)
                    ax2.set_ylabel('Density', fontsize=9)
                except:
                    pass  # Skip KDE if it fails
            
            ax.set_xlabel(col, fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.set_title(f'{col}\n(Mean: {self.df[col].mean():.2f}, Std: {self.df[col].std():.2f})',
                        fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(len(self.numerical_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'numerical_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Distribution plots saved to: {self.plots_dir / 'numerical_distributions.png'}")
        
        # Normality tests and skewness
        print("\n1. Distribution Characteristics:")
        dist_stats = []
        for col in self.numerical_cols[:10]:  # Show first 10
            skewness = self.df[col].skew()
            kurtosis = self.df[col].kurtosis()
            dist_stats.append({
                'feature': col,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'distribution': 'Normal' if abs(skewness) < 0.5 else 'Skewed'
            })
        
        dist_df = pd.DataFrame(dist_stats)
        print(dist_df.to_string(index=False))
        
        # Save all distribution statistics
        all_dist_stats = []
        for col in self.numerical_cols:
            all_dist_stats.append({
                'feature': col,
                'skewness': self.df[col].skew(),
                'kurtosis': self.df[col].kurtosis()
            })
        pd.DataFrame(all_dist_stats).to_csv(self.reports_dir / 'distribution_statistics.csv', index=False)
        
        return dist_df
    
    def boxplot_analysis(self):
        """Create boxplots to identify outliers"""
        print("\n" + "=" * 80)
        print("BOXPLOT ANALYSIS (Outlier Detection)")
        print("=" * 80)
        
        # Create boxplots
        n_cols = 4
        n_rows = (len(self.numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        outlier_summary = []
        
        for idx, col in enumerate(self.numerical_cols):
            ax = axes[idx]
            
            # Create boxplot
            bp = ax.boxplot(self.df[col].dropna(), vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            # Calculate outliers using IQR method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
            n_outliers = len(outliers)
            
            outlier_summary.append({
                'feature': col,
                'n_outliers': n_outliers,
                'outlier_percentage': (n_outliers / len(self.df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
            
            ax.set_ylabel(col, fontsize=9)
            ax.set_title(f'{col}\n(Outliers: {n_outliers})', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide extra subplots
        for idx in range(len(self.numerical_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'boxplots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Boxplots saved to: {self.plots_dir / 'boxplots.png'}")
        
        # Display outlier summary
        outlier_df = pd.DataFrame(outlier_summary).sort_values('n_outliers', ascending=False)
        print("\n1. Outlier Summary (Top 10 features with most outliers):")
        print(outlier_df.head(10).to_string(index=False))
        
        outlier_df.to_csv(self.reports_dir / 'outlier_analysis.csv', index=False)
        print(f"\n[OK] Outlier analysis saved to: {self.reports_dir / 'outlier_analysis.csv'}")
        
        return outlier_df
    
    def target_analysis(self):
        """Analyze features by target variable"""
        print("\n" + "=" * 80)
        print("TARGET VARIABLE ANALYSIS")
        print("=" * 80)
        
        if self.target_col not in self.df.columns:
            print("Target variable not found in dataset")
            return
        
        # Compare features by target
        print(f"\n1. Feature Comparison by {self.target_col}:")
        
        comparison_data = []
        for col in self.numerical_cols[:15]:  # Show first 15
            group_0 = self.df[self.df[self.target_col] == 0][col]
            group_1 = self.df[self.df[self.target_col] == 1][col]
            
            comparison_data.append({
                'feature': col,
                'no_gallstone_mean': group_0.mean(),
                'gallstone_mean': group_1.mean(),
                'difference': group_1.mean() - group_0.mean(),
                'percent_diff': ((group_1.mean() - group_0.mean()) / group_0.mean() * 100) if group_0.mean() != 0 else 0
            })
        
        comparison_df = pd.DataFrame(comparison_data).sort_values('percent_diff', key=abs, ascending=False)
        print(comparison_df.to_string(index=False))
        
        # Save full comparison
        full_comparison = []
        for col in self.numerical_cols:
            group_0 = self.df[self.df[self.target_col] == 0][col]
            group_1 = self.df[self.df[self.target_col] == 1][col]
            
            full_comparison.append({
                'feature': col,
                'no_gallstone_mean': group_0.mean(),
                'no_gallstone_std': group_0.std(),
                'gallstone_mean': group_1.mean(),
                'gallstone_std': group_1.std(),
                'difference': group_1.mean() - group_0.mean()
            })
        
        pd.DataFrame(full_comparison).to_csv(self.reports_dir / 'target_comparison.csv', index=False)
        
        # Create comparison plots for top features
        top_features = comparison_df.head(12)['feature'].tolist()
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for idx, col in enumerate(top_features):
            ax = axes[idx]
            
            # Box plot by target
            data_to_plot = [
                self.df[self.df[self.target_col] == 0][col].dropna(),
                self.df[self.df[self.target_col] == 1][col].dropna()
            ]
            
            bp = ax.boxplot(data_to_plot, labels=['No Gallstone', 'Gallstone'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            ax.set_ylabel(col, fontsize=9)
            ax.set_title(col, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'features_by_target.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Target comparison plots saved to: {self.plots_dir / 'features_by_target.png'}")
        
        return comparison_df
    
    def pairplot_analysis(self):
        """Create pairplot for key features"""
        print("\n" + "=" * 80)
        print("PAIRPLOT ANALYSIS")
        print("=" * 80)
        
        # Select top correlated features with target
        if self.target_col in self.df.columns:
            target_corr = self.df[self.numerical_cols + [self.target_col]].corr()[self.target_col].drop(self.target_col)
            top_features = target_corr.abs().sort_values(ascending=False).head(5).index.tolist()
        else:
            top_features = self.numerical_cols[:5]
        
        print(f"\nCreating pairplot for top 5 features: {', '.join(top_features)}")
        
        # Create pairplot
        plot_data = self.df[top_features + [self.target_col]].copy()
        plot_data[self.target_col] = plot_data[self.target_col].map({0: 'No Gallstone', 1: 'Gallstone'})
        
        pairplot = sns.pairplot(plot_data, hue=self.target_col, palette={'No Gallstone': 'green', 'Gallstone': 'red'},
                               diag_kind='kde', plot_kws={'alpha': 0.6}, height=3)
        pairplot.fig.suptitle('Pairplot of Top 5 Features by Gallstone Status', y=1.02, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'pairplot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Pairplot saved to: {self.plots_dir / 'pairplot.png'}")
        
        return top_features
    
    def generate_summary_report(self):
        """Generate comprehensive EDA summary report"""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)
        
        summary = {
            'Dataset Overview': {
                'Total Records': len(self.df),
                'Total Features': len(self.df.columns),
                'Numerical Features': len(self.numerical_cols),
                'Categorical Features': len(self.categorical_cols),
                'Missing Values': self.df.isnull().sum().sum(),
                'Duplicate Rows': self.df.duplicated().sum(),
                'Memory Usage (KB)': f"{self.df.memory_usage(deep=True).sum() / 1024:.2f}"
            },
            'Target Distribution': {
                'No Gallstones (0)': int(self.df[self.target_col].value_counts().get(0, 0)),
                'Gallstones (1)': int(self.df[self.target_col].value_counts().get(1, 0)),
                'Class Balance': f"{(self.df[self.target_col].value_counts().get(0, 0) / len(self.df) * 100):.1f}% / {(self.df[self.target_col].value_counts().get(1, 0) / len(self.df) * 100):.1f}%"
            },
            'Data Quality': {
                'Completeness': '100%' if self.df.isnull().sum().sum() == 0 else f"{((1 - self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100):.2f}%",
                'Uniqueness': '100%' if self.df.duplicated().sum() == 0 else f"{((1 - self.df.duplicated().sum() / len(self.df)) * 100):.2f}%"
            }
        }
        
        # Save summary
        summary_records = []
        for category, metrics in summary.items():
            for metric, value in metrics.items():
                summary_records.append({
                    'Category': category,
                    'Metric': metric,
                    'Value': value
                })
        
        summary_df = pd.DataFrame(summary_records)
        summary_df.to_csv(self.reports_dir / 'eda_summary.csv', index=False)
        print(f"\n[OK] EDA summary saved to: {self.reports_dir / 'eda_summary.csv'}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("EDA SUMMARY")
        print("=" * 80)
        for category, metrics in summary.items():
            print(f"\n{category}:")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value}")
        
        return summary_df
    
    def run_complete_eda(self):
        """Run complete EDA pipeline"""
        print("\n" + "=" * 80)
        print("GALLSTONE DATASET - EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        print(f"Start Time: {pd.Timestamp.now()}")
        
        # Load data
        self.load_data()
        
        # Run all analyses
        self.basic_statistics()
        self.correlation_analysis()
        self.distribution_analysis()
        self.boxplot_analysis()
        self.target_analysis()
        self.pairplot_analysis()
        self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("EDA COMPLETE!")
        print("=" * 80)
        print(f"\nAll outputs saved to:")
        print(f"  - Plots: {self.plots_dir}")
        print(f"  - Reports: {self.reports_dir}")
        print(f"\nEnd Time: {pd.Timestamp.now()}")
        print("\n" + "=" * 80)


def main():
    """Main execution function"""
    # Path to cleaned dataset (relative to script location)
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / 'ml_ready' / 'gallstone_cleaned.csv'
    
    # Initialize and run EDA
    eda = GallstoneEDA(data_path)
    eda.run_complete_eda()
    
    print("\n[OK] Exploratory Data Analysis completed successfully!")
    print("\nNext Steps:")
    print("  1. Review the generated plots in gallstone_project/plots/")
    print("  2. Examine the reports in gallstone_project/reports/")
    print("  3. Use insights for feature engineering and modeling")


if __name__ == "__main__":
    main()
