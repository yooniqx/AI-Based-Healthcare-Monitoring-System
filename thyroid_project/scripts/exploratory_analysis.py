"""
Thyroid Disease Dataset - Exploratory Data Analysis (EDA)
This script performs comprehensive EDA on the cleaned thyroid datasets.
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
OUTPUT_DIR = BASE_DIR / 'output'
REPORTS_DIR = BASE_DIR / 'reports'
PLOTS_DIR = BASE_DIR / 'plots'

# Ensure directories exist
PLOTS_DIR.mkdir(exist_ok=True)


class ThyroidEDA:
    """Class to perform EDA on thyroid datasets"""
    
    def __init__(self):
        self.datasets = {}
        self.eda_findings = []
        
    def load_cleaned_datasets(self):
        """Load all cleaned datasets"""
        print("\n" + "="*80)
        print("LOADING CLEANED DATASETS")
        print("="*80)
        
        dataset_files = {
            'hypothyroid': 'hypothyroid_cleaned.csv',
            'ann_thyroid': 'ann_thyroid_cleaned.csv',
            'new_thyroid': 'new_thyroid_cleaned.csv'
        }
        
        for name, filename in dataset_files.items():
            filepath = OUTPUT_DIR / filename
            if filepath.exists():
                self.datasets[name] = pd.read_csv(filepath)
                print(f"[OK] Loaded {name}: {len(self.datasets[name])} rows, {len(self.datasets[name].columns)} columns")
            else:
                print(f"[ERROR] File not found: {filename}")
        
        print(f"\n[OK] Loaded {len(self.datasets)} datasets")
        
    def analyze_dataset_overview(self):
        """Generate overview statistics for all datasets"""
        print("\n" + "="*80)
        print("DATASET OVERVIEW ANALYSIS")
        print("="*80)
        
        overview_data = []
        
        for name, df in self.datasets.items():
            overview = {
                'Dataset': name,
                'Rows': len(df),
                'Columns': len(df.columns),
                'Memory_Usage_MB': df.memory_usage(deep=True).sum() / 1024**2,
                'Numerical_Features': len(df.select_dtypes(include=[np.number]).columns),
                'Categorical_Features': len(df.select_dtypes(include=['object']).columns),
                'Missing_Values': df.isnull().sum().sum(),
                'Duplicate_Rows': df.duplicated().sum()
            }
            overview_data.append(overview)
            
            print(f"\n{name.upper()}:")
            for key, value in overview.items():
                if key != 'Dataset':
                    if isinstance(value, float):
                        print(f"   {key}: {value:.2f}")
                    else:
                        print(f"   {key}: {value}")
        
        # Save overview report
        overview_df = pd.DataFrame(overview_data)
        report_path = REPORTS_DIR / 'eda_overview.csv'
        overview_df.to_csv(report_path, index=False)
        print(f"\n[OK] Overview report saved to {report_path.name}")
        
    def analyze_class_distribution(self):
        """Analyze and visualize class distributions"""
        print("\n" + "="*80)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*80)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Class Distribution Across Datasets', fontsize=16, fontweight='bold')
        
        for idx, (name, df) in enumerate(self.datasets.items()):
            if 'class' in df.columns:
                class_counts = df['class'].value_counts()
                class_pcts = df['class'].value_counts(normalize=True) * 100
                
                print(f"\n{name.upper()}:")
                for cls in class_counts.index:
                    print(f"   {cls}: {class_counts[cls]} ({class_pcts[cls]:.2f}%)")
                
                # Plot
                ax = axes[idx]
                class_counts.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c', '#3498db'])
                ax.set_title(f'{name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Class', fontsize=10)
                ax.set_ylabel('Count', fontsize=10)
                ax.tick_params(axis='x', rotation=45)
                
                # Add percentage labels on bars
                for i, (count, pct) in enumerate(zip(class_counts, class_pcts)):
                    ax.text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_path = PLOTS_DIR / 'class_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n[OK] Class distribution plot saved to {plot_path.name}")
        
    def analyze_numerical_features(self):
        """Analyze numerical features with statistics and distributions"""
        print("\n" + "="*80)
        print("NUMERICAL FEATURES ANALYSIS")
        print("="*80)
        
        for name, df in self.datasets.items():
            print(f"\n{name.upper()}:")
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'class' in numerical_cols:
                numerical_cols.remove('class')
            
            if not numerical_cols:
                print("   No numerical features found")
                continue
            
            print(f"   Analyzing {len(numerical_cols)} numerical features")
            
            # Statistical summary
            stats = df[numerical_cols].describe().T
            stats['skewness'] = df[numerical_cols].skew()
            stats['kurtosis'] = df[numerical_cols].kurtosis()
            
            # Save statistics
            stats_path = REPORTS_DIR / f'{name}_numerical_statistics.csv'
            stats.to_csv(stats_path)
            print(f"   [OK] Statistics saved to {stats_path.name}")
            
            # Create distribution plots
            n_cols = min(4, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
            fig.suptitle(f'{name.replace("_", " ").title()} - Numerical Feature Distributions', 
                        fontsize=16, fontweight='bold')
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, col in enumerate(numerical_cols):
                row = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row, col_idx]
                
                df[col].hist(bins=30, ax=ax, color='#3498db', edgecolor='black', alpha=0.7)
                ax.set_title(col, fontsize=10, fontweight='bold')
                ax.set_xlabel('Value', fontsize=9)
                ax.set_ylabel('Frequency', fontsize=9)
                
                # Add mean and median lines
                mean_val = df[col].mean()
                median_val = df[col].median()
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
                ax.legend(fontsize=8)
            
            # Hide empty subplots
            for idx in range(len(numerical_cols), n_rows * n_cols):
                row = idx // n_cols
                col_idx = idx % n_cols
                axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            plot_path = PLOTS_DIR / f'{name}_numerical_distributions.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [OK] Distribution plots saved to {plot_path.name}")
    
    def analyze_correlations(self):
        """Analyze correlations between numerical features"""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        for name, df in self.datasets.items():
            print(f"\n{name.upper()}:")
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) < 2:
                print("   Not enough numerical features for correlation analysis")
                continue
            
            # Calculate correlation matrix
            corr_matrix = df[numerical_cols].corr()
            
            # Save correlation matrix
            corr_path = REPORTS_DIR / f'{name}_correlation_matrix.csv'
            corr_matrix.to_csv(corr_path)
            print(f"   [OK] Correlation matrix saved to {corr_path.name}")
            
            # Create correlation heatmap
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                       cmap='coolwarm', center=0, square=True, linewidths=1,
                       cbar_kws={"shrink": 0.8})
            plt.title(f'{name.replace("_", " ").title()} - Feature Correlations', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            plot_path = PLOTS_DIR / f'{name}_correlation_heatmap.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [OK] Correlation heatmap saved to {plot_path.name}")
            
            # Find strong correlations
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        strong_corr.append({
                            'Feature_1': corr_matrix.columns[i],
                            'Feature_2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j]
                        })
            
            if strong_corr:
                print(f"   Found {len(strong_corr)} strong correlations (|r| > 0.7):")
                for corr in strong_corr:
                    print(f"      {corr['Feature_1']} <-> {corr['Feature_2']}: {corr['Correlation']:.3f}")
    
    def analyze_feature_by_class(self):
        """Analyze how features vary by class"""
        print("\n" + "="*80)
        print("FEATURE BY CLASS ANALYSIS")
        print("="*80)
        
        for name, df in self.datasets.items():
            if 'class' not in df.columns:
                continue
            
            print(f"\n{name.upper()}:")
            
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'class' in numerical_cols:
                numerical_cols.remove('class')
            
            if not numerical_cols:
                print("   No numerical features to analyze")
                continue
            
            # Select top features for visualization (max 6)
            top_features = numerical_cols[:6]
            
            n_cols = 3
            n_rows = (len(top_features) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
            fig.suptitle(f'{name.replace("_", " ").title()} - Features by Class', 
                        fontsize=16, fontweight='bold')
            
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            
            for idx, col in enumerate(top_features):
                row = idx // n_cols
                col_idx = idx % n_cols
                ax = axes[row, col_idx]
                
                df.boxplot(column=col, by='class', ax=ax)
                ax.set_title(col, fontsize=11, fontweight='bold')
                ax.set_xlabel('Class', fontsize=10)
                ax.set_ylabel('Value', fontsize=10)
                plt.sca(ax)
                plt.xticks(rotation=45)
            
            # Hide empty subplots
            for idx in range(len(top_features), n_rows * n_cols):
                row = idx // n_cols
                col_idx = idx % n_cols
                axes[row, col_idx].axis('off')
            
            plt.tight_layout()
            plot_path = PLOTS_DIR / f'{name}_features_by_class.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   [OK] Feature by class plots saved to {plot_path.name}")
    
    def generate_eda_summary_report(self):
        """Generate comprehensive EDA summary report"""
        print("\n" + "="*80)
        print("GENERATING EDA SUMMARY REPORT")
        print("="*80)
        
        summary = []
        
        for name, df in self.datasets.items():
            # Basic info
            info = {
                'Dataset': name,
                'Total_Records': len(df),
                'Total_Features': len(df.columns),
                'Numerical_Features': len(df.select_dtypes(include=[np.number]).columns),
                'Categorical_Features': len(df.select_dtypes(include=['object']).columns)
            }
            
            # Class info
            if 'class' in df.columns:
                class_counts = df['class'].value_counts()
                info['Classes'] = len(class_counts)
                info['Majority_Class'] = class_counts.index[0]
                info['Majority_Class_Pct'] = (class_counts.iloc[0] / len(df)) * 100
                info['Class_Imbalance_Ratio'] = class_counts.iloc[0] / class_counts.iloc[-1]
            
            # Numerical feature stats
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'class' in numerical_cols:
                numerical_cols.remove('class')
            
            if numerical_cols:
                info['Avg_Feature_Mean'] = df[numerical_cols].mean().mean()
                info['Avg_Feature_Std'] = df[numerical_cols].std().mean()
                info['Avg_Skewness'] = df[numerical_cols].skew().mean()
            
            summary.append(info)
        
        # Save summary
        summary_df = pd.DataFrame(summary)
        report_path = REPORTS_DIR / 'eda_summary_report.csv'
        summary_df.to_csv(report_path, index=False)
        print(f"\n[OK] EDA summary report saved to {report_path.name}")
        
        # Print summary
        print("\nEDA SUMMARY:")
        for _, row in summary_df.iterrows():
            dataset_name = str(row['Dataset']).upper()
            print(f"\n{dataset_name}:")
            for col, val in row.items():
                if col != 'Dataset':
                    if isinstance(val, float):
                        print(f"   {col}: {val:.2f}")
                    else:
                        print(f"   {col}: {val}")


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("THYROID DISEASE DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    
    # Initialize EDA
    eda = ThyroidEDA()
    
    try:
        # Load datasets
        eda.load_cleaned_datasets()
        
        if not eda.datasets:
            print("\n[ERROR] No datasets loaded. Please run data_cleaning.py first.")
            return
        
        # Perform analyses
        eda.analyze_dataset_overview()
        eda.analyze_class_distribution()
        eda.analyze_numerical_features()
        eda.analyze_correlations()
        eda.analyze_feature_by_class()
        eda.generate_eda_summary_report()
        
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nReports saved to: {REPORTS_DIR}")
        print(f"Plots saved to: {PLOTS_DIR}")
        print("\nKey Findings:")
        print("1. Review class distribution plots for imbalance issues")
        print("2. Check correlation heatmaps for multicollinearity")
        print("3. Examine feature distributions for normality")
        print("4. Analyze feature-by-class plots for discriminative power")
        print("\nNext steps:")
        print("1. Feature engineering based on EDA insights")
        print("2. Handle class imbalance if needed")
        print("3. Feature selection/dimensionality reduction")
        print("4. Prepare data for modeling")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] EDA failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
