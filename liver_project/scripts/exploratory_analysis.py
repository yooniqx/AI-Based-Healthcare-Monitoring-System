"""
Exploratory Data Analysis Script for BUPA Liver Disorders Dataset
This script performs comprehensive EDA with visualizations and statistical analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from scipy import stats

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("="*80)
print("BUPA LIVER DISORDERS DATASET - EXPLORATORY DATA ANALYSIS")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load cleaned data
data_path = '../output/liver_cleaned.csv'
print(f"\nLoading cleaned data from: {data_path}")
df = pd.read_csv(data_path)
print(f"[OK] Loaded {len(df)} records with {len(df.columns)} features")

# Create output directories
os.makedirs('../plots', exist_ok=True)
os.makedirs('../reports', exist_ok=True)

# ============================================================================
# 1. DATASET OVERVIEW
# ============================================================================
print("\n" + "="*80)
print("1. DATASET OVERVIEW")
print("="*80)

print("\nDataset shape:", df.shape)
print("\nColumn names and types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())

# ============================================================================
# 2. TARGET VARIABLE ANALYSIS (SELECTOR)
# ============================================================================
print("\n" + "="*80)
print("2. TARGET VARIABLE ANALYSIS (SELECTOR)")
print("="*80)

selector_counts = df['selector'].value_counts().sort_index()
print("\nSelector distribution:")
print(selector_counts)
print(f"\nSelector 1: {selector_counts[1]} ({selector_counts[1]/len(df)*100:.1f}%)")
print(f"Selector 2: {selector_counts[2]} ({selector_counts[2]/len(df)*100:.1f}%)")

# Plot selector distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
selector_counts.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
axes[0].set_title('Selector Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Selector')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Selector 1', 'Selector 2'], rotation=0)

# Pie chart
axes[1].pie(selector_counts, labels=['Selector 1', 'Selector 2'], 
            autopct='%1.1f%%', colors=['#3498db', '#e74c3c'], startangle=90)
axes[1].set_title('Selector Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../plots/selector_distribution.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved: plots/selector_distribution.png")
plt.close()

# ============================================================================
# 3. NUMERICAL FEATURES DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print("3. NUMERICAL FEATURES DISTRIBUTION")
print("="*80)

# Select numerical features (excluding selector and categorical)
numerical_features = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 
                      'ast_alt_ratio', 'total_enzyme_score']

# Create distribution plots
fig, axes = plt.subplots(4, 2, figsize=(15, 16))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    # Histogram with KDE
    axes[idx].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[idx].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[col].mean():.2f}')
    axes[idx].axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[col].median():.2f}')
    axes[idx].set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/numerical_distributions.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved: plots/numerical_distributions.png")
plt.close()

# ============================================================================
# 4. BOX PLOTS FOR OUTLIER VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("4. BOX PLOTS FOR OUTLIER VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(4, 2, figsize=(15, 16))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    bp = axes[idx].boxplot(df[col], vert=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    axes[idx].set_title(f'{col} Box Plot', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(col)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/boxplots.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved: plots/boxplots.png")
plt.close()

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. CORRELATION ANALYSIS")
print("="*80)

# Calculate correlation matrix for numerical features
correlation_cols = ['mcv', 'alkphos', 'sgpt', 'sgot', 'gammagt', 'drinks', 
                    'ast_alt_ratio', 'total_enzyme_score', 'selector']
corr_matrix = df[correlation_cols].corr()  # type: ignore

print("\nCorrelation with selector:")
print(corr_matrix['selector'].sort_values(ascending=False))  # type: ignore

# Save correlation matrix
corr_matrix.to_csv('../reports/correlation_matrix.csv')
print("\n[OK] Saved: reports/correlation_matrix.csv")

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('../plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: plots/correlation_heatmap.png")
plt.close()

# ============================================================================
# 6. FEATURES BY SELECTOR
# ============================================================================
print("\n" + "="*80)
print("6. FEATURES BY SELECTOR")
print("="*80)

# Compare features between selector groups
fig, axes = plt.subplots(4, 2, figsize=(15, 16))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    selector1_data = df[df['selector'] == 1][col]
    selector2_data = df[df['selector'] == 2][col]
    
    axes[idx].hist([selector1_data, selector2_data], bins=20, alpha=0.7, 
                   label=['Selector 1', 'Selector 2'], color=['#3498db', '#e74c3c'])
    axes[idx].set_title(f'{col} by Selector', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/features_by_selector.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved: plots/features_by_selector.png")
plt.close()

# Statistical comparison
print("\nStatistical comparison between selector groups:")
comparison_results = []

for col in numerical_features:
    selector1_data = df[df['selector'] == 1][col]
    selector2_data = df[df['selector'] == 2][col]
    
    # T-test
    t_stat, p_value = stats.ttest_ind(selector1_data, selector2_data)
    p_val = float(p_value)  # type: ignore
    
    comparison_results.append({
        'Feature': col,
        'Selector_1_Mean': selector1_data.mean(),
        'Selector_2_Mean': selector2_data.mean(),
        'Difference': selector2_data.mean() - selector1_data.mean(),
        'T_Statistic': t_stat,
        'P_Value': p_val,
        'Significant': 'Yes' if p_val < 0.05 else 'No'
    })
    
    print(f"\n{col}:")
    print(f"  Selector 1 mean: {selector1_data.mean():.2f}")
    print(f"  Selector 2 mean: {selector2_data.mean():.2f}")
    print(f"  Difference: {selector2_data.mean() - selector1_data.mean():.2f}")
    print(f"  P-value: {p_val:.4f} {'(Significant)' if p_val < 0.05 else '(Not significant)'}")

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv('../reports/selector_comparison.csv', index=False)
print("\n[OK] Saved: reports/selector_comparison.csv")

# ============================================================================
# 7. ALCOHOL CONSUMPTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("7. ALCOHOL CONSUMPTION ANALYSIS")
print("="*80)

# Alcohol category distribution
alcohol_dist = df['alcohol_category'].value_counts()
print("\nAlcohol consumption categories:")
print(alcohol_dist)

# Plot alcohol categories
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar plot
alcohol_dist.plot(kind='bar', ax=axes[0], color='coral')
axes[0].set_title('Alcohol Consumption Categories', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45)

# Drinks distribution by selector
df.boxplot(column='drinks', by='selector', ax=axes[1])
axes[1].set_title('Drinks per Day by Selector', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Selector')
axes[1].set_ylabel('Drinks per Day')
plt.suptitle('')  # Remove default title

plt.tight_layout()
plt.savefig('../plots/alcohol_analysis.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved: plots/alcohol_analysis.png")
plt.close()

# ============================================================================
# 8. AST/ALT RATIO ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8. AST/ALT RATIO ANALYSIS")
print("="*80)

print(f"\nAST/ALT Ratio statistics:")
print(f"  Mean: {df['ast_alt_ratio'].mean():.2f}")
print(f"  Median: {df['ast_alt_ratio'].median():.2f}")
print(f"  Std: {df['ast_alt_ratio'].std():.2f}")
print(f"  Min: {df['ast_alt_ratio'].min():.2f}")
print(f"  Max: {df['ast_alt_ratio'].max():.2f}")

# Clinical significance: AST/ALT ratio > 2 may indicate alcoholic liver disease
high_ratio = (df['ast_alt_ratio'] > 2).sum()
print(f"\nRecords with AST/ALT ratio > 2: {high_ratio} ({high_ratio/len(df)*100:.1f}%)")

# Plot AST/ALT ratio
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution
axes[0].hist(df['ast_alt_ratio'], bins=30, alpha=0.7, color='purple', edgecolor='black')
axes[0].axvline(2, color='red', linestyle='--', linewidth=2, label='Clinical threshold (2.0)')
axes[0].set_title('AST/ALT Ratio Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('AST/ALT Ratio')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# By selector
df.boxplot(column='ast_alt_ratio', by='selector', ax=axes[1])
axes[1].set_title('AST/ALT Ratio by Selector', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Selector')
axes[1].set_ylabel('AST/ALT Ratio')
axes[1].axhline(2, color='red', linestyle='--', linewidth=2, label='Clinical threshold')
axes[1].legend()
plt.suptitle('')

plt.tight_layout()
plt.savefig('../plots/ast_alt_ratio_analysis.png', dpi=300, bbox_inches='tight')
print("\n[OK] Saved: plots/ast_alt_ratio_analysis.png")
plt.close()

# ============================================================================
# 9. PAIRPLOT FOR KEY FEATURES
# ============================================================================
print("\n" + "="*80)
print("9. PAIRPLOT FOR KEY FEATURES")
print("="*80)

# Select key features for pairplot
key_features = ['sgpt', 'sgot', 'gammagt', 'drinks', 'selector']
pairplot_df = df[key_features].copy()
pairplot_df['selector'] = pairplot_df['selector'].astype(str)

print("\nGenerating pairplot (this may take a moment)...")
pairplot = sns.pairplot(pairplot_df, hue='selector', palette=['#3498db', '#e74c3c'],  # type: ignore
                        diag_kind='kde', plot_kws={'alpha': 0.6})
pairplot.fig.suptitle('Pairplot of Key Features', y=1.02, fontsize=16, fontweight='bold')
plt.savefig('../plots/pairplot.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: plots/pairplot.png")
plt.close()

# ============================================================================
# 10. SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("10. SUMMARY REPORT")
print("="*80)

summary_data = {
    'Metric': [
        'Total Records',
        'Total Features',
        'Selector 1 Count',
        'Selector 2 Count',
        'Mean Drinks/Day',
        'High AST/ALT Ratio (>2)',
        'Significant Features (p<0.05)',
        'Strongest Correlation with Selector'
    ],
    'Value': [
        len(df),
        len(df.columns),
        (df['selector'] == 1).sum(),
        (df['selector'] == 2).sum(),
        f"{df['drinks'].mean():.2f}",
        f"{high_ratio} ({high_ratio/len(df)*100:.1f}%)",
        comparison_df[comparison_df['Significant'] == 'Yes'].shape[0],
        corr_matrix['selector'].abs().sort_values(ascending=False).index[1]  # type: ignore
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('../reports/eda_summary.csv', index=False)
print("\n[OK] Saved: reports/eda_summary.csv")

print("\nEDA Summary:")
for _, row in summary_df.iterrows():
    print(f"  {row['Metric']}: {row['Value']}")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "="*80)
print("EXPLORATORY DATA ANALYSIS COMPLETE")
print("="*80)

print("\nGenerated files:")
print("  Plots:")
print("    - plots/selector_distribution.png")
print("    - plots/numerical_distributions.png")
print("    - plots/boxplots.png")
print("    - plots/correlation_heatmap.png")
print("    - plots/features_by_selector.png")
print("    - plots/alcohol_analysis.png")
print("    - plots/ast_alt_ratio_analysis.png")
print("    - plots/pairplot.png")
print("  Reports:")
print("    - reports/correlation_matrix.csv")
print("    - reports/selector_comparison.csv")
print("    - reports/eda_summary.csv")

