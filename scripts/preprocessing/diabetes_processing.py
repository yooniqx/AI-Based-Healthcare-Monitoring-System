import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

CODE_MAPPING = {33: 'Regular_Insulin', 34: 'NPH_Insulin', 35: 'UltraLente_Insulin', 48: 'Unspecified_Blood_Glucose', 57: 'Unspecified_Blood_Glucose_2', 58: 'Pre_Breakfast_Blood_Glucose', 59: 'Post_Breakfast_Blood_Glucose', 60: 'Pre_Lunch_Blood_Glucose', 61: 'Post_Lunch_Blood_Glucose', 62: 'Pre_Supper_Blood_Glucose', 63: 'Post_Supper_Blood_Glucose', 64: 'Pre_Snack_Blood_Glucose', 65: 'Hypoglycemic_Symptoms', 66: 'Typical_Meal', 67: 'More_Than_Usual_Meal', 68: 'Less_Than_Usual_Meal', 69: 'Typical_Exercise', 70: 'More_Than_Usual_Exercise', 71: 'Less_Than_Usual_Exercise', 72: 'Unspecified_Special_Event'}

print('='*60)
print('DIABETES DATA PROCESSING PIPELINE')
print('='*60)
print('\n[1/5] Loading patient data...')

all_records = []
data_dir = 'Diabetes-Data'
data_files = sorted([f for f in os.listdir(data_dir) if f.startswith('data-')])

for data_file in data_files:
    patient_id = int(data_file.split('-')[1])
    file_path = os.path.join(data_dir, data_file)
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                try:
                    date_str, time_str, code, value = parts
                    datetime_str = f'{date_str} {time_str}'
                    dt = datetime.strptime(datetime_str, '%m-%d-%Y %H:%M')
                    all_records.append({'Patient_ID': patient_id, 'Date': date_str, 'Time': time_str, 'DateTime': dt, 'Code': int(code), 'Code_Description': CODE_MAPPING.get(int(code), 'Unknown'), 'Value': int(value)})
                except (ValueError, TypeError):
                    continue

df_raw = pd.DataFrame(all_records)
print(f'   Loaded {len(df_raw)} records from {len(data_files)} patients')

print('\n[2/5] Creating master dataset...')
glucose_codes = [48, 57, 58, 59, 60, 61, 62, 63, 64]
insulin_codes = [33, 34, 35]
glucose_df = df_raw[df_raw['Code'].isin(glucose_codes)].copy()
insulin_df = df_raw[df_raw['Code'].isin(insulin_codes)].copy()

patient_features = []
for patient_id in df_raw['Patient_ID'].unique():
    pg = glucose_df[glucose_df['Patient_ID'] == patient_id]
    pi = insulin_df[insulin_df['Patient_ID'] == patient_id]
    pa = df_raw[df_raw['Patient_ID'] == patient_id]
    features = {'Patient_ID': patient_id, 'Avg_Blood_Glucose': pg['Value'].mean() if len(pg) > 0 else np.nan, 'Min_Blood_Glucose': pg['Value'].min() if len(pg) > 0 else np.nan, 'Max_Blood_Glucose': pg['Value'].max() if len(pg) > 0 else np.nan, 'Std_Blood_Glucose': pg['Value'].std() if len(pg) > 0 else np.nan, 'Median_Blood_Glucose': pg['Value'].median() if len(pg) > 0 else np.nan, 'Avg_Regular_Insulin': pi[pi['Code'] == 33]['Value'].mean() if len(pi[pi['Code'] == 33]) > 0 else 0, 'Avg_NPH_Insulin': pi[pi['Code'] == 34]['Value'].mean() if len(pi[pi['Code'] == 34]) > 0 else 0, 'Total_Daily_Insulin': pi.groupby(pi['DateTime'].dt.date)['Value'].sum().mean() if len(pi) > 0 else 0, 'Num_Glucose_Measurements': len(pg), 'Num_Insulin_Doses': len(pi), 'Num_Hypoglycemic_Events': len(pa[pa['Code'] == 65]), 'Days_Monitored': (pa['DateTime'].max() - pa['DateTime'].min()).days if len(pa) > 0 else 0, 'Glucose_Variability': pg['Value'].std() / pg['Value'].mean() if len(pg) > 0 and pg['Value'].mean() > 0 else np.nan, 'Hyperglycemia_Percentage': (pg['Value'] > 200).sum() / len(pg) * 100 if len(pg) > 0 else 0, 'Hypoglycemia_Percentage': (pg['Value'] < 70).sum() / len(pg) * 100 if len(pg) > 0 else 0, 'Target_Range_Percentage': ((pg['Value'] >= 70) & (pg['Value'] <= 180)).sum() / len(pg) * 100 if len(pg) > 0 else 0}
    patient_features.append(features)

df_master = pd.DataFrame(patient_features)
df_master.to_csv('datasets/diabetes/diabetes_master_dataset.csv', index=False)
print(f'   Master dataset created: {df_master.shape}')

print('\n[3/5] Generating dataset summary...')
with open('reports/diabetes/DATASET_SUMMARY.txt', 'w') as f:
    f.write('DIABETES DATASET SUMMARY\n' + '='*60 + '\n\n')
    f.write(f'Total Patients: {len(df_master)}\nTotal Features: {len(df_master.columns)}\nTotal Raw Records: {len(df_raw)}\nDate Range: 1989-1991\n\n')
    f.write('Feature Statistics:\n' + '-'*60 + '\n' + df_master.describe().to_string() + '\n\n' + '='*60 + '\n')
    f.write('Missing Values:\n' + '-'*60 + '\n' + df_master.isnull().sum().to_string() + '\n\n' + '='*60 + '\n')
    f.write('Data Quality Insights:\n' + '-'*60 + '\n')
    f.write(f'Patients with complete glucose data: {df_master["Avg_Blood_Glucose"].notna().sum()}\n')
    f.write(f'Average monitoring period: {df_master["Days_Monitored"].mean():.1f} days\n')
    f.write(f'Average glucose measurements per patient: {df_master["Num_Glucose_Measurements"].mean():.1f}\n')
print(' Dataset summary saved')

print('\n[4/5] Performing EDA and generating visualizations...')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

plt.figure(figsize=(12, 10))
numeric_cols = df_master.select_dtypes(include=[np.number]).columns
corr_matrix = df_master[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/diabetes/eda/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].hist(df_master['Avg_Blood_Glucose'].dropna(), bins=30, color='skyblue', edgecolor='black')
axes[0, 0].axvline(150, color='red', linestyle='--', label='Target: 150 mg/dl')
axes[0, 0].set_title('Average Blood Glucose Distribution')
axes[0, 0].set_xlabel('Blood Glucose (mg/dl)')
axes[0, 0].legend()
axes[0, 1].hist(df_master['Std_Blood_Glucose'].dropna(), bins=30, color='lightcoral', edgecolor='black')
axes[0, 1].set_title('Blood Glucose Variability Distribution')
axes[0, 1].set_xlabel('Standard Deviation (mg/dl)')
axes[1, 0].hist(df_master['Hyperglycemia_Percentage'].dropna(), bins=30, color='orange', edgecolor='black')
axes[1, 0].set_title('Hyperglycemia Percentage Distribution')
axes[1, 0].set_xlabel('Percentage (%)')
axes[1, 1].hist(df_master['Target_Range_Percentage'].dropna(), bins=30, color='lightgreen', edgecolor='black')
axes[1, 1].set_title('Target Range Percentage Distribution')
axes[1, 1].set_xlabel('Percentage (%)')
plt.tight_layout()
plt.savefig('reports/diabetes/eda/glucose_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(df_master['Avg_Regular_Insulin'], df_master['Avg_Blood_Glucose'], alpha=0.6, color='blue')
axes[0].set_title('Regular Insulin vs Average Blood Glucose')
axes[0].set_xlabel('Average Regular Insulin Dose')
axes[0].set_ylabel('Average Blood Glucose (mg/dl)')
axes[1].scatter(df_master['Total_Daily_Insulin'], df_master['Avg_Blood_Glucose'], alpha=0.6, color='green')
axes[1].set_title('Total Daily Insulin vs Average Blood Glucose')
axes[1].set_xlabel('Total Daily Insulin')
axes[1].set_ylabel('Average Blood Glucose (mg/dl)')
plt.tight_layout()
plt.savefig('reports/diabetes/eda/insulin_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df_master['Days_Monitored'], bins=30, color='purple', edgecolor='black')
axes[0].set_title('Monitoring Duration Distribution')
axes[0].set_xlabel('Days Monitored')
axes[0].set_ylabel('Number of Patients')
axes[1].hist(df_master['Num_Glucose_Measurements'], bins=30, color='teal', edgecolor='black')
axes[1].set_title('Glucose Measurements per Patient')
axes[1].set_xlabel('Number of Measurements')
axes[1].set_ylabel('Number of Patients')
plt.tight_layout()
plt.savefig('reports/diabetes/eda/monitoring_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print(' Generated 4 visualization files')

print('\n[5/5] Generating EDA summary report...')
with open('reports/diabetes/EDA_SUMMARY.txt', 'w') as f:
    f.write('EXPLORATORY DATA ANALYSIS SUMMARY\n' + '='*60 + '\n\nKey Findings:\n' + '-'*60 + '\n\n')
    f.write('1. BLOOD GLUCOSE CONTROL:\n')
    avg_glucose = df_master['Avg_Blood_Glucose'].mean()
    f.write(f'   - Average blood glucose across all patients: {avg_glucose:.1f} mg/dl\n')
    f.write(f'   - Patients with good control (<150 mg/dl): {(df_master["Avg_Blood_Glucose"] < 150).sum()}\n')
    f.write(f'   - Patients with poor control (>200 mg/dl): {(df_master["Avg_Blood_Glucose"] > 200).sum()}\n\n')
    f.write('2. GLUCOSE VARIABILITY:\n')
    f.write(f'   - Average glucose variability: {df_master["Glucose_Variability"].mean():.3f}\n')
    f.write(f'   - High variability patients (>0.3): {(df_master["Glucose_Variability"] > 0.3).sum()}\n\n')
    f.write('3. HYPOGLYCEMIA RISK:\n')
    f.write(f'   - Average hypoglycemia percentage: {df_master["Hypoglycemia_Percentage"].mean():.1f}%\n')
    f.write(f'   - Patients with hypoglycemic events: {(df_master["Num_Hypoglycemic_Events"] > 0).sum()}\n\n')
    f.write('4. INSULIN THERAPY:\n')
    f.write(f'   - Average regular insulin dose: {df_master["Avg_Regular_Insulin"].mean():.1f} units\n')
    f.write(f'   - Average NPH insulin dose: {df_master["Avg_NPH_Insulin"].mean():.1f} units\n')
    f.write(f'   - Average total daily insulin: {df_master["Total_Daily_Insulin"].mean():.1f} units\n\n')
    f.write('5. MONITORING COMPLIANCE:\n')
    f.write(f'   - Average monitoring duration: {df_master["Days_Monitored"].mean():.1f} days\n')
    f.write(f'   - Average measurements per patient: {df_master["Num_Glucose_Measurements"].mean():.1f}\n\n')
    f.write('='*60 + '\nVisualizations Generated:\n' + '-'*60 + '\n')
    f.write('1. correlation_heatmap.png - Feature correlations\n2. glucose_distributions.png - Glucose metrics distributions\n3. insulin_analysis.png - Insulin vs glucose relationships\n4. monitoring_patterns.png - Patient monitoring patterns\n')
print('   EDA summary saved')

print('\n' + '='*60)
print('PROCESSING COMPLETE!')
print('='*60)
print('\nOutputs:')
print(' - Master dataset: datasets/diabetes/diabetes_master_dataset.csv')
print(' - Dataset summary: reports/diabetes/DATASET_SUMMARY.txt')
print(' - EDA summary: reports/diabetes/EDA_SUMMARY.txt')
print(' - Visualizations: reports/diabetes/eda/ (4 files)')
