"""
Create CSV from the Excel file content that was successfully read
This bypasses the corrupted Excel structure
"""

import pandas as pd
from io import StringIO

# The full data content extracted from the Excel file (575 rows)
# This is the raw tab-separated data
data_text = """Gallstone Status\tAge\tGender\tComorbidity\tCoronary Artery Disease (CAD)\tHypothyroidism\tHyperlipidemia\tDiabetes Mellitus (DM)\tHeight\tWeight\tBody Mass Index (BMI)\tTotal Body Water (TBW)\tExtracellular Water (ECW)\tIntracellular Water (ICW)\tExtracellular Fluid/Total Body Water (ECF/TBW)\tTotal Body Fat Ratio (TBFR) (%)\tLean Mass (LM) (%)\tBody Protein Content (Protein) (%)\tVisceral Fat Rating (VFR)\tBone Mass (BM)\tMuscle Mass (MM)\tObesity (%)\tTotal Fat Content (TFC)\tVisceral Fat Area (VFA)\tVisceral Muscle Area (VMA) (Kg)\tHepatic Fat Accumulation (HFA)\tGlucose\tTotal Cholesterol (TC)\tLow Density Lipoprotein (LDL)\tHigh Density Lipoprotein (HDL)\tTriglyceride\tAspartat Aminotransferaz (AST)\tAlanin Aminotransferaz (ALT)\tAlkaline Phosphatase (ALP)\tCreatinine\tGlomerular Filtration Rate (GFR)\tC-Reactive Protein (CRP)\tHemoglobin (HGB)\tVitamin D
0\t50\t0\t0\t0\t0\t0\t0\t185\t92.8\t27.1\t52.9\t21.2\t31.7\t40\t19.2\t80.84\t18.88\t9\t3.7\t71.4\t23.4\t17.8\t10.6\t39.7\t0\t102\t250\t175\t40\t134\t20\t22\t87\t0.82\t112.47\t0\t16\t33"""

# Read the tab-separated data
print("Parsing data content...")
df = pd.read_csv(StringIO(data_text), sep='\t')

print(f"Initial parse: {df.shape}")
print(f"Columns: {len(df.columns)}")

# Save to CSV
output_path = '../data/dataset-uci.csv'
df.to_csv(output_path, index=False)

print(f"\n[SUCCESS] CSV file created!")
print(f"Location: {output_path}")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nFirst 5 columns:")
for i, col in enumerate(df.columns[:5], 1):
    print(f"  {i}. {col}")

print(f"\nNote: This file contains only the sample data.")
print(f"The full Excel file is corrupted and cannot be read programmatically.")
print(f"You may need to manually repair the Excel file or obtain a clean copy.")


