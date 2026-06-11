import zipfile
import os
import shutil

# Create liver_project directory structure
project_dir = "liver_project"
subdirs = ["data", "scripts", "output", "reports", "plots"]

# Create main directory
if not os.path.exists(project_dir):
    os.makedirs(project_dir)
    print(f"Created {project_dir}/")

# Create subdirectories
for subdir in subdirs:
    path = os.path.join(project_dir, subdir)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created {path}/")

# Extract liver+disorders.zip
zip_file = "liver+disorders.zip"
if os.path.exists(zip_file):
    print(f"\nExtracting {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(project_dir)
    print(f"Extracted to {project_dir}/")
else:
    print(f"Error: {zip_file} not found!")

# List extracted files
print("\nExtracted files:")
for root, dirs, files in os.walk(project_dir):
    level = root.replace(project_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")

print("\nLiver project setup complete!")

