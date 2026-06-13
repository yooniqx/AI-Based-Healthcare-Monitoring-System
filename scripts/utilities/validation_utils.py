# Data Validation Utilities
# Provides validation functions for dataset quality and completeness
# Usage: from utilities.validation_utils import validate_dataset

import pandas as pd

def validate_dataset(filepath):
    """Validate dataset structure and quality
    
    Args:
        filepath: Path to the CSV dataset file
        
    Returns:
        DataFrame: The loaded and validated dataset
    """
    df = pd.read_csv(filepath)
    print(f'Dataset shape: {df.shape}')
    print(f'Missing values: {df.isnull().sum().sum()}')
    return df

if __name__ == '__main__':
    # Example usage
    validate_dataset('datasets/diabetes/diabetes_master_dataset.csv')


