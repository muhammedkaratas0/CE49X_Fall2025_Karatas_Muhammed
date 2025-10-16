# CE 49X - Lab 2: Soil Test Data Analysis

# Student Name: Muhammed Ali Karataş  
# Student ID: 2012403030
# Date: 14.10.2025

import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the soil test dataset from a CSV file.
    Handles missing file errors gracefully.
    """
    try:
        df = pd.read_csv(file_path)
        print("✅ File loaded successfully!")
        return df
    except FileNotFoundError:
        print("❌ Error: File not found. Check the file path.")
        return None



def clean_data(df):
    """
    Cleans the dataset by handling missing values and removing outliers.
    Args:
        df (pd.DataFrame): Input dataset.
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    df_cleaned = df.copy()  # Work on a copy to preserve the original

    # Fill missing values for each numeric column with its mean
    for col in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
        if col in df_cleaned.columns:
            mean_val = df_cleaned[col].mean()
            df_cleaned[col] = df_cleaned[col].fillna(mean_val)
            print(f"🧽 Filled missing values in '{col}' with mean ({mean_val:.2f})")

     # Remove outliers in 'soil_ph' (more than 3 std deviations from mean)
    if 'soil_ph' in df_cleaned.columns:
        mean = df_cleaned['soil_ph'].mean()
        std = df_cleaned['soil_ph'].std()
        before = len(df_cleaned)
        df_cleaned = df_cleaned[(df_cleaned['soil_ph'] >= mean - 3*std) &
                                (df_cleaned['soil_ph'] <= mean + 3*std)]
        after = len(df_cleaned)
        print(f"📉 Removed {before - after} outlier rows in 'soil_ph'")

    print("\n🧾 Preview of cleaned data:")
    print(df_cleaned.head())
    return df_cleaned

def compute_statistics(df, column):
    """
    Compute and print descriptive statistics for the specified column.
    """
    try:
        min_val = df[column].min()
        max_val = df[column].max()
        mean_val = df[column].mean()
        median_val = df[column].median()
        std_val = df[column].std()

        print(f"\n📈 Descriptive statistics for '{column}':")
        print(f"  Minimum: {min_val:.2f}")
        print(f"  Maximum: {max_val:.2f}")
        print(f"  Mean: {mean_val:.2f}")
        print(f"  Median: {median_val:.2f}")
        print(f"  Standard Deviation: {std_val:.2f}")

    except KeyError:
        print(f"❌ Error: Column '{column}' not found in the dataset.")

def main():
    file_path = '/Users/alikaratas/Desktop/CE49X/Lab_2/soil_test.csv'
      # File must be in same folder

    df = load_data(file_path)
    if df is None:
        return

    df_clean = clean_data(df)
    compute_statistics(df_clean, 'soil_ph')

    # Optional: compute for other columns
    # compute_statistics(df_clean, 'nitrogen')
    # compute_statistics(df_clean, 'phosphorus')
    # compute_statistics(df_clean, 'moisture')

    
if __name__ == '__main__':
    main()


# =============================================================================
# REFLECTION QUESTIONS
# =============================================================================
# Answer these questions in comments below:

# 1. What was the most challenging part of this lab?
# Answer:  Cleaning data and removing them

# 2. How could soil data analysis help civil engineers in real projects?
# Answer: It makes analyzing data, removing outliers and make data sets logical and in correct order.

# 3. What additional features would make this soil analysis tool more useful?
# Answer: Easy upload panel would make it better and easy to use and also that would make this programme more scalable 
# and developable.

# 4. How did error handling improve the robustness of your code?
# Answer: Error handling improved the robustness of our code by preventing crashes 
# and allowing it to handle unexpected errors gracefully.