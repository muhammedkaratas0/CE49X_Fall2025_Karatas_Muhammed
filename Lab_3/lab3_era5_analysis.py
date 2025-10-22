"""
Lab 3: ERA5 Weather Data Analysis
Analyzing wind data for Berlin and Munich from ERA5 reanalysis dataset.

Author: Ali Karatas
Date: October 21, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style for better visualizations
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_era5_data(filepath):
    """
    Load ERA5 weather data from CSV file.

    Parameters:
    -----------
    filepath : str or Path
        Path to the CSV file

    Returns:
    --------
    pd.DataFrame
        Loaded dataframe with parsed timestamps
    """
    try:
        # Load the data
        df = pd.read_csv(filepath)

        # Convert timestamp column to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        print(f"Successfully loaded data from {filepath}")
        print(f"Shape: {df.shape}")
        return df

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def calculate_wind_speed(u10m, v10m):
    """
    Calculate wind speed from u and v components.

    Wind speed = sqrt(u^2 + v^2)

    Parameters:
    -----------
    u10m : array-like
        Eastward wind component at 10m (m/s)
    v10m : array-like
        Northward wind component at 10m (m/s)

    Returns:
    --------
    array-like
        Wind speed magnitude (m/s)
    """
    return np.sqrt(u10m**2 + v10m**2)


def calculate_wind_direction(u10m, v10m):
    """
    Calculate wind direction from u and v components.

    Direction is calculated as the meteorological wind direction
    (direction FROM which the wind is blowing).

    Parameters:
    -----------
    u10m : array-like
        Eastward wind component at 10m (m/s)
    v10m : array-like
        Northward wind component at 10m (m/s)

    Returns:
    --------
    array-like
        Wind direction in degrees (0-360, 0=North)
    """
    # Calculate angle in radians, convert to meteorological convention
    wind_dir = np.arctan2(-u10m, -v10m)
    # Convert to degrees (0-360, 0=North)
    wind_dir = np.degrees(wind_dir) + 180
    return wind_dir


def explore_dataset(df, city_name):
    """
    Display basic information about the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset to explore
    city_name : str
        Name of the city for display
    """
    print(f"\n{'='*60}")
    print(f"Dataset Exploration: {city_name}")
    print(f"{'='*60}")

    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn names and types:")
    print(df.dtypes)

    print(f"\nMissing values:")
    print(df.isnull().sum())

    print(f"\nBasic statistics:")
    print(df.describe())

    print(f"\nDate range: {df.index.min()} to {df.index.max()}")
    print(f"Total records: {len(df)}")


def handle_missing_values(df):
    """
    Handle missing values in the dataset.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with potential missing values

    Returns:
    --------
    pd.DataFrame
        Dataset with missing values handled
    """
    missing_count = df.isnull().sum().sum()

    if missing_count > 0:
        print(f"\nHandling {missing_count} missing values...")
        # For time series data, forward fill is often appropriate
        df_clean = df.ffill().bfill()
        print("Missing values filled using forward/backward fill.")
        return df_clean
    else:
        print("\nNo missing values found.")
        return df


def calculate_monthly_averages(df):
    """
    Calculate monthly averages for all variables.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with datetime index

    Returns:
    --------
    pd.DataFrame
        Monthly averaged data
    """
    monthly = df.resample('ME').mean()
    return monthly


def calculate_seasonal_averages(df):
    """
    Calculate seasonal averages.

    Seasons: Winter (DJF), Spring (MAM), Summer (JJA), Fall (SON)

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with datetime index

    Returns:
    --------
    pd.DataFrame
        Seasonal averaged data
    """
    # Add season column
    df_seasonal = df.copy()
    df_seasonal['month'] = df_seasonal.index.month

    # Map months to seasons
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}

    df_seasonal['season'] = df_seasonal['month'].map(season_map)

    # Calculate seasonal averages
    seasonal = df_seasonal.groupby('season').mean(numeric_only=True)

    # Order seasons properly
    season_order = ['Winter', 'Spring', 'Summer', 'Fall']
    seasonal = seasonal.reindex([s for s in season_order if s in seasonal.index])

    return seasonal


def find_extreme_weather(df, city_name):
    """
    Identify periods with extreme weather conditions.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with weather variables
    city_name : str
        Name of the city
    """
    print(f"\n{'='*60}")
    print(f"Extreme Weather Analysis: {city_name}")
    print(f"{'='*60}")

    if 'wind_speed' in df.columns:
        # Find highest wind speeds
        top_10_wind = df.nlargest(10, 'wind_speed')[['wind_speed', 'wind_direction']]
        print(f"\nTop 10 highest wind speed events:")
        print(top_10_wind)

        # Find the windiest day
        daily_max_wind = df['wind_speed'].resample('D').max()
        windiest_day = daily_max_wind.idxmax()
        windiest_speed = daily_max_wind.max()
        print(f"\nWindiest day: {windiest_day.date()} with max wind speed of {windiest_speed:.2f} m/s")


def calculate_diurnal_patterns(df):
    """
    Calculate average diurnal (daily) patterns.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with datetime index

    Returns:
    --------
    pd.DataFrame
        Hourly averaged patterns
    """
    df_diurnal = df.copy()
    df_diurnal['hour'] = df_diurnal.index.hour
    diurnal = df_diurnal.groupby('hour').mean(numeric_only=True)
    return diurnal


def plot_monthly_wind_speeds(berlin_monthly, munich_monthly):
    """
    Create time series plot of monthly average wind speeds.

    Parameters:
    -----------
    berlin_monthly : pd.DataFrame
        Berlin monthly data
    munich_monthly : pd.DataFrame
        Munich monthly data
    """
    plt.figure(figsize=(14, 6))

    plt.plot(berlin_monthly.index, berlin_monthly['wind_speed'],
             marker='o', label='Berlin', linewidth=2, markersize=8)
    plt.plot(munich_monthly.index, munich_monthly['wind_speed'],
             marker='s', label='Munich', linewidth=2, markersize=8)

    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.title('Monthly Average Wind Speed: Berlin vs Munich (2024)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('monthly_wind_speeds.png', dpi=300, bbox_inches='tight')
    print("\nSaved: monthly_wind_speeds.png")
    plt.close()


def plot_seasonal_comparison(berlin_seasonal, munich_seasonal):
    """
    Create bar chart comparing seasonal wind speeds.

    Parameters:
    -----------
    berlin_seasonal : pd.DataFrame
        Berlin seasonal data
    munich_seasonal : pd.DataFrame
        Munich seasonal data
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(berlin_seasonal.index))
    width = 0.35

    ax.bar(x - width/2, berlin_seasonal['wind_speed'], width,
           label='Berlin', color='skyblue', edgecolor='black')
    ax.bar(x + width/2, munich_seasonal['wind_speed'], width,
           label='Munich', color='lightcoral', edgecolor='black')

    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Average Wind Speed (m/s)', fontsize=12)
    ax.set_title('Seasonal Wind Speed Comparison (2024)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(berlin_seasonal.index)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('seasonal_wind_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: seasonal_wind_comparison.png")
    plt.close()


def plot_wind_rose(df, city_name):
    """
    Create a wind rose diagram showing wind direction distribution.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with wind_direction and wind_speed
    city_name : str
        Name of the city
    """
    if 'wind_direction' not in df.columns or 'wind_speed' not in df.columns:
        print(f"Warning: Cannot create wind rose for {city_name} - missing data")
        return

    # Create direction bins (16 directions)
    # Define 16 directions
    dir_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    # Create bins that wrap around 360 degrees
    # Each bin is 22.5 degrees wide
    df_wind = df[['wind_direction', 'wind_speed']].copy()

    # Bin the directions manually to avoid duplicate label issue
    bins = np.arange(-11.25, 360, 22.5)
    # Handle wrap-around: values > 348.75 should map to N
    df_wind['dir_index'] = pd.cut(df_wind['wind_direction'], bins=bins, labels=False)
    df_wind['dir_index'] = df_wind['dir_index'].fillna(0).astype(int) % 16
    df_wind['dir_bin'] = df_wind['dir_index'].apply(lambda x: dir_labels[x])

    # Calculate frequency for each direction
    dir_counts = df_wind['dir_bin'].value_counts()

    # Create polar plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='polar')

    # Convert direction labels to angles (in radians)
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)

    # Get counts for each direction
    counts = [dir_counts.get(label, 0) for label in dir_labels]
    counts += counts[:1]  # Close the circle

    angles_plot = list(angles) + [angles[0]]

    ax.plot(angles_plot, counts, 'o-', linewidth=2, color='blue')
    ax.fill(angles_plot, counts, alpha=0.25, color='blue')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(dir_labels)
    ax.set_title(f'Wind Rose: {city_name} (2024)', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)

    plt.tight_layout()
    filename = f'wind_rose_{city_name.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()


def plot_diurnal_patterns(berlin_diurnal, munich_diurnal):
    """
    Plot diurnal (hourly) wind speed patterns.

    Parameters:
    -----------
    berlin_diurnal : pd.DataFrame
        Berlin diurnal data
    munich_diurnal : pd.DataFrame
        Munich diurnal data
    """
    plt.figure(figsize=(12, 6))

    plt.plot(berlin_diurnal.index, berlin_diurnal['wind_speed'],
             marker='o', label='Berlin', linewidth=2, markersize=8)
    plt.plot(munich_diurnal.index, munich_diurnal['wind_speed'],
             marker='s', label='Munich', linewidth=2, markersize=8)

    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Wind Speed (m/s)', fontsize=12)
    plt.title('Average Diurnal Wind Speed Pattern (2024)', fontsize=14, fontweight='bold')
    plt.xticks(range(0, 24))
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('diurnal_wind_pattern.png', dpi=300, bbox_inches='tight')
    print("Saved: diurnal_wind_pattern.png")
    plt.close()


def main():
    """
    Main function to run the ERA5 data analysis.
    """
    print("="*60)
    print("Lab 3: ERA5 Weather Data Analysis")
    print("="*60)

    # Define file paths
    berlin_file = Path("berlin_era5_wind_20241231_20241231.csv")
    munich_file = Path("munich_era5_wind_20241231_20241231.csv")

    # Check if files exist
    if not berlin_file.exists() or not munich_file.exists():
        print("\nError: Data files not found!")
        print(f"Looking for:")
        print(f"  - {berlin_file}")
        print(f"  - {munich_file}")
        print("\nPlease ensure the ERA5 data files are in the current directory.")
        sys.exit(1)

    # Load datasets
    print("\n1. Loading datasets...")
    berlin_df = load_era5_data(berlin_file)
    munich_df = load_era5_data(munich_file)

    # Explore datasets
    print("\n2. Exploring datasets...")
    explore_dataset(berlin_df, "Berlin")
    explore_dataset(munich_df, "Munich")

    # Handle missing values
    print("\n3. Handling missing values...")
    berlin_df = handle_missing_values(berlin_df)
    munich_df = handle_missing_values(munich_df)

    # Calculate wind speed and direction
    print("\n4. Calculating wind speed and direction...")
    berlin_df['wind_speed'] = calculate_wind_speed(berlin_df['u10m'], berlin_df['v10m'])
    berlin_df['wind_direction'] = calculate_wind_direction(berlin_df['u10m'], berlin_df['v10m'])

    munich_df['wind_speed'] = calculate_wind_speed(munich_df['u10m'], munich_df['v10m'])
    munich_df['wind_direction'] = calculate_wind_direction(munich_df['u10m'], munich_df['v10m'])

    print("Wind speed and direction calculated successfully.")

    # Calculate monthly averages
    print("\n5. Calculating monthly averages...")
    berlin_monthly = calculate_monthly_averages(berlin_df)
    munich_monthly = calculate_monthly_averages(munich_df)

    print("\nBerlin Monthly Average Wind Speed:")
    print(berlin_monthly['wind_speed'])
    print("\nMunich Monthly Average Wind Speed:")
    print(munich_monthly['wind_speed'])

    # Calculate seasonal averages
    print("\n6. Calculating seasonal averages...")
    berlin_seasonal = calculate_seasonal_averages(berlin_df)
    munich_seasonal = calculate_seasonal_averages(munich_df)

    print("\nBerlin Seasonal Average Wind Speed:")
    print(berlin_seasonal['wind_speed'])
    print("\nMunich Seasonal Average Wind Speed:")
    print(munich_seasonal['wind_speed'])

    # Extreme weather analysis
    print("\n7. Analyzing extreme weather conditions...")
    find_extreme_weather(berlin_df, "Berlin")
    find_extreme_weather(munich_df, "Munich")

    # Diurnal patterns
    print("\n8. Calculating diurnal patterns...")
    berlin_diurnal = calculate_diurnal_patterns(berlin_df)
    munich_diurnal = calculate_diurnal_patterns(munich_df)

    print("\nBerlin Average Hourly Wind Speed:")
    print(berlin_diurnal['wind_speed'])
    print("\nMunich Average Hourly Wind Speed:")
    print(munich_diurnal['wind_speed'])

    # Create visualizations
    print("\n9. Creating visualizations...")
    plot_monthly_wind_speeds(berlin_monthly, munich_monthly)
    plot_seasonal_comparison(berlin_seasonal, munich_seasonal)
    plot_wind_rose(berlin_df, "Berlin")
    plot_wind_rose(munich_df, "Munich")
    plot_diurnal_patterns(berlin_diurnal, munich_diurnal)

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("\nGenerated visualizations:")
    print("  1. monthly_wind_speeds.png - Monthly wind speed comparison")
    print("  2. seasonal_wind_comparison.png - Seasonal comparison bar chart")
    print("  3. wind_rose_berlin.png - Berlin wind rose diagram")
    print("  4. wind_rose_munich.png - Munich wind rose diagram")
    print("  5. diurnal_wind_pattern.png - Daily hourly pattern")

    print("\nKey Findings:")
    print(f"  - Berlin average wind speed: {berlin_df['wind_speed'].mean():.2f} m/s")
    print(f"  - Munich average wind speed: {munich_df['wind_speed'].mean():.2f} m/s")
    print(f"  - Berlin windiest season: {berlin_seasonal['wind_speed'].idxmax()}")
    print(f"  - Munich windiest season: {munich_seasonal['wind_speed'].idxmax()}")


if __name__ == "__main__":
    main()
