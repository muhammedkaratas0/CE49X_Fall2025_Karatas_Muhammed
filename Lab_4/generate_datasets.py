"""
Generate synthetic datasets for Lab 4: Statistical Analysis
Creates concrete_strength.csv, structural_loads.csv, and material_properties.csv
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

def generate_concrete_strength_data():
    """Generate concrete strength dataset (100 samples)."""
    batch_ids = np.repeat(['B1', 'B2', 'B3', 'B4', 'B5'], 20)
    ages = np.random.choice([7, 14, 28, 56, 90], size=100)
    mix_types = np.random.choice(['Type I', 'Type II', 'Type III'], size=100)

    # Generate strength values with some variation based on age and mix type
    base_strength = 30  # MPa
    age_factor = ages / 28  # Normalized age
    mix_factor = {'Type I': 1.0, 'Type II': 1.1, 'Type III': 1.2}

    strength_mpa = []
    for i in range(100):
        mix_mult = mix_factor[mix_types[i]]
        mean_strength = base_strength * age_factor[i] * mix_mult
        # Add normal variation
        strength = np.random.normal(mean_strength, mean_strength * 0.1)
        strength_mpa.append(max(10, strength))  # Minimum 10 MPa

    df = pd.DataFrame({
        'batch_id': batch_ids,
        'age_days': ages,
        'mix_type': mix_types,
        'strength_mpa': strength_mpa
    })

    return df

def generate_structural_loads_data():
    """Generate structural load dataset (200 hourly measurements)."""
    timestamps = pd.date_range(start='2024-01-01', periods=200, freq='H')
    component_types = np.random.choice(['Beam', 'Column', 'Slab', 'Foundation'], size=200)

    # Generate load values with daily patterns
    hour_of_day = timestamps.hour
    base_load = 100  # kN

    # Higher loads during day hours (8-18)
    time_factor = 1 + 0.3 * np.sin((hour_of_day - 6) * np.pi / 12)
    time_factor = np.where((hour_of_day >= 6) & (hour_of_day <= 18), time_factor, 0.7)

    load_kn = []
    for i in range(200):
        mean_load = base_load * time_factor[i]
        load = np.random.normal(mean_load, mean_load * 0.15)
        load_kn.append(max(10, load))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'component_type': component_types,
        'load_kn': load_kn
    })

    return df

def generate_material_properties_data():
    """Generate material property dataset (200 samples, 50 per material)."""
    materials = ['Steel', 'Concrete', 'Aluminum', 'Composite']

    # Mean yield strengths for each material (MPa)
    mean_strengths = {
        'Steel': 250,
        'Concrete': 30,
        'Aluminum': 200,
        'Composite': 180
    }

    # Standard deviations (10% of mean)
    std_strengths = {k: v * 0.10 for k, v in mean_strengths.items()}

    material_list = []
    test_numbers = []
    yield_strengths = []

    for material in materials:
        for test_num in range(1, 51):
            material_list.append(material)
            test_numbers.append(test_num)
            strength = np.random.normal(mean_strengths[material], std_strengths[material])
            yield_strengths.append(max(5, strength))  # Minimum 5 MPa

    df = pd.DataFrame({
        'material_type': material_list,
        'test_number': test_numbers,
        'yield_strength_mpa': yield_strengths
    })

    return df

def main():
    """Generate all datasets and save to CSV files."""
    print("Generating synthetic datasets for Lab 4...")

    # Generate concrete strength data
    concrete_df = generate_concrete_strength_data()
    concrete_df.to_csv('../datasets/concrete_strength.csv', index=False)
    print(f"✓ Generated concrete_strength.csv: {len(concrete_df)} samples")

    # Generate structural loads data
    loads_df = generate_structural_loads_data()
    loads_df.to_csv('../datasets/structural_loads.csv', index=False)
    print(f"✓ Generated structural_loads.csv: {len(loads_df)} samples")

    # Generate material properties data
    materials_df = generate_material_properties_data()
    materials_df.to_csv('../datasets/material_properties.csv', index=False)
    print(f"✓ Generated material_properties.csv: {len(materials_df)} samples")

    print("\nDatasets created successfully in '../datasets/' folder")
    print("\nSample data from concrete_strength.csv:")
    print(concrete_df.head())
    print(f"\nDescriptive statistics for concrete strength:")
    print(concrete_df['strength_mpa'].describe())

if __name__ == "__main__":
    main()
