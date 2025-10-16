# -----------------------------------------------------------
# Week 3: Numpy for Structural Analysis
# -----------------------------------------------------------
# This file demonstrates how to use numpy for structural engineering
# calculations, combining with your existing structural analysis code
# -----------------------------------------------------------

import numpy as np
import math

print("="*60)
print("NUMPY FOR STRUCTURAL ANALYSIS")
print("="*60)

# 1. Material Properties Matrix
print("\n1. MATERIAL PROPERTIES")
print("-" * 30)

# Define material properties as numpy arrays
steel_E = 200e9  # Young's modulus for steel (Pa)
concrete_E = 30e9  # Young's modulus for concrete (Pa)
wood_E = 12e9   # Young's modulus for wood (Pa)

materials = np.array([
    ['Steel', steel_E, 7850, 250e6],      # Name, E, density, yield strength
    ['Concrete', concrete_E, 2400, 30e6],
    ['Wood', wood_E, 600, 40e6]
])

print("Material Properties:")
print("Material    | E (GPa) | Density (kg/m³) | Yield (MPa)")
print("-" * 55)
for i, material in enumerate(materials):
    name = material[0]
    E_gpa = float(material[1]) / 1e9
    density = float(material[2])
    yield_mpa = float(material[3]) / 1e6
    print(f"{name:10} | {E_gpa:7.0f} | {density:13.0f} | {yield_mpa:10.0f}")

# 2. Cross-sectional Properties
print("\n2. CROSS-SECTIONAL PROPERTIES")
print("-" * 30)

# Rectangle dimensions (width, height) in meters
sections = np.array([
    [0.2, 0.4],   # 200mm x 400mm
    [0.3, 0.6],   # 300mm x 600mm
    [0.25, 0.5],  # 250mm x 500mm
])

print("Cross-sections (width x height):")
for i, section in enumerate(sections):
    width, height = section
    area = width * height
    moment_of_inertia = (width * height**3) / 12
    print(f"Section {i+1}: {width*1000:.0f}mm x {height*1000:.0f}mm")
    print(f"  Area: {area*1e4:.1f} cm²")
    print(f"  Moment of Inertia: {moment_of_inertia*1e8:.1f} cm⁴")

# 3. Load Analysis using Numpy
print("\n3. LOAD ANALYSIS")
print("-" * 30)

# Define loads as numpy arrays
point_loads = np.array([10e3, 15e3, 20e3])  # Point loads in N
distributed_loads = np.array([5e3, 8e3, 12e3])  # Distributed loads in N/m
load_positions = np.array([2.0, 4.0, 6.0])  # Positions in meters

print("Load Analysis:")
print("Load Type        | Magnitude | Position")
print("-" * 40)
for i in range(len(point_loads)):
    print(f"Point Load {i+1}    | {point_loads[i]/1e3:8.1f} kN | {load_positions[i]:8.1f} m")

for i in range(len(distributed_loads)):
    print(f"Distributed Load {i+1} | {distributed_loads[i]/1e3:8.1f} kN/m | {load_positions[i]:8.1f} m")

# Calculate total loads
total_point_load = np.sum(point_loads)
total_distributed_load = np.sum(distributed_loads)
print(f"\nTotal Point Load: {total_point_load/1e3:.1f} kN")
print(f"Total Distributed Load: {total_distributed_load/1e3:.1f} kN/m")

# 4. Deflection Calculations
print("\n4. DEFLECTION CALCULATIONS")
print("-" * 30)

# Simple beam deflection calculation
L = 8.0  # Beam length in meters
E = steel_E
I = 1e-4  # Moment of inertia in m⁴
P = 20e3  # Point load in N

# Maximum deflection for simply supported beam with point load at center
max_deflection = (P * L**3) / (48 * E * I)

# Deflection along the beam (at various points)
x_points = np.linspace(0, L, 11)  # 11 points along the beam
deflections = []

for x in x_points:
    if x <= L/2:
        # Deflection formula for first half of beam
        deflection = (P * x) / (48 * E * I) * (3 * L**2 - 4 * x**2)
    else:
        # Deflection formula for second half of beam
        deflection = (P * (L - x)) / (48 * E * I) * (3 * L**2 - 4 * (L - x)**2)
    deflections.append(deflection)

deflections = np.array(deflections)

print(f"Beam Properties:")
print(f"Length: {L:.1f} m")
print(f"Point Load: {P/1e3:.1f} kN")
print(f"Maximum Deflection: {max_deflection*1000:.2f} mm")

print(f"\nDeflection along beam:")
print("Position (m) | Deflection (mm)")
print("-" * 28)
for i, (x, deflection) in enumerate(zip(x_points, deflections)):
    print(f"{x:11.1f} | {deflection*1000:13.2f}")

# 5. Stress Analysis
print("\n5. STRESS ANALYSIS")
print("-" * 30)

# Calculate bending stress
M = P * L / 4  # Maximum moment at center
c = 0.2  # Distance from neutral axis to extreme fiber (m)
bending_stress = M * c / I

# Calculate shear stress
V = P / 2  # Maximum shear force
A = 0.2 * 0.4  # Cross-sectional area
shear_stress = 1.5 * V / A  # Approximate for rectangular section

print(f"Stress Analysis:")
print(f"Maximum Moment: {M/1e3:.1f} kN⋅m")
print(f"Maximum Shear: {V/1e3:.1f} kN")
print(f"Bending Stress: {bending_stress/1e6:.1f} MPa")
print(f"Shear Stress: {shear_stress/1e6:.1f} MPa")

# Safety factor check
yield_strength = 250e6  # Steel yield strength
safety_factor = yield_strength / bending_stress
print(f"Safety Factor: {safety_factor:.2f}")

# 6. Matrix Operations for Structural Analysis
print("\n6. MATRIX OPERATIONS")
print("-" * 30)

# Simple truss analysis using matrix methods
# Global stiffness matrix for a simple 2D truss
K = np.array([
    [1.0, 0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0, -1.0],
    [-1.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, 1.0]
])

# Load vector
F = np.array([0, -10e3, 0, 0])  # Applied loads

# Solve for displacements
try:
    displacements = np.linalg.solve(K, F)
    print("Truss Analysis Results:")
    print("Node Displacements (m):")
    for i, disp in enumerate(displacements):
        print(f"  Node {i//2 + 1}, {'X' if i%2==0 else 'Y'}: {disp*1000:.2f} mm")
except np.linalg.LinAlgError:
    print("Matrix is singular - cannot solve")

# 7. Statistical Analysis of Material Properties
print("\n7. STATISTICAL ANALYSIS")
print("-" * 30)

# Simulate material strength test data
np.random.seed(42)
test_results = np.random.normal(250e6, 10e6, 50)  # 50 test results with mean 250 MPa, std 10 MPa

print("Material Strength Test Results:")
print(f"Number of tests: {len(test_results)}")
print(f"Mean strength: {np.mean(test_results)/1e6:.1f} MPa")
print(f"Standard deviation: {np.std(test_results)/1e6:.1f} MPa")
print(f"Minimum strength: {np.min(test_results)/1e6:.1f} MPa")
print(f"Maximum strength: {np.max(test_results)/1e6:.1f} MPa")
print(f"Coefficient of variation: {np.std(test_results)/np.mean(test_results)*100:.1f}%")

# Calculate design strength (mean - 2 standard deviations)
design_strength = np.mean(test_results) - 2 * np.std(test_results)
print(f"Design strength (mean - 2σ): {design_strength/1e6:.1f} MPa")

print("\n" + "="*60)
print("STRUCTURAL ANALYSIS WITH NUMPY COMPLETED!")
print("="*60)
