# -*- coding: utf-8 -*-

"""
Basit Test - Numpy Olmadan
Simple Test - Without Numpy
"""

print("=" * 60)
print("BASIT PYTHON TESTİ - NUMPY OLMADAN")
print("SIMPLE PYTHON TEST - WITHOUT NUMPY")
print("=" * 60)

# 1. Basic Python operations
print("\n1. TEMEL PYTHON İŞLEMLERİ")
print("1. BASIC PYTHON OPERATIONS")
print("-" * 30)

# Lists (like numpy arrays)
my_list = [1, 2, 3, 4, 5]
print(f"List: {my_list}")
print(f"Length: {len(my_list)}")
print(f"Sum: {sum(my_list)}")
print(f"Max: {max(my_list)}")
print(f"Min: {min(my_list)}")

# List operations
list2 = [6, 7, 8, 9, 10]
combined = my_list + list2
print(f"Combined lists: {combined}")

# 2. Mathematical calculations
print("\n2. MATEMATİK HESAPLAMALARI")
print("2. MATHEMATICAL CALCULATIONS")
print("-" * 30)

import math

# Basic math
numbers = [1, 4, 9, 16, 25]
print(f"Numbers: {numbers}")

# Calculate squares manually
squares = [x**2 for x in range(1, 6)]
print(f"Squares: {squares}")

# Calculate square roots
square_roots = [math.sqrt(x) for x in numbers]
print(f"Square roots: {[round(x, 2) for x in square_roots]}")

# 3. Structural calculations without numpy
print("\n3. YAPISAL HESAPLAMALAR (NUMPY OLMADAN)")
print("3. STRUCTURAL CALCULATIONS (WITHOUT NUMPY)")
print("-" * 30)

# Beam properties
L = 8.0  # Length in meters
P = 20e3  # Load in Newtons
E = 200e9  # Young's modulus in Pa
I = 1e-4  # Moment of inertia in m^4

# Calculate maximum deflection
max_deflection = (P * L**3) / (48 * E * I)
print(f"Beam length: {L} m")
print(f"Load: {P/1e3} kN")
print(f"Max deflection: {max_deflection*1000:.2f} mm")

# Calculate stress
M = P * L / 4  # Maximum moment
c = 0.2  # Distance from neutral axis
bending_stress = M * c / I
print(f"Maximum moment: {M/1e3:.1f} kN⋅m")
print(f"Bending stress: {bending_stress/1e6:.1f} MPa")

# 4. Data analysis without numpy
print("\n4. VERİ ANALİZİ (NUMPY OLMADAN)")
print("4. DATA ANALYSIS (WITHOUT NUMPY)")
print("-" * 30)

# Simulate some test data
test_data = [245, 250, 248, 252, 246, 249, 251, 247, 250, 249]

# Calculate statistics manually
mean = sum(test_data) / len(test_data)
variance = sum((x - mean)**2 for x in test_data) / len(test_data)
std_dev = math.sqrt(variance)

print(f"Test data: {test_data}")
print(f"Mean: {mean:.1f} MPa")
print(f"Standard deviation: {std_dev:.1f} MPa")
print(f"Min: {min(test_data)} MPa")
print(f"Max: {max(test_data)} MPa")

# 5. Matrix operations without numpy
print("\n5. MATRİS İŞLEMLERİ (NUMPY OLMADAN)")
print("5. MATRIX OPERATIONS (WITHOUT NUMPY)")
print("-" * 30)

# Simple 2x2 matrix
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

print(f"Matrix A: {A}")
print(f"Matrix B: {B}")

# Matrix addition
C = [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]
print(f"A + B: {C}")

# Matrix multiplication
def matrix_multiply(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

D = matrix_multiply(A, B)
print(f"A × B: {D}")

# 6. Try to import numpy
print("\n6. NUMPY İMPORT DENEMESİ")
print("6. NUMPY IMPORT ATTEMPT")
print("-" * 30)

try:
    import numpy as np
    print("✅ Numpy başarıyla import edildi!")
    print("✅ Numpy imported successfully!")
    print(f"Numpy version: {np.__version__}")
    
    # Test with numpy
    np_array = np.array([1, 2, 3, 4, 5])
    print(f"Numpy array: {np_array}")
    print(f"Sum: {np.sum(np_array)}")
    
except ImportError:
    print("❌ Numpy import edilemedi")
    print("❌ Cannot import numpy")
    print("\nÇözüm önerileri / Solution suggestions:")
    print("1. pip install numpy")
    print("2. pip3 install numpy")
    print("3. conda install numpy")
    print("4. install_numpy.py dosyasını çalıştırın")

print("\n" + "=" * 60)
print("TEST TAMAMLANDI!")
print("TEST COMPLETED!")
print("=" * 60)
