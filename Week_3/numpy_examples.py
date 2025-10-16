# -----------------------------------------------------------
# Week 3: Numpy Examples and Practice
# -----------------------------------------------------------
# This file contains various numpy examples for learning
# -----------------------------------------------------------

import numpy as np

print("="*60)
print("NUMPY EXAMPLES AND PRACTICE")
print("="*60)

# 1. Creating Arrays
print("\n1. CREATING ARRAYS")
print("-" * 30)

# From lists
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("1D array:", arr1)
print("2D array:")
print(arr2)

# Using built-in functions
zeros = np.zeros(5)
ones = np.ones((3, 3))
random_arr = np.random.random(5)
range_arr = np.arange(0, 10, 2)  # Start, stop, step
linspace_arr = np.linspace(0, 1, 5)  # Start, stop, num_points

print(f"Zeros: {zeros}")
print(f"Ones array:")
print(ones)
print(f"Random array: {random_arr}")
print(f"Range array: {range_arr}")
print(f"Linspace array: {linspace_arr}")

# 2. Array Properties
print("\n2. ARRAY PROPERTIES")
print("-" * 30)

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Array A:")
print(A)
print(f"Shape: {A.shape}")
print(f"Size: {A.size}")
print(f"Dimensions: {A.ndim}")
print(f"Data type: {A.dtype}")

# 3. Array Operations
print("\n3. ARRAY OPERATIONS")
print("-" * 30)

a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9, 10])

print(f"Array a: {a}")
print(f"Array b: {b}")
print(f"a + b: {a + b}")
print(f"a - b: {a - b}")
print(f"a * b: {a * b}")
print(f"a / b: {a / b}")
print(f"a ** 2: {a ** 2}")

# 4. Mathematical Functions
print("\n4. MATHEMATICAL FUNCTIONS")
print("-" * 30)

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Data: {data}")
print(f"Sum: {np.sum(data)}")
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Standard deviation: {np.std(data)}")
print(f"Min: {np.min(data)}")
print(f"Max: {np.max(data)}")
print(f"Argmin (index of min): {np.argmin(data)}")
print(f"Argmax (index of max): {np.argmax(data)}")

# 5. Array Reshaping
print("\n5. ARRAY RESHAPING")
print("-" * 30)

original = np.arange(12)
print(f"Original array: {original}")
reshaped = original.reshape(3, 4)
print(f"Reshaped to (3,4):")
print(reshaped)
flattened = reshaped.flatten()
print(f"Flattened back: {flattened}")

# 6. Array Indexing and Slicing
print("\n6. ARRAY INDEXING AND SLICING")
print("-" * 30)

matrix = np.array([[1, 2, 3, 4], 
                   [5, 6, 7, 8], 
                   [9, 10, 11, 12]])
print("Matrix:")
print(matrix)
print(f"Element at [1,2]: {matrix[1, 2]}")
print(f"First row: {matrix[0, :]}")
print(f"Second column: {matrix[:, 1]}")
print(f"Submatrix [0:2, 1:3]:")
print(matrix[0:2, 1:3])

# 7. Boolean Indexing
print("\n7. BOOLEAN INDEXING")
print("-" * 30)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"Array: {arr}")
condition = arr > 5
print(f"Condition (arr > 5): {condition}")
print(f"Elements > 5: {arr[condition]}")
print(f"Elements > 5 (alternative): {arr[arr > 5]}")

# 8. Array Concatenation
print("\n8. ARRAY CONCATENATION")
print("-" * 30)

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Concatenated: {np.concatenate([arr1, arr2])}")
print(f"Stacked vertically:")
print(np.vstack([arr1, arr2]))
print(f"Stacked horizontally:")
print(np.hstack([arr1, arr2]))

# 9. Matrix Operations
print("\n9. MATRIX OPERATIONS")
print("-" * 30)

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"Matrix A:")
print(A)
print(f"Matrix B:")
print(B)
print(f"Matrix multiplication (A @ B):")
print(A @ B)
print(f"Element-wise multiplication (A * B):")
print(A * B)
print(f"Transpose of A:")
print(A.T)

# 10. Statistical Operations
print("\n10. STATISTICAL OPERATIONS")
print("-" * 30)

data = np.random.normal(0, 1, 100)  # 100 random numbers from normal distribution
print(f"Random data (first 10): {data[:10]}")
print(f"Mean: {np.mean(data):.3f}")
print(f"Standard deviation: {np.std(data):.3f}")
print(f"Variance: {np.var(data):.3f}")
print(f"Sum: {np.sum(data):.3f}")
print(f"Product: {np.prod(data[:5]):.3f}")  # Product of first 5 elements

print("\n" + "="*60)
print("NUMPY PRACTICE COMPLETED!")
print("="*60)
