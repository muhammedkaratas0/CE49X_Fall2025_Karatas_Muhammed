# -----------------------------------------------------------
# Week 3: Simple Numpy Test
# -----------------------------------------------------------
# A simple test to check if numpy is working properly
# -----------------------------------------------------------

print("Testing numpy import...")

try:
    import numpy as np
    print("✅ Numpy imported successfully!")
    print(f"Numpy version: {np.__version__}")
    
    # Create a simple array
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Array: {arr}")
    print(f"Array shape: {arr.shape}")
    print(f"Array type: {type(arr)}")
    
    # Test basic operations
    print(f"Sum: {np.sum(arr)}")
    print(f"Mean: {np.mean(arr)}")
    print(f"Max: {np.max(arr)}")
    print(f"Min: {np.min(arr)}")
    
    print("\n✅ All numpy tests passed!")
    
except ImportError as e:
    print(f"❌ Cannot import numpy: {e}")
    print("\nTo fix this, try:")
    print("1. pip install numpy")
    print("2. conda install numpy")
    print("3. Restart your Python environment")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\nTest completed.")
