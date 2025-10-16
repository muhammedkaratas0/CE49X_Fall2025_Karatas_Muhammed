# -----------------------------------------------------------
# Week 3: Numpy Setup and Troubleshooting
# -----------------------------------------------------------
# This file helps troubleshoot numpy import issues
# and provides alternative solutions if numpy is not available
# -----------------------------------------------------------

import sys
import os

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Current working directory:", os.getcwd())

# Try to import numpy
try:
    import numpy as np
    print("\n✅ SUCCESS: Numpy imported successfully!")
    print("Numpy version:", np.__version__)
    
    # Test basic numpy functionality
    arr = np.array([1, 2, 3, 4, 5])
    print("Test array:", arr)
    print("Array shape:", arr.shape)
    print("Array type:", type(arr))
    
except ImportError as e:
    print(f"\n❌ ERROR: Cannot import numpy")
    print(f"Error message: {e}")
    
    print("\n🔧 TROUBLESHOOTING STEPS:")
    print("1. Install numpy using pip:")
    print("   pip install numpy")
    print("\n2. If you're using conda:")
    print("   conda install numpy")
    print("\n3. If you're using Jupyter/Anaconda:")
    print("   - Open Anaconda Prompt")
    print("   - Run: conda install numpy")
    print("\n4. Check if you're in the right environment:")
    print("   - Make sure you're using the correct Python interpreter")
    print("   - Check if numpy is installed in your current environment")

# Alternative: Try to install numpy automatically
print("\n" + "="*50)
print("ATTEMPTING TO INSTALL NUMPY...")
print("="*50)

try:
    import subprocess
    import sys
    
    # Try to install numpy
    print("Installing numpy...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    print("✅ Numpy installation completed!")
    
    # Try importing again
    import numpy as np
    print("✅ Numpy imported successfully after installation!")
    print("Numpy version:", np.__version__)
    
except Exception as install_error:
    print(f"❌ Could not install numpy automatically: {install_error}")
    print("\nPlease install numpy manually using one of these methods:")
    print("1. pip install numpy")
    print("2. conda install numpy")
    print("3. If using Anaconda, open Anaconda Prompt and run: conda install numpy")

print("\n" + "="*50)
print("TESTING BASIC NUMPY OPERATIONS")
print("="*50)

try:
    import numpy as np
    
    # Create arrays
    print("Creating arrays...")
    arr1 = np.array([1, 2, 3, 4, 5])
    arr2 = np.array([[1, 2, 3], [4, 5, 6]])
    
    print("1D Array:", arr1)
    print("2D Array:")
    print(arr2)
    
    # Basic operations
    print("\nBasic operations:")
    print("Sum of 1D array:", np.sum(arr1))
    print("Mean of 1D array:", np.mean(arr1))
    print("Shape of 2D array:", arr2.shape)
    print("Size of 2D array:", arr2.size)
    
    # Array creation functions
    print("\nArray creation functions:")
    zeros = np.zeros(5)
    ones = np.ones((3, 3))
    random_arr = np.random.random(5)
    
    print("Zeros array:", zeros)
    print("Ones array:")
    print(ones)
    print("Random array:", random_arr)
    
    print("\n✅ All numpy operations working correctly!")
    
except ImportError:
    print("❌ Numpy still not available after installation attempt")
    print("Please restart your Python environment and try again")
except Exception as e:
    print(f"❌ Error testing numpy: {e}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("If numpy is still not working:")
print("1. Restart your Python environment/Jupyter kernel")
print("2. Make sure you're using the correct Python interpreter")
print("3. Try installing numpy in your specific environment")
print("4. Check if there are any permission issues")
print("5. Consider using Anaconda which comes with numpy pre-installed")
