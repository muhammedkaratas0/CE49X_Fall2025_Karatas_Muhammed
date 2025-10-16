#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Numpy Installation Script
This script will help you install numpy if it's not available
"""

import sys
import subprocess
import os

def install_numpy():
    """Install numpy using pip"""
    print("Numpy yüklü değil. Yüklemeye çalışıyorum...")
    print("Numpy is not installed. Trying to install...")
    
    try:
        # Try different installation methods
        commands = [
            [sys.executable, "-m", "pip", "install", "numpy"],
            ["pip3", "install", "numpy"],
            ["pip", "install", "numpy"],
            ["conda", "install", "numpy", "-y"]
        ]
        
        for cmd in commands:
            try:
                print(f"Trying command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    print("✅ Numpy başarıyla yüklendi!")
                    print("✅ Numpy installed successfully!")
                    return True
                else:
                    print(f"❌ Command failed: {result.stderr}")
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"❌ Command failed: {e}")
                continue
        
        print("❌ Numpy yüklenemedi. Manuel olarak yüklemeniz gerekiyor.")
        print("❌ Could not install numpy. You need to install it manually.")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_numpy():
    """Test if numpy is working"""
    try:
        import numpy as np
        print(f"✅ Numpy çalışıyor! Versiyon: {np.__version__}")
        print(f"✅ Numpy is working! Version: {np.__version__}")
        
        # Test basic functionality
        arr = np.array([1, 2, 3, 4, 5])
        print(f"Test array: {arr}")
        print(f"Sum: {np.sum(arr)}")
        
        return True
    except ImportError:
        print("❌ Numpy hala çalışmıyor")
        print("❌ Numpy is still not working")
        return False

def main():
    print("=" * 60)
    print("NUMPY INSTALLATION AND TEST")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # First test if numpy is already available
    if test_numpy():
        print("\n✅ Numpy zaten yüklü ve çalışıyor!")
        print("✅ Numpy is already installed and working!")
        return
    
    # Try to install numpy
    if install_numpy():
        # Test again after installation
        print("\nTesting numpy after installation...")
        test_numpy()
    else:
        print("\n" + "=" * 60)
        print("MANUAL INSTALLATION INSTRUCTIONS")
        print("=" * 60)
        print("1. Terminal/Command Prompt açın")
        print("2. Şu komutlardan birini deneyin:")
        print("   - pip install numpy")
        print("   - pip3 install numpy")
        print("   - conda install numpy")
        print("   - python -m pip install numpy")
        print("\n3. Eğer Anaconda kullanıyorsanız:")
        print("   - Anaconda Prompt açın")
        print("   - conda install numpy")
        print("\n4. Kurulumdan sonra Python'u yeniden başlatın")
        
        print("\n" + "=" * 60)
        print("ALTERNATIVE SOLUTIONS")
        print("=" * 60)
        print("1. Anaconda kullanın (numpy dahil gelir)")
        print("2. Google Colab kullanın (numpy dahil gelir)")
        print("3. Jupyter Notebook kullanın")

if __name__ == "__main__":
    main()
