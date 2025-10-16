# Numpy Kurulum Rehberi / Numpy Installation Guide

## Türkçe

### Problem
`ModuleNotFoundError: No module named 'numpy'` hatası alıyorsunuz.

### Çözüm Adımları

#### 1. Python Sürümünü Kontrol Edin
```bash
python --version
# veya
python3 --version
```

#### 2. Numpy Kurulum Yöntemleri

**Yöntem 1: pip ile kurulum**
```bash
pip install numpy
# veya
pip3 install numpy
```

**Yöntem 2: python -m pip ile kurulum**
```bash
python -m pip install numpy
# veya
python3 -m pip install numpy
```

**Yöntem 3: conda ile kurulum (Anaconda kullanıyorsanız)**
```bash
conda install numpy
```

#### 3. Kurulumu Test Edin
```bash
python -c "import numpy; print('Numpy version:', numpy.__version__)"
```

#### 4. Sorun Devam Ederse

1. **Python ortamınızı kontrol edin:**
   - Hangi Python kullanıyorsunuz? (Anaconda, system Python, vs.)
   - Doğru ortamda mısınız?

2. **Anaconda kullanıyorsanız:**
   - Anaconda Prompt açın
   - `conda install numpy` komutunu çalıştırın

3. **Sistem Python kullanıyorsanız:**
   - `pip3 install --user numpy` deneyin

### Alternatif Çözümler

1. **Google Colab kullanın** - Numpy dahil gelir
2. **Jupyter Notebook kullanın** - Genellikle numpy dahil gelir
3. **Anaconda kurun** - Numpy dahil gelir

---

## English

### Problem
You're getting `ModuleNotFoundError: No module named 'numpy'` error.

### Solution Steps

#### 1. Check Python Version
```bash
python --version
# or
python3 --version
```

#### 2. Numpy Installation Methods

**Method 1: Using pip**
```bash
pip install numpy
# or
pip3 install numpy
```

**Method 2: Using python -m pip**
```bash
python -m pip install numpy
# or
python3 -m pip install numpy
```

**Method 3: Using conda (if using Anaconda)**
```bash
conda install numpy
```

#### 3. Test Installation
```bash
python -c "import numpy; print('Numpy version:', numpy.__version__)"
```

#### 4. If Problem Persists

1. **Check your Python environment:**
   - Which Python are you using? (Anaconda, system Python, etc.)
   - Are you in the correct environment?

2. **If using Anaconda:**
   - Open Anaconda Prompt
   - Run `conda install numpy`

3. **If using system Python:**
   - Try `pip3 install --user numpy`

### Alternative Solutions

1. **Use Google Colab** - Numpy included
2. **Use Jupyter Notebook** - Usually includes numpy
3. **Install Anaconda** - Numpy included

---

## Dosyalar / Files

Bu klasördeki dosyalar:
- `install_numpy.py` - Otomatik kurulum scripti
- `simple_test_without_numpy.py` - Numpy olmadan test
- `numpy_installation_guide.md` - Bu rehber

Files in this folder:
- `install_numpy.py` - Automatic installation script
- `simple_test_without_numpy.py` - Test without numpy
- `numpy_installation_guide.md` - This guide

