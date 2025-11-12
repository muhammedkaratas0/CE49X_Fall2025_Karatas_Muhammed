# How to Run Lab 5

**Student:** Muhammed Ali Karataş (2021403030)

---

## ✅ Everything is Fixed and Ready!

All files have been:
- ✓ Organized into folders
- ✓ Updated with correct paths
- ✓ Tested and working

---

## 🚀 Three Ways to Run Your Lab

### **Method 1: Jupyter Notebook (RECOMMENDED for Submission)**

```bash
cd /Users/alikaratas/Downloads/lab5
jupyter notebook Lab5_BiasVariance.ipynb
```

**Then:**
1. Your browser will open with the notebook
2. Click **"Kernel"** → **"Restart & Run All"**
3. Wait for all cells to execute (~30 seconds)
4. All results will appear!

**Note:** Make sure you're using Python 3.13 (conda), not Python 3.11 (Homebrew)

---

### **Method 2: Quick Python Script**

```bash
cd /Users/alikaratas/Downloads/lab5
python3 code/run_lab5.py
```

**Results:**
- Generates 2 plots (features & validation curve)
- Shows summary in terminal
- Takes ~15 seconds
- Perfect for quick testing

---

### **Method 3: Full Implementation**

```bash
cd /Users/alikaratas/Downloads/lab5
python3 code/lab5_implementation.py
```

**Results:**
- Generates all 6 plots
- Detailed console output
- Complete analysis
- Takes ~30 seconds

---

## 🔧 If You Get "ModuleNotFoundError"

This means you're using the wrong Python version.

**Check your Python:**
```bash
which python3
# Should show: /Users/alikaratas/miniconda3/bin/python3
```

**If it shows Homebrew path, use this instead:**
```bash
/Users/alikaratas/miniconda3/bin/python3 code/run_lab5.py
```

**Or activate conda:**
```bash
conda activate base
python3 code/run_lab5.py
```

---

## 📊 What Gets Generated

### **From run_lab5.py:**
- `outputs/plot_01_features.png` - Feature relationships
- `outputs/plot_02_validation_curve.png` - Main result

### **From lab5_implementation.py:**
- All 6 PNG files in `outputs/` folder

### **From Jupyter Notebook:**
- All plots displayed inline in the notebook
- Can also save notebook as PDF for submission

---

## ✅ Testing Checklist

**Run this to verify everything works:**

```bash
cd /Users/alikaratas/Downloads/lab5

# Test 1: Quick script
echo "Testing quick script..."
python3 code/run_lab5.py

# Test 2: Check outputs
echo "Checking outputs..."
ls -lh outputs/

# Test 3: View main plot
echo "Opening main plot..."
open outputs/plot_02_validation_curve.png
```

**If all three work, you're ready!** ✅

---

## 📝 For Submission

### **What to Submit:**
1. `Lab5_BiasVariance.ipynb` (main file)

### **Optional Bonus:**
2. `StepS/Lab5_Reflection_MuhammedAliKaratas.md`
3. `StepS/Results_Analysis_and_Interpretation.md`
4. `outputs/` folder (all plots)

### **Before Submitting:**
```bash
# Make sure notebook runs without errors
jupyter notebook Lab5_BiasVariance.ipynb
# Then: Kernel → Restart & Run All
# Check that all cells execute successfully
```

---

## 🐛 Common Issues & Solutions

### **Issue 1: "No module named 'sklearn'"**

**Cause:** Using wrong Python version (Homebrew 3.11 instead of Conda 3.13)

**Solution:**
```bash
conda activate base
python3 code/run_lab5.py
```

---

### **Issue 2: "FileNotFoundError: AirQualityUCI.csv"**

**Cause:** Running script from wrong directory

**Solution:**
```bash
cd /Users/alikaratas/Downloads/lab5
# Make sure you're in the lab5 directory
python3 code/run_lab5.py
```

---

### **Issue 3: Jupyter won't open**

**Cause:** Jupyter not installed

**Solution:**
```bash
conda activate base
pip install jupyter
jupyter notebook Lab5_BiasVariance.ipynb
```

---

## 🎯 Quick Reference

### **View Results:**
```bash
# View plots
open outputs/

# View main validation curve
open outputs/02_validation_curve.png

# View documentation
open README.md
open StepS/Step_by_Step_Implementation.md
```

### **Run Analysis:**
```bash
# Quick (15 seconds)
python3 code/run_lab5.py

# Full (30 seconds)
python3 code/lab5_implementation.py

# Notebook (interactive)
jupyter notebook Lab5_BiasVariance.ipynb
```

---

## 📞 Final Notes

✅ **All paths are correct** - data files moved to `dataset/`
✅ **All scripts updated** - pointing to correct locations
✅ **Everything tested** - working with Python 3.13
✅ **Ready for submission** - notebook at root level

---

## 🎉 You're All Set!

**To run right now:**

```bash
cd /Users/alikaratas/Downloads/lab5
python3 code/run_lab5.py
```

**This will:**
1. Load 7,344 data samples ✓
2. Train 10 polynomial models ✓
3. Generate validation curve ✓
4. Show optimal model (degree 9) ✓
5. Complete in ~15 seconds ✓

**Expected output:**
```
✓ Optimal degree: 9 (Test MSE: 1.9837)
✓ Cross-validation optimal degree: 1 (CV MSE: 2.2073)
✓ Test RMSE: 1.4084 mg/m³
✓ Test R²: 0.0430 (4.3% variance explained)
```

---

**Prepared by:** Muhammed Ali Karataş (2021403030)
**Date:** November 12, 2025
**Status:** ✅ READY TO RUN
