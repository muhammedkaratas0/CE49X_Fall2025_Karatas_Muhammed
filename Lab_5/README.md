# Lab 5: Bias-Variance Tradeoff - Organized Project

**Student:** Muhammed Ali Karataş (2021403030)
**Course:** CE49X – Introduction to Computational Thinking and Data Science
**Instructor:** Dr. Eyuphan Koç
**Semester:** Fall 2025

---

## 📁 Project Structure

```
lab5/
├── 📓 Lab5_BiasVariance.ipynb          # ⭐ MAIN DELIVERABLE - Submit this!
│
├── 📂 StepS/                            # Documentation & Learning Materials
│   ├── Step_by_Step_Implementation.md
│   ├── Lab5_Reflection_MuhammedAliKaratas.md
│   └── Results_Analysis_and_Interpretation.md
│
├── 📂 dataset/                          # Raw Data Files
│   ├── AirQualityUCI.csv               # Main dataset (767 KB)
│   └── AirQualityUCI.xlsx              # Excel version
│
├── 📂 outputs/                          # Generated Visualizations
│   ├── 01_feature_relationships.png
│   ├── 02_validation_curve.png         # ⭐ Key plot showing bias-variance tradeoff
│   ├── 03_rmse_comparison.png
│   ├── 04_r2_scores.png
│   ├── 05_error_gap.png
│   └── 06_cross_validation_comparison.png
│
├── 📂 code/                             # Python Scripts
│   ├── lab5_implementation.py          # Full implementation with all plots
│   └── run_lab5.py                     # Simplified runnable version
│
└── 📂 documentation/                    # Project Documentation
    ├── README_Complete_Lab_Package.md   # Comprehensive package guide
    └── lab5 (1).md                      # Original assignment

```

---

## 🚀 Quick Start

### **Option 1: Run the Jupyter Notebook (Recommended for Submission)**

```bash
cd /Users/alikaratas/Downloads/lab5
jupyter notebook Lab5_BiasVariance.ipynb
```

Then: **Cell → Run All**

---

### **Option 2: Run Python Script (Quick Test)**

```bash
cd /Users/alikaratas/Downloads/lab5
python3 code/run_lab5.py
```

Generates key plots and shows results in ~15 seconds.

---

### **Option 3: Run Full Implementation**

```bash
cd /Users/alikaratas/Downloads/lab5
python3 code/lab5_implementation.py
```

Generates all 6 visualizations with detailed analysis.

---

## 📂 Folder Details

### 📂 **StepS/** - Documentation & Learning
- `Step_by_Step_Implementation.md` - Detailed code explanations
- `Lab5_Reflection_MuhammedAliKaratas.md` - Personal learning journey
- `Results_Analysis_and_Interpretation.md` - Complete analysis & discussion answers

### 📂 **dataset/** - Raw Data
- `AirQualityUCI.csv` - 9,471 hourly air quality measurements (Italian station)
- `AirQualityUCI.xlsx` - Excel format (alternative)

### 📂 **outputs/** - Visualizations
All plots are high-resolution (150 DPI) PNG files:
1. Feature relationships scatter plots
2. **Validation curve (main result)** - Shows U-shaped test error
3. RMSE comparison
4. R² scores
5. Error gap analysis (overfitting indicator)
6. Cross-validation comparison

### 📂 **code/** - Python Scripts
- `lab5_implementation.py` - Comprehensive script with detailed output
- `run_lab5.py` - Simplified version for quick testing

### 📂 **documentation/** - Project Info
- `README_Complete_Lab_Package.md` - Full package documentation
- `lab5 (1).md` - Original lab assignment from Dr. Koç

---

## 🎯 Key Results Summary

| Metric | Value |
|--------|-------|
| **Dataset Size (cleaned)** | 7,344 samples |
| **Features Used** | T, RH, AH (3 meteorological variables) |
| **Optimal Degree (Single Split)** | 9 |
| **Optimal Degree (Cross-Validation)** | 1 |
| **Best Test RMSE** | 1.4084 mg/m³ |
| **Best Test R²** | 0.0430 (4.3%) |

### Key Findings:
✅ Training error decreases continuously (NOT useful for selection)
✅ Testing error is U-shaped (demonstrates bias-variance tradeoff)
✅ Cross-validation suggests simpler model (degree 1) is more reliable
✅ Weak feature correlations (r < 0.05) limit overall performance

---

## 📝 For Submission

**Submit to Dr. Eyuphan Koç:**

### Required:
- `Lab5_BiasVariance.ipynb` (in root directory)

### Optional (for bonus/deeper engagement):
- `StepS/Lab5_Reflection_MuhammedAliKaratas.md`
- `StepS/Results_Analysis_and_Interpretation.md`
- `outputs/` folder (all visualizations)

---

## 📚 Documentation Reading Order

For best understanding, read in this sequence:

1. **`StepS/Step_by_Step_Implementation.md`**
   → Understand what each line of code does and why

2. **Run the notebook or script**
   → See the implementation in action

3. **`StepS/Results_Analysis_and_Interpretation.md`**
   → Interpret the results and answer discussion questions

4. **`StepS/Lab5_Reflection_MuhammedAliKaratas.md`**
   → See the complete learning journey

5. **`documentation/README_Complete_Lab_Package.md`**
   → Comprehensive package overview

---

## 🔧 Troubleshooting

### If plots don't generate:
```bash
cd /Users/alikaratas/Downloads/lab5
python3 code/run_lab5.py
```

### If Jupyter doesn't open:
```bash
pip3 install jupyter
# OR
pip install jupyter
```

### If libraries are missing:
```bash
pip3 install pandas numpy scikit-learn matplotlib seaborn
```

---

## 🎓 What This Lab Demonstrates

### Technical Skills:
✅ Python data science stack (pandas, numpy, sklearn, matplotlib)
✅ Machine learning implementation (polynomial regression)
✅ Model evaluation (MSE, RMSE, R², cross-validation)
✅ Data preprocessing and cleaning
✅ Professional visualization

### Conceptual Understanding:
✅ Bias-variance tradeoff (deep understanding)
✅ Underfitting vs overfitting
✅ Train-test methodology
✅ Cross-validation importance
✅ Model selection principles

### Professional Qualities:
✅ Systematic organization
✅ Thorough documentation
✅ Clear communication
✅ Reproducible research
✅ Real-world engineering perspective

---

## 🏆 Lab Status

**✅ ALL REQUIREMENTS COMPLETE**

- [x] Data loaded and preprocessed
- [x] Polynomial regression models (degrees 1-10) trained
- [x] Training and testing errors calculated
- [x] Validation curve created and labeled
- [x] Discussion questions answered comprehensively
- [x] Bonus cross-validation implemented
- [x] Professional visualizations generated
- [x] Complete documentation provided

---

## 📞 Quick Reference

### View Results:
```bash
# Open notebook
jupyter notebook Lab5_BiasVariance.ipynb

# Quick run
python3 code/run_lab5.py

# Full analysis
python3 code/lab5_implementation.py
```

### View Documentation:
```bash
# Step-by-step guide
open StepS/Step_by_Step_Implementation.md

# Results analysis
open StepS/Results_Analysis_and_Interpretation.md

# Personal reflection
open StepS/Lab5_Reflection_MuhammedAliKaratas.md
```

### View Plots:
```bash
open outputs/02_validation_curve.png  # Main result
open outputs/  # Open folder to view all
```

---

## ✅ Final Checklist

**For Submission:**
- [x] Notebook runs without errors
- [x] All visualizations generated
- [x] Discussion questions answered
- [x] Bonus section completed
- [x] Professional presentation
- [x] Student name/ID included
- [x] Files organized and ready

**For Learning:**
- [x] Understand bias-variance tradeoff
- [x] Can explain underfitting/overfitting
- [x] Know how to implement polynomial regression
- [x] Can interpret validation curves
- [x] Understand cross-validation
- [x] Ready for exam/discussion

---

## 🎉 Project Complete!

This organized structure provides:
- Clear separation of concerns
- Easy navigation
- Professional organization
- Ready for submission
- Complete documentation
- Reproducible results

**Everything you need is here and properly organized!** 🌟

---

**Prepared by:** Muhammed Ali Karataş (2021403030)
**Date:** November 12, 2025
**Status:** ✅ Complete and Ready for Submission
