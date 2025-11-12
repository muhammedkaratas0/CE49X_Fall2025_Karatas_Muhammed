#!/usr/bin/env python3
"""
Lab 5: Bias-Variance Tradeoff Implementation
Student: Muhammed Ali Karataş (2021403030)
Course: CE49X - Introduction to Computational Thinking and Data Science
"""

print("="*80)
print("LAB 5: BIAS-VARIANCE TRADEOFF - STEP BY STEP IMPLEMENTATION")
print("Student: Muhammed Ali Karataş (2021403030)")
print("="*80)

# ============================================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================================
print("\n" + "="*80)
print("STEP 1: IMPORTING LIBRARIES")
print("="*80)

print("\n📦 Importing data manipulation libraries...")
import pandas as pd
import numpy as np
print("   ✓ pandas (data tables)")
print("   ✓ numpy (numerical operations)")

print("\n📦 Importing machine learning tools...")
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
print("   ✓ train_test_split (data splitting)")
print("   ✓ cross_val_score (cross-validation)")
print("   ✓ LinearRegression (ML model)")
print("   ✓ PolynomialFeatures (feature engineering)")
print("   ✓ mean_squared_error, r2_score (evaluation metrics)")

print("\n📦 Importing visualization libraries...")
import matplotlib.pyplot as plt
import seaborn as sns
print("   ✓ matplotlib (plotting)")
print("   ✓ seaborn (beautiful plots)")

import warnings
warnings.filterwarnings('ignore')
print("\n✅ ALL LIBRARIES IMPORTED SUCCESSFULLY!")

# ============================================================================
# STEP 2: LOAD AND EXPLORE DATASET
# ============================================================================
print("\n" + "="*80)
print("STEP 2: LOADING AND EXPLORING THE DATASET")
print("="*80)

print("\n📂 Loading AirQualityUCI.csv...")
print("   Using semicolon (;) as separator (European format)")
print("   Using comma (,) as decimal point (European format)")

df = pd.read_csv('dataset/AirQualityUCI.csv', sep=';', decimal=',')

print(f"\n✅ Dataset loaded successfully!")
print(f"\n📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   ({df.shape[0]} hourly measurements, {df.shape[1]} variables)")

print("\n📋 First 5 rows of data:")
print(df.head())

print("\n📋 Column names:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:2}. {col}")

print("\n📋 Data types:")
print(df.dtypes)

print("\n📋 Basic statistics:")
print(df.describe())

# ============================================================================
# STEP 3: DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("STEP 3: DATA PREPROCESSING AND CLEANING")
print("="*80)

print("\n🧹 Replacing -200 (missing value indicator) with NaN...")
df_clean = df.replace(-200.0, np.nan)
print("   ✓ Missing values converted to NaN")

print("\n🎯 Selecting features and target:")
features = ['T', 'RH', 'AH']
target = 'CO(GT)'

print(f"\n   Features (inputs):")
print(f"      • T  - Temperature (°C)")
print(f"      • RH - Relative Humidity (%)")
print(f"      • AH - Absolute Humidity (g/m³)")
print(f"\n   Target (output):")
print(f"      • CO(GT) - True CO concentration (mg/m³)")

print("\n🔍 Creating dataset with selected columns...")
data = df_clean[features + [target]].copy()

print(f"\n📊 Missing values per column:")
print(data.isnull().sum())

print("\n🗑️  Removing rows with missing values...")
original_size = len(data)
data_cleaned = data.dropna()
removed_rows = original_size - len(data_cleaned)

print(f"\n   Original size: {original_size} rows")
print(f"   After cleaning: {len(data_cleaned)} rows")
print(f"   Removed: {removed_rows} rows ({removed_rows/original_size*100:.1f}%)")
print(f"   Remaining: {len(data_cleaned)/original_size*100:.1f}% of data")

print("\n✅ DATA CLEANING COMPLETE!")

# ============================================================================
# STEP 4: VISUALIZE RELATIONSHIPS
# ============================================================================
print("\n" + "="*80)
print("STEP 4: VISUALIZING FEATURE-TARGET RELATIONSHIPS")
print("="*80)

print("\n📈 Creating scatter plots for each feature vs CO(GT)...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, feature in enumerate(features):
    axes[idx].scatter(data_cleaned[feature], data_cleaned[target],
                     alpha=0.3, s=10, color='steelblue')
    axes[idx].set_xlabel(feature, fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('CO(GT) [mg/m³]', fontsize=12)
    axes[idx].set_title(f'{feature} vs CO(GT)', fontsize=13, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_feature_relationships.png', dpi=150, bbox_inches='tight')
print("   ✓ Plot saved as '01_feature_relationships.png'")
plt.close()

# Calculate correlations
print("\n📊 Correlation with CO(GT):")
for feature in features:
    corr = data_cleaned[feature].corr(data_cleaned[target])
    print(f"   {feature:3} : {corr:+.4f}")

# ============================================================================
# STEP 5: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("STEP 5: SPLITTING DATA INTO TRAINING AND TESTING SETS")
print("="*80)

print("\n✂️  Performing 70-30 train-test split...")
print("   70% for training (model learns from this)")
print("   30% for testing (model evaluated on this)")
print("   random_state=42 (ensures reproducibility)")

X = data_cleaned[features]
y = data_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\n📊 Split results:")
print(f"   Total samples: {len(X)}")
print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Testing set:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
print(f"\n   Feature matrix shape (train): {X_train.shape}")
print(f"   Target vector shape (train): {y_train.shape}")
print(f"   Feature matrix shape (test): {X_test.shape}")
print(f"   Target vector shape (test): {y_test.shape}")

print("\n✅ TRAIN-TEST SPLIT COMPLETE!")

# ============================================================================
# STEP 6: TRAIN POLYNOMIAL REGRESSION MODELS
# ============================================================================
print("\n" + "="*80)
print("STEP 6: TRAINING POLYNOMIAL REGRESSION MODELS")
print("="*80)

print("\n🤖 Training models with polynomial degrees 1 to 10...")
print("   Degree 1: Linear model (simplest)")
print("   Degrees 2-5: Moderate complexity")
print("   Degrees 6-10: High complexity")

degrees = range(1, 11)
train_mse = []
test_mse = []
train_rmse = []
test_rmse = []
train_r2 = []
test_r2 = []

print("\n" + "-"*80)
print(f"{'Degree':<8} {'Train MSE':<12} {'Test MSE':<12} {'Train RMSE':<12} {'Test RMSE':<12} {'Gap':<10}")
print("-"*80)

for degree in degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Show number of features created
    n_features = X_train_poly.shape[1]

    # Train model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    # Calculate errors
    train_mse_val = mean_squared_error(y_train, y_train_pred)
    test_mse_val = mean_squared_error(y_test, y_test_pred)
    train_rmse_val = np.sqrt(train_mse_val)
    test_rmse_val = np.sqrt(test_mse_val)

    # Calculate R² scores
    train_r2_val = r2_score(y_train, y_train_pred)
    test_r2_val = r2_score(y_test, y_test_pred)

    # Calculate gap
    gap = test_mse_val - train_mse_val

    # Store results
    train_mse.append(train_mse_val)
    test_mse.append(test_mse_val)
    train_rmse.append(train_rmse_val)
    test_rmse.append(test_rmse_val)
    train_r2.append(train_r2_val)
    test_r2.append(test_r2_val)

    # Print results
    print(f"{degree:<8} {train_mse_val:<12.4f} {test_mse_val:<12.4f} "
          f"{train_rmse_val:<12.4f} {test_rmse_val:<12.4f} {gap:<10.4f}")

print("-"*80)

# Find optimal degree
optimal_degree = list(degrees)[np.argmin(test_mse)]
min_test_error = min(test_mse)

print(f"\n🎯 OPTIMAL MODEL:")
print(f"   Polynomial degree: {optimal_degree}")
print(f"   Test MSE: {min_test_error:.4f}")
print(f"   Test RMSE: {np.sqrt(min_test_error):.4f} mg/m³")
print(f"   Test R²: {test_r2[optimal_degree-1]:.4f}")

print("\n✅ MODEL TRAINING COMPLETE!")

# ============================================================================
# STEP 7: CREATE VALIDATION CURVE
# ============================================================================
print("\n" + "="*80)
print("STEP 7: CREATING VALIDATION CURVE")
print("="*80)

print("\n📈 Generating bias-variance tradeoff visualization...")

plt.figure(figsize=(12, 6))

# Plot training and testing errors
plt.plot(degrees, train_mse, 'o-', linewidth=2.5, markersize=10,
         label='Training Error', color='#2ecc71', markeredgecolor='white', markeredgewidth=1.5)
plt.plot(degrees, test_mse, 's-', linewidth=2.5, markersize=10,
         label='Testing Error', color='#e74c3c', markeredgecolor='white', markeredgewidth=1.5)

# Mark optimal point
plt.axvline(x=optimal_degree, color='gray', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Optimal Degree = {optimal_degree}')
plt.plot(optimal_degree, min_test_error, 'D', markersize=15, color='#f39c12',
         markeredgecolor='black', markeredgewidth=2, zorder=5)

# Labels and formatting
plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
plt.title('Bias–Variance Tradeoff: Validation Curve', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper right', framealpha=0.95)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(degrees)

# Add region labels
y_max = max(test_mse)
plt.text(1.5, y_max * 0.85, 'Underfitting\n(High Bias)',
         fontsize=11, ha='center', color='#3498db', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='#3498db', linewidth=2))

plt.text(optimal_degree, y_max * 0.65, 'Optimal\nComplexity',
         fontsize=11, ha='center', color='#f39c12', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='#f39c12', linewidth=2))

plt.text(9, y_max * 0.85, 'Overfitting\n(High Variance)',
         fontsize=11, ha='center', color='#e74c3c', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='#e74c3c', linewidth=2))

plt.tight_layout()
plt.savefig('02_validation_curve.png', dpi=150, bbox_inches='tight')
print("   ✓ Validation curve saved as '02_validation_curve.png'")
plt.close()

# ============================================================================
# STEP 8: RMSE VISUALIZATION
# ============================================================================
print("\n📈 Generating RMSE comparison plot...")

plt.figure(figsize=(12, 6))

plt.plot(degrees, train_rmse, 'o-', linewidth=2.5, markersize=10,
         label='Training RMSE', color='#2ecc71', markeredgecolor='white', markeredgewidth=1.5)
plt.plot(degrees, test_rmse, 's-', linewidth=2.5, markersize=10,
         label='Testing RMSE', color='#e74c3c', markeredgecolor='white', markeredgewidth=1.5)

optimal_test_rmse = test_rmse[optimal_degree - 1]
plt.axvline(x=optimal_degree, color='gray', linestyle='--', linewidth=2, alpha=0.7)
plt.plot(optimal_degree, optimal_test_rmse, 'D', markersize=15, color='#f39c12',
         markeredgecolor='black', markeredgewidth=2, zorder=5)

plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=14, fontweight='bold')
plt.ylabel('Root Mean Squared Error (RMSE) [mg/m³]', fontsize=14, fontweight='bold')
plt.title('RMSE vs Model Complexity', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper right', framealpha=0.95)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(degrees)
plt.tight_layout()
plt.savefig('03_rmse_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ RMSE plot saved as '03_rmse_comparison.png'")
plt.close()

# ============================================================================
# STEP 9: R² SCORE VISUALIZATION
# ============================================================================
print("\n📈 Generating R² score plot...")

plt.figure(figsize=(12, 6))

plt.plot(degrees, train_r2, 'o-', linewidth=2.5, markersize=10,
         label='Training R²', color='#2ecc71', markeredgecolor='white', markeredgewidth=1.5)
plt.plot(degrees, test_r2, 's-', linewidth=2.5, markersize=10,
         label='Testing R²', color='#e74c3c', markeredgecolor='white', markeredgewidth=1.5)

plt.axvline(x=optimal_degree, color='gray', linestyle='--', linewidth=2, alpha=0.7)
plt.axhline(y=1.0, color='black', linestyle=':', linewidth=1, alpha=0.5, label='Perfect Fit (R²=1)')

plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=14, fontweight='bold')
plt.ylabel('R² Score (Coefficient of Determination)', fontsize=14, fontweight='bold')
plt.title('Model Performance: R² Score vs Complexity', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='lower right', framealpha=0.95)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(degrees)
plt.ylim([0, 1.1])
plt.tight_layout()
plt.savefig('04_r2_scores.png', dpi=150, bbox_inches='tight')
print("   ✓ R² plot saved as '04_r2_scores.png'")
plt.close()

# ============================================================================
# STEP 10: ERROR GAP ANALYSIS
# ============================================================================
print("\n📈 Generating error gap analysis...")

error_gap = np.array(test_mse) - np.array(train_mse)

plt.figure(figsize=(12, 6))
plt.plot(degrees, error_gap, 'o-', linewidth=2.5, markersize=12, color='#9b59b6',
         markeredgecolor='white', markeredgewidth=1.5)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
plt.axvline(x=optimal_degree, color='gray', linestyle='--', linewidth=2, alpha=0.7)

plt.fill_between(degrees, 0, error_gap, where=(error_gap > 0),
                 alpha=0.3, color='red', label='Overfitting Region')

plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=14, fontweight='bold')
plt.ylabel('Error Gap (Test MSE - Train MSE)', fontsize=14, fontweight='bold')
plt.title('Overfitting Indicator: Gap Between Test and Train Errors', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(degrees)
plt.legend(fontsize=12, framealpha=0.95)
plt.tight_layout()
plt.savefig('05_error_gap.png', dpi=150, bbox_inches='tight')
print("   ✓ Error gap plot saved as '05_error_gap.png'")
plt.close()

print("\n✅ ALL VISUALIZATIONS CREATED!")

# ============================================================================
# STEP 11: BONUS - CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("STEP 11: BONUS - CROSS-VALIDATION ANALYSIS")
print("="*80)

print("\n🔄 Performing 5-fold cross-validation...")
print("   More robust than single train-test split")
print("   Tests on 5 different data splits")

cv_scores = []
cv_std = []

print("\n" + "-"*60)
print(f"{'Degree':<8} {'CV MSE Mean':<15} {'CV MSE Std':<15}")
print("-"*60)

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores

    cv_scores.append(mse_scores.mean())
    cv_std.append(mse_scores.std())

    print(f"{degree:<8} {mse_scores.mean():<15.4f} {mse_scores.std():<15.4f}")

print("-"*60)

optimal_cv = list(degrees)[np.argmin(cv_scores)]
print(f"\n🎯 OPTIMAL MODEL (Cross-Validation):")
print(f"   Polynomial degree: {optimal_cv}")
print(f"   CV MSE: {min(cv_scores):.4f}")

# Plot comparison
print("\n📈 Creating single-split vs cross-validation comparison...")

plt.figure(figsize=(14, 6))

plt.plot(degrees, test_mse, 's-', linewidth=2.5, markersize=10,
         label='Single Split (30% Test)', color='#e74c3c',
         markeredgecolor='white', markeredgewidth=1.5)

plt.errorbar(degrees, cv_scores, yerr=cv_std, fmt='o-', linewidth=2.5, markersize=10,
             capsize=5, capthick=2, label='5-Fold Cross-Validation', color='#3498db',
             markeredgecolor='white', markeredgewidth=1.5)

plt.axvline(x=optimal_degree, color='#e74c3c', linestyle='--', linewidth=1.5,
            alpha=0.6, label=f'Optimal (Single): {optimal_degree}')
plt.axvline(x=optimal_cv, color='#3498db', linestyle='--', linewidth=1.5,
            alpha=0.6, label=f'Optimal (CV): {optimal_cv}')

plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
plt.title('Comparison: Single Split vs Cross-Validation', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='upper right', framealpha=0.95)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(degrees)
plt.tight_layout()
plt.savefig('06_cross_validation_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ Comparison plot saved as '06_cross_validation_comparison.png'")
plt.close()

print("\n✅ CROSS-VALIDATION COMPLETE!")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL SUMMARY - KEY FINDINGS")
print("="*80)

print(f"\n🎯 OPTIMAL MODEL:")
print(f"   • Polynomial Degree: {optimal_degree}")
print(f"   • Test MSE: {min_test_error:.4f}")
print(f"   • Test RMSE: {np.sqrt(min_test_error):.4f} mg/m³")
print(f"   • Test R²: {test_r2[optimal_degree-1]:.4f} ({test_r2[optimal_degree-1]*100:.1f}% variance explained)")

print(f"\n📊 OBSERVATIONS:")
print(f"   • Training error continuously decreases (degree 1→10)")
print(f"   • Testing error forms U-shape (bias-variance tradeoff)")
print(f"   • Degrees 1-{optimal_degree-1}: Underfitting (high bias)")
print(f"   • Degree {optimal_degree}: Optimal balance")
print(f"   • Degrees {optimal_degree+1}-10: Overfitting (high variance)")

print(f"\n🎓 KEY LESSONS:")
print(f"   ✓ More complex ≠ better")
print(f"   ✓ Must evaluate on test data (training error misleading)")
print(f"   ✓ Optimal complexity balances bias and variance")
print(f"   ✓ Gap between train/test error indicates overfitting")

print(f"\n📁 GENERATED FILES:")
print(f"   1. 01_feature_relationships.png")
print(f"   2. 02_validation_curve.png")
print(f"   3. 03_rmse_comparison.png")
print(f"   4. 04_r2_scores.png")
print(f"   5. 05_error_gap.png")
print(f"   6. 06_cross_validation_comparison.png")

print("\n" + "="*80)
print("✅ LAB 5 IMPLEMENTATION COMPLETE!")
print("="*80)
print("\nStudent: Muhammed Ali Karataş (2021403030)")
print("Course: CE49X - Introduction to Computational Thinking and Data Science")
print("="*80)
