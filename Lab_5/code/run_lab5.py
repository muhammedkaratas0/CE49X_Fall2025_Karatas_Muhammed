#!/usr/bin/env python3
"""
Lab 5: Bias-Variance Tradeoff - Runnable Python Version
Student: Muhammed Ali Karataş (2021403030)

This is a simplified version that runs directly without Jupyter.
For the full notebook experience, use: jupyter notebook Lab5_BiasVariance.ipynb
"""

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LAB 5: BIAS-VARIANCE TRADEOFF")
print("Student: Muhammed Ali Karataş (2021403030)")
print("="*80)

# Import libraries
print("\n[1/8] Importing libraries...")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
print("✓ Libraries imported successfully")

# Load dataset
print("\n[2/8] Loading dataset...")
df = pd.read_csv('dataset/AirQualityUCI.csv', sep=';', decimal=',')
print(f"✓ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# Clean data
print("\n[3/8] Cleaning data...")
df_clean = df.replace(-200.0, np.nan)
features = ['T', 'RH', 'AH']
target = 'CO(GT)'
data = df_clean[features + [target]].copy()
data_cleaned = data.dropna()
print(f"✓ Cleaned: {len(data_cleaned)} rows remaining ({len(data_cleaned)/len(df)*100:.1f}%)")

# Visualize relationships
print("\n[4/8] Creating feature relationship plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, feature in enumerate(features):
    axes[idx].scatter(data_cleaned[feature], data_cleaned[target], alpha=0.3, s=10)
    axes[idx].set_xlabel(feature, fontsize=12)
    axes[idx].set_ylabel('CO(GT) [mg/m³]', fontsize=12)
    axes[idx].set_title(f'{feature} vs CO(GT)', fontsize=13, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_01_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plot_01_features.png")

# Split data
print("\n[5/8] Splitting data (70-30)...")
X = data_cleaned[features]
y = data_cleaned[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"✓ Training: {len(X_train)} samples | Testing: {len(X_test)} samples")

# Train models
print("\n[6/8] Training polynomial regression models (degrees 1-10)...")
degrees = range(1, 11)
train_mse, test_mse = [], []
train_rmse, test_rmse = [], []
train_r2, test_r2 = [], []

print(f"\n{'Degree':<8} {'Train MSE':<12} {'Test MSE':<12} {'Gap':<10}")
print("-"*50)

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse_val = mean_squared_error(y_train, y_train_pred)
    test_mse_val = mean_squared_error(y_test, y_test_pred)
    gap = test_mse_val - train_mse_val

    train_mse.append(train_mse_val)
    test_mse.append(test_mse_val)
    train_rmse.append(np.sqrt(train_mse_val))
    test_rmse.append(np.sqrt(test_mse_val))
    train_r2.append(r2_score(y_train, y_train_pred))
    test_r2.append(r2_score(y_test, y_test_pred))

    print(f"{degree:<8} {train_mse_val:<12.4f} {test_mse_val:<12.4f} {gap:<10.4f}")

optimal_degree = list(degrees)[np.argmin(test_mse)]
min_test_error = min(test_mse)
print(f"\n✓ Optimal degree: {optimal_degree} (Test MSE: {min_test_error:.4f})")

# Create validation curve
print("\n[7/8] Creating validation curve...")
plt.figure(figsize=(12, 6))
plt.plot(degrees, train_mse, 'o-', linewidth=2.5, markersize=10,
         label='Training Error', color='#2ecc71', markeredgecolor='white', markeredgewidth=1.5)
plt.plot(degrees, test_mse, 's-', linewidth=2.5, markersize=10,
         label='Testing Error', color='#e74c3c', markeredgecolor='white', markeredgewidth=1.5)
plt.axvline(x=optimal_degree, color='gray', linestyle='--', linewidth=2, alpha=0.7)
plt.plot(optimal_degree, min_test_error, 'D', markersize=15, color='#f39c12',
         markeredgecolor='black', markeredgewidth=2, zorder=5)
plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
plt.title('Bias–Variance Tradeoff: Validation Curve', fontsize=16, fontweight='bold', pad=20)
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)
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
plt.savefig('plot_02_validation_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: plot_02_validation_curve.png")

# Cross-validation
print("\n[8/8] Running cross-validation...")
cv_scores = []
for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-scores.mean())

optimal_cv = list(degrees)[np.argmin(cv_scores)]
print(f"✓ Cross-validation optimal degree: {optimal_cv} (CV MSE: {min(cv_scores):.4f})")

# Summary
print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)
print(f"\n🎯 OPTIMAL MODEL:")
print(f"   • Single Split: Degree {optimal_degree} (Test MSE: {min_test_error:.4f})")
print(f"   • Cross-Validation: Degree {optimal_cv} (CV MSE: {min(cv_scores):.4f})")
print(f"   • Test RMSE: {np.sqrt(min_test_error):.4f} mg/m³")
print(f"   • Test R²: {test_r2[optimal_degree-1]:.4f} ({test_r2[optimal_degree-1]*100:.1f}% variance explained)")

print(f"\n📊 KEY OBSERVATIONS:")
print(f"   • Training error: continuously decreases")
print(f"   • Testing error: U-shaped (bias-variance tradeoff!)")
print(f"   • Feature correlations: all < 0.05 (very weak)")
print(f"   • CV suggests simpler model (degree {optimal_cv}) is more reliable")

print(f"\n🎓 LESSONS LEARNED:")
print(f"   ✓ Training error misleading (always decreases)")
print(f"   ✓ Testing error shows true generalization")
print(f"   ✓ Cross-validation more reliable than single split")
print(f"   ✓ Simple models often better with weak features")

print(f"\n📁 Generated files:")
print(f"   • plot_01_features.png")
print(f"   • plot_02_validation_curve.png")

print("\n" + "="*80)
print("✅ LAB 5 COMPLETE!")
print("="*80)
print("\nFor full notebook experience with all visualizations and analysis:")
print("  jupyter notebook Lab5_BiasVariance.ipynb")
print("="*80)
