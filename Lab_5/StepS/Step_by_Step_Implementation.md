# Lab 5: Step-by-Step Implementation Guide

**Student:** Muhammed Ali Karataş (2021403030)
**Date:** November 12, 2025

---

## 📚 STEP 1: Import Libraries and Setup

### **What We're Doing:**
We're importing all the Python libraries needed for data manipulation, machine learning, and visualization.

### **How We Do It:**
Using `import` statements to load external libraries into our Python environment.

### **Why Each Library:**

```python
import pandas as pd
import numpy as np
```
- **pandas (pd):** Like Excel for Python - handles data in tables (DataFrames)
  - Read CSV files
  - Filter, sort, and manipulate data
  - Handle missing values

- **numpy (np):** Numerical Python - fast mathematical operations
  - Arrays and matrices
  - Mathematical functions (mean, std, sqrt, etc.)
  - Random number generation

```python
from sklearn.model_selection import train_test_split, cross_val_score
```
- **train_test_split:** Splits data into training and testing sets
  - Ensures we have separate data for evaluation
  - Prevents cheating (testing on training data)

- **cross_val_score:** Performs k-fold cross-validation
  - More robust evaluation than single split
  - Tests model on multiple different splits

```python
from sklearn.linear_model import LinearRegression
```
- **LinearRegression:** The actual machine learning model
  - Finds best-fit line (or hyperplane) through data
  - Learns relationship between features and target
  - Used for all polynomial models (after feature transformation)

```python
from sklearn.preprocessing import PolynomialFeatures
```
- **PolynomialFeatures:** Transforms features to create polynomial terms
  - Example: [x₁, x₂] with degree=2 → [x₁, x₂, x₁², x₁x₂, x₂²]
  - Allows linear regression to fit curves
  - Higher degree = more complex patterns

```python
from sklearn.metrics import mean_squared_error, r2_score
```
- **mean_squared_error (MSE):** Measures prediction error
  - Average of squared differences: MSE = (1/n)Σ(actual - predicted)²
  - Lower is better
  - Penalizes large errors more than small ones

- **r2_score (R²):** Measures how much variance is explained
  - Ranges from 0 to 1 (negative if terrible)
  - R² = 1 means perfect predictions
  - R² = 0 means model is no better than predicting the mean

```python
import matplotlib.pyplot as plt
import seaborn as sns
```
- **matplotlib.pyplot (plt):** Main plotting library
  - Create line plots, scatter plots, bar charts
  - Customize colors, labels, titles, legends
  - Professional scientific visualization

- **seaborn (sns):** Makes matplotlib prettier
  - Better default colors and styles
  - Statistical visualizations
  - Built on top of matplotlib

```python
import warnings
warnings.filterwarnings('ignore')
```
- **warnings:** Suppresses warning messages
  - Keeps output clean
  - Warnings often aren't critical errors

---

## 📂 STEP 2: Load and Explore the Dataset

### **What We're Doing:**
Loading the Air Quality CSV file and examining its structure.

### **How We Do It:**

```python
df = pd.read_csv('AirQualityUCI.csv', sep=';', decimal=',')
```

### **Why These Parameters:**

**`sep=';'`** (separator)
- The CSV uses semicolons (;) instead of commas to separate columns
- European format often uses semicolons because commas are decimal separators
- Without this, pandas would read the entire row as one column!

**`decimal=','`** (decimal point)
- The dataset uses commas (,) for decimals instead of periods (.)
- Example: "2,5" means 2.5
- Italian/European number format
- Without this, "2,5" would be read as text, not a number!

### **Exploration Commands:**

```python
df.shape
```
- Returns: (rows, columns)
- Example: (9471, 15) means 9471 measurements, 15 variables
- Tells us dataset size

```python
df.head()
```
- Shows first 5 rows
- Quick preview of data structure
- Check if data loaded correctly

```python
df.columns.tolist()
```
- Lists all column names
- Helps us know what variables are available
- Needed to select features later

```python
df.dtypes
```
- Shows data type of each column (int, float, object/string)
- Identifies if columns were read correctly
- Object type might indicate parsing issues

```python
df.isnull().sum()
```
- Counts missing values (NaN) in each column
- Helps plan data cleaning strategy
- BUT: in this dataset, missing values are marked as -200, not NaN!

```python
df.describe()
```
- Statistical summary: mean, std, min, max, quartiles
- Helps understand data distribution
- Can spot outliers or unusual values

---

## 🧹 STEP 3: Data Preprocessing and Cleaning

### **What We're Doing:**
Cleaning the data and selecting relevant features for our model.

### **How We Do It:**

```python
df_clean = df.replace(-200.0, np.nan)
```

### **Why This Is Critical:**

**Understanding the -200 Missing Value Indicator:**
- The dataset documentation states: "-200 = missing value"
- Sensors sometimes fail or provide invalid readings
- These are marked as -200 (an impossible value for these measurements)
- We must convert these to NaN (Not a Number) so pandas recognizes them as missing
- If we don't, -200 would be treated as a real measurement and skew our model!

```python
features = ['T', 'RH', 'AH']
target = 'CO(GT)'
```

**Why These Features:**

**T (Temperature)** [°C]
- Temperature affects air density and pollutant dispersion
- Warmer air rises, affecting mixing
- Influences chemical reaction rates

**RH (Relative Humidity)** [%]
- Affects particle formation and chemistry
- Influences sensor performance
- Related to atmospheric stability

**AH (Absolute Humidity)** [g/m³]
- Actual water vapor content
- Different from relative humidity
- Affects atmospheric processes

**CO(GT) (Target)** [mg/m³]
- "GT" = Ground Truth (actual measurement)
- Carbon monoxide concentration
- What we're trying to predict
- Dangerous pollutant from combustion

```python
data = df_clean[features + [target]].copy()
```

**Why `.copy()`:**
- Creates independent copy of data
- Prevents modifying original DataFrame
- Avoids "SettingWithCopyWarning"
- Good practice for data safety

```python
data_cleaned = data.dropna()
```

**Why Drop Missing Values:**
- Most machine learning algorithms can't handle NaN
- Could use imputation (filling with mean, median, etc.), but:
  - We have enough data (~9000 rows)
  - Air quality readings: missing data might be non-random (sensor failures during extreme conditions)
  - Dropping is simpler and safer for this lab
- After dropping, we still have plenty of data

---

## 📊 STEP 4: Visualize Feature Relationships

### **What We're Doing:**
Creating scatter plots to see how each feature relates to CO concentration.

### **How We Do It:**

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
```

**What This Means:**
- `subplots(1, 3)`: Create 1 row, 3 columns of plots (one for each feature)
- `figsize=(15, 4)`: Width=15 inches, Height=4 inches
- Returns: `fig` (whole figure) and `axes` (array of plot objects)

```python
for idx, feature in enumerate(features):
    axes[idx].scatter(data_cleaned[feature], data_cleaned[target], alpha=0.3, s=10)
```

**Why Each Parameter:**
- `scatter()`: Creates scatter plot (point cloud)
- `alpha=0.3`: Transparency (0=invisible, 1=solid)
  - Helps see overlapping points (density)
  - 30% opacity lets us see pattern through overlaps
- `s=10`: Size of markers (10 points)
  - Small points for large datasets
  - Prevents overcrowding

**Why Visualize First:**
- See if relationships are linear or nonlinear
- Spot outliers or unusual patterns
- Understand data before modeling
- Guides choice of model complexity
- Confirms features are related to target

---

## ✂️ STEP 5: Split Data into Training and Testing Sets

### **What We're Doing:**
Dividing data into two sets: one for training models, one for testing them.

### **How We Do It:**

```python
X = data_cleaned[features]  # Feature matrix
y = data_cleaned[target]    # Target vector

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

### **Why This Is Essential:**

**The Fundamental Problem:**
- If we train and test on the same data, we can't know if the model generalizes
- It's like memorizing exam answers and taking the same exam
- We'd get perfect scores but learn nothing!

**The Solution:**
- **Training Set (70%):** Model learns patterns from this
- **Testing Set (30%):** Model is evaluated on this (never seen during training)
- This simulates real-world: predicting on new, unseen data

**Why 70-30 Split:**
- 70%: Enough data for model to learn patterns
- 30%: Enough data for reliable evaluation
- Standard practice (alternatives: 80-20, 60-40)
- Tradeoff: More training data = better learning, more test data = better evaluation

**Why `random_state=42`:**
- Makes split reproducible (same split every time)
- 42 is arbitrary (but traditional in programming - Hitchhiker's Guide reference!)
- Without it, different runs give different results
- Essential for debugging and comparing experiments

**Data Shapes After Split:**
- If we have 8000 samples with 3 features:
  - `X_train`: (5600, 3) - 5600 samples, 3 features
  - `X_test`: (2400, 3) - 2400 samples, 3 features
  - `y_train`: (5600,) - 5600 target values
  - `y_test`: (2400,) - 2400 target values

---

## 🤖 STEP 6: Train Polynomial Regression Models (The Main Event!)

### **What We're Doing:**
Training 10 different models with increasing complexity (polynomial degrees 1-10).

### **How We Do It:**

```python
degrees = range(1, 11)  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_mse = []
test_mse = []
```

### **The Training Loop - Line by Line:**

```python
for degree in degrees:
```
- Loop through each polynomial degree (1 to 10)

```python
    poly = PolynomialFeatures(degree=degree, include_bias=False)
```

**What PolynomialFeatures Does:**

**Example with degree=1 (Linear):**
- Input: [T, RH, AH]
- Output: [T, RH, AH]
- No transformation (linear model)

**Example with degree=2 (Quadratic):**
- Input: [T, RH, AH]
- Output: [T, RH, AH, T², T×RH, T×AH, RH², RH×AH, AH²]
- Creates all products up to degree 2
- Now model can fit curves and interactions!

**Example with degree=3 (Cubic):**
- Input: [T, RH, AH]
- Output: All degree 1, 2, AND 3 terms (T³, T²×RH, etc.)
- Even more flexibility!

**Why `include_bias=False`:**
- Bias term (constant 1) isn't needed
- LinearRegression adds its own intercept
- Avoids redundancy

```python
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
```

**Why Two Different Calls:**

**`fit_transform()` on training:**
- **Fit:** Learn the feature names and structure
- **Transform:** Apply the transformation
- Used on training data

**`transform()` on testing:**
- **Only transform**, don't fit again!
- Must use same transformation as training
- Otherwise train and test would have different feature engineering
- This prevents "data leakage"

```python
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
```

**What Happens Here:**
- Create a linear regression model
- `fit()`: Find the best coefficients (weights) to minimize training error
- Math: Finds β in: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
- Uses least squares: minimizes Σ(actual - predicted)²
- Result: model has learned the relationship!

```python
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
```

**Making Predictions:**
- `predict()`: Use learned model to make predictions
- Plugs in feature values and calculates output
- For training: Check how well we fit the training data
- For testing: Check how well we generalize to new data

```python
    train_mse_val = mean_squared_error(y_train, y_train_pred)
    test_mse_val = mean_squared_error(y_test, y_test_pred)
```

**Mean Squared Error (MSE):**

**Formula:** MSE = (1/n) Σ(actual - predicted)²

**Example:**
- Actual CO: [2.0, 3.0, 2.5]
- Predicted: [2.1, 2.8, 2.6]
- Errors: [0.1, -0.2, 0.1]
- Squared: [0.01, 0.04, 0.01]
- MSE = (0.01 + 0.04 + 0.01) / 3 = 0.02

**Why Square the Errors:**
- Makes all errors positive (otherwise they cancel out)
- Penalizes large errors more than small errors
- Mathematical properties (differentiable, convex)

```python
    train_rmse_val = np.sqrt(train_mse_val)
    test_rmse_val = np.sqrt(test_mse_val)
```

**Root Mean Squared Error (RMSE):**

**Why Take Square Root:**
- Returns error to original units (mg/m³)
- More interpretable than MSE
- Example: RMSE = 0.5 mg/m³ means typical error is 0.5 mg/m³

```python
    train_r2_val = r2_score(y_train, y_train_pred)
    test_r2_val = r2_score(y_test, y_test_pred)
```

**R² Score (Coefficient of Determination):**

**Formula:** R² = 1 - (SS_residual / SS_total)

Where:
- SS_residual = Σ(actual - predicted)²
- SS_total = Σ(actual - mean)²

**Interpretation:**
- R² = 1.0: Perfect predictions
- R² = 0.7: Model explains 70% of variance
- R² = 0.0: Model no better than predicting the mean
- R² < 0: Model worse than predicting the mean!

```python
    train_mse.append(train_mse_val)
    test_mse.append(test_mse_val)
    # ... store all metrics
```

**Why Store Results:**
- Need all values for plotting
- Compare across all degrees
- Find optimal degree (minimum test error)

---

## 📈 STEP 7: Create the Validation Curve

### **What We're Doing:**
Visualizing how training and testing errors change with model complexity.

### **How We Do It:**

```python
plt.figure(figsize=(12, 6))
```
- Create new figure, size 12×6 inches
- Larger canvas for detailed plot

```python
plt.plot(degrees, train_mse, 'o-', linewidth=2, markersize=8,
         label='Training Error', color='#2ecc71')
```

**Plot Parameters:**
- `degrees`: x-axis values [1, 2, 3, ..., 10]
- `train_mse`: y-axis values (error for each degree)
- `'o-'`: Style = circles (`o`) connected by lines (`-`)
- `linewidth=2`: Thick line for visibility
- `markersize=8`: Medium circles
- `label`: Text for legend
- `color='#2ecc71'`: Green hex color

```python
plt.plot(degrees, test_mse, 's-', linewidth=2, markersize=8,
         label='Testing Error', color='#e74c3c')
```
- Same as training, but:
  - `'s-'`: Squares instead of circles (easier to distinguish)
  - Red color for contrast

```python
optimal_degree = degrees[np.argmin(test_mse)]
```

**Finding Optimal Degree:**
- `np.argmin(test_mse)`: Index of minimum test error
- Example: if test_mse = [10, 8, 7, 6, 7, 8, 9, 10, 11, 12]
  - Minimum is 6 (at index 3)
  - `degrees[3]` = degree 4 → optimal!

```python
plt.axvline(x=optimal_degree, color='gray', linestyle='--', alpha=0.7)
```
- Vertical line at optimal degree
- Dashed (`--`) for distinction
- Semi-transparent (`alpha=0.7`)
- Visually marks the sweet spot

```python
plt.xlabel('Model Complexity (Polynomial Degree)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
plt.title('Bias–Variance Tradeoff: Validation Curve', fontsize=16, fontweight='bold')
```
- Axes labels and title
- Large, bold fonts for readability
- Clear, descriptive text

```python
plt.text(1.5, max(test_mse) * 0.9, 'Underfitting\n(High Bias)', ...)
```

**Region Labels:**
- Annotates different regions of the plot
- Positioned relative to data (dynamic placement)
- Helps interpret the tradeoff visually
- Educational: shows what each region means

### **What We Should See:**

**Training Error (Green):**
- Starts high (degree 1)
- Continuously decreases
- Very low at degree 10
- Smooth downward trend

**Testing Error (Red):**
- Starts high (degree 1 - underfitting)
- Decreases to a minimum (optimal degree)
- Increases again (high degrees - overfitting)
- **U-shaped curve** ← Key observation!

**The Gap:**
- Small at low degrees (both errors high)
- Moderate at optimal degree
- Large at high degrees (overfitting signature)

---

## 🎓 STEP 8: Interpret the Results

### **What to Look For:**

**Sign of Underfitting (Low Degrees):**
- Both train and test errors are high
- Small gap between them
- Model too simple to capture patterns

**Sign of Optimal Complexity:**
- Test error at minimum
- Reasonable gap
- Best generalization

**Sign of Overfitting (High Degrees):**
- Training error very low
- Testing error high or increasing
- Large gap (model memorizing training data)

### **Expected Optimal Degree:**
- Likely between 2 and 5
- Depends on actual data patterns
- Found by minimum test error

---

## 🎁 BONUS: Cross-Validation

### **What We're Doing:**
Testing on multiple different splits to get more reliable estimates.

### **How 5-Fold Cross-Validation Works:**

**The Process:**
1. Split data into 5 equal parts (folds)
2. Train on folds 1-4, test on fold 5
3. Train on folds 1-3,5, test on fold 4
4. Train on folds 1-2,4-5, test on fold 3
5. Train on folds 1,3-5, test on fold 2
6. Train on folds 2-5, test on fold 1
7. Average all 5 test scores

**Why This Is Better:**
- Single split might be lucky/unlucky
- Uses all data for both training and testing
- More robust estimate of performance
- Reduces variance in evaluation

```python
scores = cross_val_score(model, X_poly, y, cv=5, scoring='neg_mean_squared_error')
```

**Parameters:**
- `cv=5`: 5-fold cross-validation
- `scoring='neg_mean_squared_error'`: What metric to use
- Returns negative MSE (scikit-learn convention: higher is better)
- We negate it to get positive MSE: `mse_scores = -scores`

---

## Summary of Key Concepts

### **The Workflow:**
1. **Import** → Load tools
2. **Load** → Read data
3. **Clean** → Handle missing values
4. **Split** → Train/test separation
5. **Transform** → Polynomial features
6. **Train** → Fit models
7. **Evaluate** → Calculate errors
8. **Visualize** → Plot results
9. **Interpret** → Draw conclusions

### **The Key Insight:**
Model complexity is a tradeoff:
- Too simple → High bias → Underfitting
- Too complex → High variance → Overfitting
- Just right → Minimum test error → Good generalization

This is the **bias-variance tradeoff**!

---

**Ready for implementation! Let's run the code and see the results.**
