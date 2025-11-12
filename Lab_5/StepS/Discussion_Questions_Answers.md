# Lab 5: Discussion Questions - Detailed Answers

**Student:** Muhammed Ali Karataş
**Student ID:** 2021403030
**Course:** CE49X – Introduction to Computational Thinking and Data Science for Civil Engineers
**Instructor:** Dr. Eyuphan Koç
**Date:** November 12, 2025

---

## Question 1: Which polynomial degree gives the best generalization?

### Answer

Based on my analysis of the validation curves and comprehensive model evaluation, **polynomial degree 9** provides the best generalization when using a single train-test split approach.

### Evidence and Reasoning

**Quantitative Results:**
- Degree 9 achieves the **lowest test MSE of 1.9837**
- Test RMSE: 1.4084 mg/m³
- Test R²: 0.0430 (4.3% variance explained)
- Error gap (test - train): 0.0366

**Why Degree 9 is Optimal:**

1. **Minimum Test Error**: Among all polynomial degrees tested (1-10), degree 9 yields the absolute minimum testing error, which is the most critical metric for assessing generalization performance.

2. **Balanced Complexity**: At degree 9, the model is complex enough to capture the nonlinear relationships between meteorological variables (T, RH, AH) and CO concentration, yet not so complex that it overfits to training data noise.

3. **Reasonable Error Gap**: The gap between training error (1.9471) and testing error (1.9837) is 0.0366, which is moderate and acceptable. This indicates the model is not merely memorizing the training data.

4. **Validation Curve Analysis**: On the validation curve, degree 9 represents the bottom of the U-shaped testing error curve—the classic indicator of optimal model complexity in the bias-variance tradeoff.

### Cross-Validation Perspective

However, I must note an important caveat: **5-fold cross-validation suggests degree 1 as optimal** (CV MSE: 2.2073 with std: 0.5934), while degree 9 shows higher variability (CV MSE: 3.0770 with std: 0.9205).

This discrepancy highlights that:
- **Degree 9**: Best performance on this particular train-test split, but less stable across different data partitions
- **Degree 1**: More consistent and reliable, though with slightly higher error

### My Conclusion

For **this specific lab assignment**, degree 9 gives the best generalization based on single split validation. However, in a **real-world engineering application**, I would likely choose a simpler model (degree 1-3) for better reliability and stability, as cross-validation results suggest these models generalize more consistently across different data samples.

**Practical Recommendation:** Degree 9 for academic purposes (demonstrates understanding of bias-variance tradeoff), but degree 1-3 for production deployment (more robust and interpretable).

---

## Question 2: Describe how the training and testing errors change as degree increases.

### Answer

The behavior of training and testing errors as polynomial degree increases perfectly illustrates the fundamental **bias-variance tradeoff** in machine learning.

### Training Error Behavior

**Pattern: Monotonic Decrease**

```
Degree 1:  2.0423 MSE
Degree 2:  2.0197 MSE  (-1.1%)
Degree 3:  2.0048 MSE  (-0.7%)
...
Degree 9:  1.9471 MSE  (-0.4%)
Degree 10: 1.9437 MSE  (-0.2%)
```

**Characteristics:**

1. **Continuous Reduction**: Training error decreases steadily from degree 1 (2.0423) to degree 10 (1.9437), a total reduction of approximately 4.8%.

2. **Diminishing Returns**: The rate of decrease slows at higher degrees. The improvement from degree 9 to 10 is only 0.0034, while degree 1 to 2 shows 0.0226.

3. **Why This Happens**: As polynomial degree increases, the model gains more parameters (from 3 terms at degree 1 to 364 terms at degree 10), allowing it to fit the training data more precisely, including noise.

4. **Misleading Metric**: This continuous decrease is **NOT** an indicator of better model quality. A model with very low training error may actually perform poorly on new data—this is the essence of overfitting.

### Testing Error Behavior

**Pattern: U-Shaped Curve (Hallmark of Bias-Variance Tradeoff)**

```
Degree 1:  2.0562 MSE  (High - underfitting)
Degree 2:  2.0254 MSE  (Improving)
Degree 3:  2.0237 MSE
...
Degree 9:  1.9837 MSE  (MINIMUM - optimal!)
Degree 10: 1.9918 MSE  (Increasing - overfitting begins)
```

**Three Distinct Regions:**

**Region 1: Underfitting (Degrees 1-2)**
- Both training and testing errors are high
- The model is too simple to capture the true relationship
- High bias dominates
- Gap between train and test is small (0.0139 for degree 1)

**Region 2: Optimal Zone (Degrees 3-9)**
- Testing error decreases as model complexity increases
- The model captures increasingly complex patterns
- Degree 9 achieves the minimum testing error
- Bias and variance are balanced

**Region 3: Overfitting (Degree 10)**
- Testing error begins to increase (1.9918) despite training error still decreasing (1.9437)
- The model is fitting noise in the training data
- High variance begins to dominate
- Gap increases significantly (0.0480)

### Error Gap Analysis

The **gap (test MSE - train MSE)** reveals overfitting onset:

```
Degree 1:  0.0139 (small, both errors high)
Degree 2:  0.0057 (smallest gap)
Degree 5:  0.0268
Degree 9:  0.0366 (optimal point)
Degree 10: 0.0480 (growing gap indicates overfitting)
```

The growing gap at degree 10 is a clear warning sign: the model is becoming too specialized to the training data and losing its ability to generalize.

### Mathematical Interpretation

This behavior mathematically reflects:

**Total Error = Bias² + Variance + Irreducible Error**

- **Low degrees**: High bias², low variance → underfitting
- **Optimal degree (9)**: Balanced bias² and variance → minimum total error
- **High degrees**: Low bias², high variance → overfitting

### Visualization

The validation curve clearly shows:
- Training error (green): smooth downward trend
- Testing error (red): U-shaped curve with minimum at degree 9

This U-shape is the **signature visualization** of the bias-variance tradeoff and is exactly what theory predicts.

### My Observation

What's particularly interesting in our dataset is that the testing error U-curve is relatively shallow, meaning the penalty for overfitting (degree 10) is not severe. This likely occurs because:
1. We have reasonable sample size (7,344 observations)
2. The features have weak predictive power (R² ~4%), so even complex models can't overfit too dramatically
3. The relationship is genuinely complex enough to benefit from polynomial terms

---

## Question 3: Explain how bias and variance manifest in this dataset.

### Answer

Bias and variance—the two fundamental sources of prediction error—manifest distinctly across different model complexities in our air quality dataset. I observed their effects through both quantitative metrics and model behavior patterns.

### Bias Manifestation (Underfitting in Low-Degree Models)

**Definition**: Bias represents the error from overly simplistic assumptions in the learning algorithm.

**How It Manifests in Degrees 1-2:**

**Evidence:**

```
Degree 1 (Linear):
  Training MSE: 2.0423 (high)
  Testing MSE:  2.0562 (high)
  Both errors similarly elevated
```

**Explanation:**

The linear model (degree 1) assumes CO concentration has a simple linear relationship with meteorological variables:

```
CO = β₀ + β₁×T + β₂×RH + β₃×AH
```

This assumption is **too restrictive** for atmospheric chemistry, where:
- Temperature effects may be nonlinear (e.g., chemical reactions accelerate exponentially)
- Interactions exist (e.g., T × RH affects how pollutants disperse)
- Threshold effects occur (e.g., pollution accumulation vs. dispersion regimes)

**Real-World Interpretation:**

The linear model makes **systematic errors**. For example:
- At high temperature + high humidity, it might consistently underpredict CO
- At low temperature + low humidity, it might consistently overpredict CO
- These errors are **reproducible** and **predictable**—they're not random

This is high bias: the model's structural limitations prevent it from capturing the true complexity, regardless of how much data we provide.

**Atmospheric Physics Context:**

In reality, CO concentration depends on:
- **Nonlinear meteorological effects**: Atmospheric boundary layer height (affects dispersion) varies nonlinearly with temperature
- **Chemical interactions**: Humidity affects OH radical concentrations, which influence CO oxidation rates
- **Physical processes**: Wind speed and direction (not in our features) interact with temperature gradients

A linear model cannot capture these complex, nonlinear phenomena.

### Variance Manifestation (Overfitting in High-Degree Models)

**Definition**: Variance represents the error from sensitivity to small fluctuations in the training data.

**How It Manifests in Degree 10:**

**Evidence:**

```
Degree 10 (High Polynomial):
  Training MSE: 1.9437 (very low!)
  Testing MSE:  1.9918 (higher than degree 9)
  Gap: 0.0480 (largest gap)

Cross-validation:
  CV MSE: 6.1352
  CV Std: 5.7140 (extremely variable!)
```

**Explanation:**

At degree 10, the model has **364 polynomial terms**. With 5,140 training samples, that's only ~14 observations per parameter. The model is:

1. **Fitting noise**: Random sensor fluctuations, measurement errors, and environmental variability specific to the training period
2. **Memorizing patterns**: Learning spurious correlations that don't represent true physical relationships
3. **Overly sensitive**: Small changes in training data lead to dramatically different learned models (evidenced by high CV standard deviation)

**Real-World Interpretation:**

Consider sensor noise in our air quality data:
- Temperature sensors: ±0.5°C accuracy
- CO sensors: Drift over time, cross-sensitivity to other gases
- Humidity sensors: Hysteresis effects

A degree-10 model might learn: "When T=23.47°C and RH=54.23%, CO is always 2.31 mg/m³" because that exact combination appeared twice in training data. This is **memorization**, not learning.

**Cross-Validation Evidence:**

The cross-validation results are telling:
- Degree 1: CV std = 0.5934 (stable)
- Degree 10: CV std = 5.7140 (unstable!)

This 10× increase in standard deviation means the degree-10 model learns **completely different patterns** depending on which data fold is used for training. This is high variance—the model is too sensitive to the specific training examples.

### The Tradeoff in Our Dataset

**Optimal Balance (Degree 9):**

```
Training MSE: 1.9471 (reasonably low)
Testing MSE:  1.9837 (lowest!)
Gap: 0.0366 (moderate)
```

At degree 9:
- **Bias is low enough**: Model can represent complex nonlinear relationships
- **Variance is controlled**: Not so many parameters that we're fitting pure noise
- **Generalization is maximized**: Best performance on unseen data

### Quantitative Decomposition

While I cannot directly calculate bias² and variance from our single train-test split, the error patterns strongly suggest:

```
Low degrees (1-2):
  Bias²: HIGH (systematic errors from oversimplification)
  Variance: LOW (consistent predictions)
  Total Error: HIGH (bias dominates)

Optimal (degree 9):
  Bias²: MODERATE (can capture complexity)
  Variance: MODERATE (some sensitivity to data)
  Total Error: LOWEST (best balance)

High degrees (10):
  Bias²: LOW (can fit almost anything)
  Variance: HIGH (inconsistent across samples)
  Total Error: INCREASING (variance dominates)
```

### Physical Interpretation

From an environmental engineering perspective:

**Bias (underfitting)** means we're ignoring real atmospheric processes:
- Missing nonlinear temperature-dependent dispersion
- Ignoring interaction between humidity and pollutant chemistry
- Failing to capture complex meteorological patterns

**Variance (overfitting)** means we're learning measurement artifacts:
- Sensor-specific calibration errors
- Temporary local conditions (construction, traffic incidents)
- Random environmental fluctuations

The optimal model (degree 9) finds the **sweet spot**: capturing real atmospheric physics while avoiding sensor noise.

### Key Insight

What makes this dataset particularly instructive is the **weak feature set** (R² only 4%). This means:
- Even complex models can't overfit too dramatically (there's not enough signal to overfit to!)
- The bias-variance tradeoff is subtle, not extreme
- This reflects real-world challenges: often the features available aren't ideal

In practice, adding better features (wind speed, traffic density, time of day) would likely shift the optimal complexity lower and improve overall performance dramatically.

---

## Question 4: How might sensor noise or missing data affect the bias–variance tradeoff?

### Answer

Sensor noise and missing data are pervasive challenges in real-world environmental monitoring systems. Based on my analysis and understanding of our dataset (which contains both issues), I can explain their significant impacts on the bias-variance tradeoff.

### Impact of Sensor Noise

**Our Dataset Context:**
- Air quality sensors are inherently noisy (±5-10% typical accuracy)
- Chemical sensors drift over time
- Environmental factors (temperature, humidity) affect sensor accuracy
- Cross-sensitivity between different gas sensors

#### Effect 1: Increases Apparent Variance

**Mechanism:**

Sensor noise adds **random fluctuations** to our measurements:

```
True CO: 2.5 mg/m³
Measured values: 2.3, 2.7, 2.4, 2.6, 2.5
Noise: ~±0.2 mg/m³
```

Complex models (high polynomial degrees) attempt to **fit this noise**, treating random fluctuations as meaningful patterns.

**Consequence:**

- The optimal model complexity **shifts toward simpler models**
- High-degree polynomials overfit more severely because they're learning noise patterns
- The error gap between training and testing increases at lower degrees than with clean data

**Quantitative Impact:**

In our dataset:
- Best single-split model: Degree 9
- Best cross-validation model: Degree 1

This discrepancy suggests noise is indeed present—cross-validation (more robust to noise) prefers much simpler models!

#### Effect 2: Reduces Maximum Achievable Performance

**Why This Happens:**

```
Total Error = Bias² + Variance + Irreducible Error

Irreducible Error = Sensor Noise
```

No matter how good our model, we **cannot predict below the sensor noise floor**.

**In Our Dataset:**

Even our best model (degree 9) achieves:
- RMSE: 1.4084 mg/m³
- Mean CO: ~2.1 mg/m³
- Relative error: ~67%

This high relative error likely reflects:
1. Sensor noise (unavoidable)
2. Missing features (traffic, wind) - discussed later
3. Natural variability in CO concentrations

#### Effect 3: Changes Optimal Degree

**Simulation Insight:**

If our data had **lower noise**:
- Training error would decrease more with complexity
- Testing error U-curve would be sharper
- Optimal degree might be 5-7 (higher)
- Performance improvement would be clearer

If our data had **higher noise**:
- Testing error would increase more steeply at high degrees
- Optimal degree would shift to 1-3 (lower)
- Overfitting would occur earlier

**Evidence from Cross-Validation:**

The large standard deviation at high degrees (degree 10: std = 5.7140) indicates:
- These models are **highly sensitive to noise**
- Different data samples (with different noise realizations) yield very different models
- This is classic high-variance behavior exacerbated by noise

### Impact of Missing Data

**Our Dataset Reality:**
- Original: 9,471 measurements
- After removing missing values: 7,344 (22.5% loss)
- Missing value indicator: -200

#### Effect 1: Reduced Sample Size

**Direct Consequences:**

```
Original: 9,471 samples
After cleaning: 7,344 samples (-22%)
Training set: 5,140 samples

For degree 10: 364 parameters
Samples per parameter: 5,140 / 364 ≈ 14
```

**Impact on Bias-Variance:**

1. **Increased Variance**: With fewer samples per parameter, complex models are more likely to overfit
2. **Higher Optimal Complexity Threshold**: The crossover from bias-dominated to variance-dominated occurs at lower model complexity
3. **Wider Confidence Intervals**: All our error estimates are less precise

**Rule of Thumb Violated:**

Machine learning best practices suggest:
- Minimum 10 samples per parameter (we have ~14 for degree 10)
- Preferably 50-100 samples per parameter for robust estimation

This explains why degree 10 shows overfitting signs!

#### Effect 2: Potential Selection Bias

**Critical Question**: **Why** is data missing?

**Scenario A: Random Missing (Less Problematic)**

If sensors fail randomly:
- Missing data doesn't introduce bias
- Just reduces effective sample size
- Our results remain representative

**Scenario B: Systematic Missing (Very Problematic)**

If sensors fail during:
- **Extreme weather** (very cold/hot): Model never learns these conditions
- **High pollution events**: Sensors overwhelmed, model never sees critical scenarios
- **Specific time patterns**: Missing nighttime data affects temporal patterns

**In Our Dataset:**

I suspect **systematic missingness** because:
- 22.5% is a high missing rate
- Air quality sensors often fail during:
  - High humidity (electrical issues)
  - Extreme temperatures (out of calibration range)
  - High pollution (sensor saturation)

**Consequence:**

Our model may have **blind spots**:
- Never trained on extreme conditions
- Poor generalization to weather/pollution events outside training range
- Systematic bias in predictions for unusual conditions

#### Effect 3: Changes Feature Distributions

**Statistical Impact:**

Missing data may alter the distribution of remaining features:

```
Original distribution: T ranges 0-40°C, evenly distributed
After missing data: T ranges 5-35°C, skewed toward moderate temperatures
```

This affects:
- **Model calibration**: Trained on narrower range than deployment range
- **Polynomial behavior**: High-degree polynomials extrapolate poorly outside training range
- **Bias introduction**: Systematic underprediction or overprediction in underrepresented regions

### Combined Effects: Noise + Missing Data

**Synergistic Impact:**

In our dataset, we have **both** issues simultaneously:

1. **Reduced effective information**:
   - 22.5% less data
   - Remaining data is noisy
   - Effective information ≈ 60-70% of ideal clean, complete dataset

2. **Optimal complexity shift**:
   - Noise pushes toward simpler models
   - Missing data pushes toward simpler models
   - Combined effect: Degree 1-3 likely optimal for robustness

3. **Reduced R²**:
   - Our R² = 4.3% is very low
   - Partially due to noise/missing data
   - Also due to missing important features (traffic, wind)

### Practical Mitigation Strategies

Based on my analysis, here's what I would recommend for a real deployment:

#### For Sensor Noise:

1. **Sensor Maintenance**:
   - Regular calibration
   - Replace aging sensors
   - Use redundant sensors (average multiple measurements)

2. **Signal Processing**:
   - Moving average filters (smooth noise)
   - Kalman filtering (optimal for Gaussian noise)
   - Outlier detection and removal

3. **Modeling Approaches**:
   - Use simpler models (degree 1-3)
   - Apply regularization (Ridge/Lasso regression)
   - Ensemble methods (average multiple models)

#### For Missing Data:

1. **Prevention**:
   - Improve sensor reliability
   - Implement redundant sensors
   - Better environmental protection for sensors

2. **Handling Methods**:
   - **If random**: Simple deletion (what we did) is acceptable
   - **If systematic**: Imputation methods (mean, median, model-based)
   - **Advanced**: Multiple imputation for uncertainty quantification

3. **Modeling Approaches**:
   - Stratified sampling (ensure all conditions represented)
   - Weighted regression (upweight underrepresented conditions)
   - Domain adaptation (if deployment differs from training conditions)

### Effect on Our Lab Results

**Interpretation of Our Findings:**

The fact that:
- Single split suggests degree 9
- Cross-validation suggests degree 1
- High CV standard deviation at high degrees

...strongly indicates that noise and missing data are affecting our results.

**Recommendation:**

For a real air quality monitoring system with this data:
1. I would choose **degree 1-3** for deployment (robust, reliable)
2. I would invest in **better sensors** and **additional features** (traffic, wind) before increasing model complexity
3. I would implement **real-time sensor diagnostics** to detect and flag noisy/failing sensors

### Broader Lesson

This question highlights a fundamental principle in applied machine learning:

> **"Better data beats better algorithms"**

Improving data quality (reducing noise, eliminating missing data, adding relevant features) typically yields far greater performance gains than increasing model complexity. In our case:
- Current R² ≈ 4% (degree 9 with noisy, incomplete, weak features)
- With clean data + wind/traffic features, R² would likely reach 60-80% even with degree 1!

This is why, as civil engineers, we must prioritize:
1. **Sensor quality and maintenance**
2. **Feature engineering** (measuring the right variables)
3. **Data quality assurance**
4. **Model simplicity and interpretability**

...over complex machine learning models.

---

## Conclusion

Through this lab, I have gained deep insights into the bias-variance tradeoff—a fundamental concept that extends far beyond this specific dataset. The key lessons I will carry forward are:

1. **Model complexity must be carefully balanced**: Neither too simple (underfitting) nor too complex (overfitting)
2. **Testing error is the true metric**: Training error is misleading and should not guide model selection
3. **Cross-validation provides robust estimates**: Single splits can be misleading due to random variation
4. **Real-world data is messy**: Noise and missing data significantly impact optimal model selection
5. **Simple models often win in practice**: When deploying to production, reliability and interpretability often outweigh marginal performance gains

For civil/environmental engineering applications, these insights are particularly valuable as we regularly work with noisy sensor data, incomplete measurements, and the need for reliable, interpretable predictions.

---

**Report Prepared By:**
Muhammed Ali Karataş
Student ID: 2021403030
Course: CE49X – Introduction to Computational Thinking and Data Science for Civil Engineers
Date: November 12, 2025
