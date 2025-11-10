"""
Lab 4: Statistical Analysis
Descriptive Statistics and Probability Distributions

Author: Ali Karatas
Date: 2025-11-10
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, poisson, uniform, expon
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def load_data(file_path):
    """
    Load dataset from CSV file in root-level datasets/ folder.

    Parameters:
    -----------
    file_path : str
        Name of the CSV file (e.g., 'concrete_strength.csv')

    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    try:
        # Construct path to datasets folder
        full_path = os.path.join('..', 'datasets', file_path)
        df = pd.read_csv(full_path)
        print(f"✓ Successfully loaded {file_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        return df
    except FileNotFoundError:
        print(f"✗ Error: {file_path} not found in '../datasets/' folder")
        return None
    except Exception as e:
        print(f"✗ Error loading {file_path}: {e}")
        return None


def calculate_descriptive_stats(data, column='strength_mpa'):
    """
    Calculate all descriptive statistics for a given column.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    column : str
        Column name to analyze

    Returns:
    --------
    dict
        Dictionary containing all descriptive statistics
    """
    values = data[column].dropna()

    # Central tendency
    mean_val = np.mean(values)
    median_val = np.median(values)
    mode_result = stats.mode(values, keepdims=True)
    mode_val = mode_result.mode[0] if len(mode_result.mode) > 0 else mean_val

    # Spread
    variance = np.var(values, ddof=1)
    std_dev = np.std(values, ddof=1)
    range_val = np.max(values) - np.min(values)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    # Shape
    skewness = stats.skew(values)
    kurtosis_val = stats.kurtosis(values)

    # Quantiles
    min_val = np.min(values)
    max_val = np.max(values)

    stats_dict = {
        'mean': mean_val,
        'median': median_val,
        'mode': mode_val,
        'variance': variance,
        'std_dev': std_dev,
        'range': range_val,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'skewness': skewness,
        'kurtosis': kurtosis_val,
        'min': min_val,
        'max': max_val,
        'count': len(values)
    }

    return stats_dict


def plot_distribution(data, column, title, save_path=None):
    """
    Create distribution plot with statistics marked.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    column : str
        Column name to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    """
    values = data[column].dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Calculate statistics
    mean_val = np.mean(values)
    median_val = np.median(values)
    std_dev = np.std(values, ddof=1)

    # 1. Histogram with density and normal curve
    ax1 = axes[0, 0]
    ax1.hist(values, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # Fit and plot normal distribution
    xmin, xmax = ax1.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_val, std_dev)
    ax1.plot(x, p, 'r-', linewidth=2, label='Normal fit')

    # Mark mean and median
    ax1.axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax1.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')

    ax1.set_xlabel(column)
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution with Normal Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Histogram with standard deviation bands
    ax2 = axes[0, 1]
    ax2.hist(values, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')

    # Mark ±1σ, ±2σ, ±3σ
    ax2.axvline(mean_val, color='red', linewidth=2, label='Mean')
    ax2.axvline(mean_val - std_dev, color='blue', linestyle='--', alpha=0.7, label='±1σ')
    ax2.axvline(mean_val + std_dev, color='blue', linestyle='--', alpha=0.7)
    ax2.axvline(mean_val - 2*std_dev, color='purple', linestyle='--', alpha=0.5, label='±2σ')
    ax2.axvline(mean_val + 2*std_dev, color='purple', linestyle='--', alpha=0.5)
    ax2.axvline(mean_val - 3*std_dev, color='orange', linestyle='--', alpha=0.3, label='±3σ')
    ax2.axvline(mean_val + 3*std_dev, color='orange', linestyle='--', alpha=0.3)

    ax2.set_xlabel(column)
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution with Standard Deviation Bands')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Boxplot
    ax3 = axes[1, 0]
    bp = ax3.boxplot(values, vert=False, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][0].set_alpha(0.7)

    ax3.set_xlabel(column)
    ax3.set_title('Box Plot (Quartiles and Outliers)')
    ax3.grid(True, alpha=0.3)

    # 4. Q-Q Plot
    ax4 = axes[1, 1]
    stats.probplot(values, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (Normality Check)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {save_path}")

    plt.close()


def fit_distribution(data, column, distribution_type='normal'):
    """
    Fit probability distribution to data.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    column : str
        Column name
    distribution_type : str
        Type of distribution ('normal', 'exponential', 'uniform')

    Returns:
    --------
    tuple
        Distribution parameters
    """
    values = data[column].dropna()

    if distribution_type == 'normal':
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        print(f"\nFitted Normal Distribution:")
        print(f"  Mean (μ): {mean_val:.4f}")
        print(f"  Std Dev (σ): {std_val:.4f}")
        return mean_val, std_val

    elif distribution_type == 'exponential':
        mean_val = np.mean(values)
        lambda_val = 1 / mean_val
        print(f"\nFitted Exponential Distribution:")
        print(f"  Lambda (λ): {lambda_val:.4f}")
        print(f"  Mean: {mean_val:.4f}")
        return lambda_val,

    elif distribution_type == 'uniform':
        min_val = np.min(values)
        max_val = np.max(values)
        print(f"\nFitted Uniform Distribution:")
        print(f"  Min (a): {min_val:.4f}")
        print(f"  Max (b): {max_val:.4f}")
        return min_val, max_val

    return None


def calculate_probability_binomial(n, p, k):
    """
    Calculate binomial probabilities.

    Parameters:
    -----------
    n : int
        Number of trials
    p : float
        Probability of success
    k : int or str
        Number of successes, or 'all' for PMF

    Returns:
    --------
    float or array
        Probability value(s)
    """
    if k == 'all':
        return binom.pmf(range(n+1), n, p)
    else:
        return binom.pmf(k, n, p)


def calculate_probability_normal(mean, std, x_lower=None, x_upper=None):
    """
    Calculate normal probabilities.

    Parameters:
    -----------
    mean : float
        Mean of distribution
    std : float
        Standard deviation
    x_lower : float, optional
        Lower bound (None for -∞)
    x_upper : float, optional
        Upper bound (None for +∞)

    Returns:
    --------
    float
        Probability
    """
    if x_lower is None and x_upper is not None:
        # P(X ≤ x_upper)
        return norm.cdf(x_upper, mean, std)
    elif x_lower is not None and x_upper is None:
        # P(X > x_lower)
        return 1 - norm.cdf(x_lower, mean, std)
    elif x_lower is not None and x_upper is not None:
        # P(x_lower < X ≤ x_upper)
        return norm.cdf(x_upper, mean, std) - norm.cdf(x_lower, mean, std)
    else:
        return 1.0


def calculate_probability_poisson(lambda_param, k):
    """
    Calculate Poisson probabilities.

    Parameters:
    -----------
    lambda_param : float
        Rate parameter (mean)
    k : int or str
        Number of events, or 'all' for PMF

    Returns:
    --------
    float or array
        Probability value(s)
    """
    if k == 'all':
        return poisson.pmf(range(int(lambda_param*3)), lambda_param)
    else:
        return poisson.pmf(k, lambda_param)


def calculate_probability_exponential(mean, x):
    """
    Calculate exponential probabilities.

    Parameters:
    -----------
    mean : float
        Mean of distribution
    x : float or str
        Value or 'all' for PDF

    Returns:
    --------
    float or array
        Probability value(s)
    """
    scale = mean  # For exponential, scale = mean

    if x == 'all':
        x_vals = np.linspace(0, mean*5, 100)
        return expon.pdf(x_vals, scale=scale)
    else:
        # Return CDF (probability of failure before x)
        return expon.cdf(x, scale=scale)


def apply_bayes_theorem(prior, sensitivity, specificity):
    """
    Apply Bayes' theorem for diagnostic test scenario.

    Parameters:
    -----------
    prior : float
        Prior probability (base rate)
    sensitivity : float
        True positive rate (P(Test+|Disease+))
    specificity : float
        True negative rate (P(Test-|Disease-))

    Returns:
    --------
    dict
        Dictionary with all probabilities
    """
    # P(Disease) and P(No Disease)
    p_disease = prior
    p_no_disease = 1 - prior

    # P(Test+|Disease+) and P(Test-|Disease-)
    p_test_pos_given_disease = sensitivity
    p_test_neg_given_no_disease = specificity

    # P(Test+|No Disease) = 1 - Specificity (False positive rate)
    p_test_pos_given_no_disease = 1 - specificity

    # P(Test+) using law of total probability
    p_test_pos = (p_test_pos_given_disease * p_disease +
                  p_test_pos_given_no_disease * p_no_disease)

    # Bayes' Theorem: P(Disease|Test+)
    p_disease_given_test_pos = (p_test_pos_given_disease * p_disease) / p_test_pos

    results = {
        'prior': prior,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'p_test_positive': p_test_pos,
        'posterior': p_disease_given_test_pos,
        'false_positive_rate': p_test_pos_given_no_disease
    }

    return results


def plot_material_comparison(data, column, group_column, save_path=None):
    """
    Create comparative boxplot for material types.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    column : str
        Column to compare
    group_column : str
        Grouping column
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Material Property Comparison', fontsize=16, fontweight='bold')

    # Boxplot
    ax1 = axes[0]
    data.boxplot(column=column, by=group_column, ax=ax1, patch_artist=True)
    ax1.set_xlabel('Material Type')
    ax1.set_ylabel(column)
    ax1.set_title('Box Plot Comparison')
    plt.sca(ax1)
    plt.xticks(rotation=45)

    # Violin plot
    ax2 = axes[1]
    sns.violinplot(data=data, x=group_column, y=column, ax=ax2)
    ax2.set_xlabel('Material Type')
    ax2.set_ylabel(column)
    ax2.set_title('Violin Plot Comparison')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {save_path}")

    plt.close()


def plot_distribution_fitting(data, column, fitted_dist=None, save_path=None):
    """
    Visualize fitted distribution with synthetic data comparison.

    Parameters:
    -----------
    data : pd.DataFrame
        Input dataset
    column : str
        Column name
    fitted_dist : tuple, optional
        Fitted distribution parameters (mean, std)
    save_path : str, optional
        Path to save figure
    """
    values = data[column].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Distribution Fitting and Validation', fontsize=16, fontweight='bold')

    # Fit distribution if not provided
    if fitted_dist is None:
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
    else:
        mean_val, std_val = fitted_dist

    # Generate synthetic data
    synthetic_data = np.random.normal(mean_val, std_val, len(values))

    # Plot 1: Original data with fitted distribution
    ax1 = axes[0]
    ax1.hist(values, bins=30, density=True, alpha=0.6, color='blue',
             edgecolor='black', label='Original Data')

    x = np.linspace(values.min(), values.max(), 100)
    ax1.plot(x, norm.pdf(x, mean_val, std_val), 'r-', linewidth=2,
             label=f'Fitted Normal\nμ={mean_val:.2f}, σ={std_val:.2f}')

    ax1.set_xlabel(column)
    ax1.set_ylabel('Density')
    ax1.set_title('Original Data with Fitted Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Comparison with synthetic data
    ax2 = axes[1]
    ax2.hist(values, bins=30, density=True, alpha=0.5, color='blue',
             edgecolor='black', label='Original Data')
    ax2.hist(synthetic_data, bins=30, density=True, alpha=0.5, color='red',
             edgecolor='black', label='Synthetic Data')

    ax2.set_xlabel(column)
    ax2.set_ylabel('Density')
    ax2.set_title('Original vs Synthetic Data Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {save_path}")

    plt.close()

    # Print validation statistics
    print("\nValidation Statistics:")
    print(f"Original Data - Mean: {np.mean(values):.4f}, Std: {np.std(values, ddof=1):.4f}")
    print(f"Synthetic Data - Mean: {np.mean(synthetic_data):.4f}, Std: {np.std(synthetic_data, ddof=1):.4f}")
    print(f"Fitted Parameters - Mean: {mean_val:.4f}, Std: {std_val:.4f}")


def plot_probability_distributions(save_path=None):
    """
    Create comprehensive probability distribution plots.

    Parameters:
    -----------
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Probability Distributions', fontsize=16, fontweight='bold')

    # 1. Binomial Distribution
    ax1 = axes[0, 0]
    n, p = 100, 0.05
    x = np.arange(0, 20)
    pmf = binom.pmf(x, n, p)
    ax1.bar(x, pmf, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(n*p, color='red', linestyle='--', label=f'Mean = {n*p}')
    ax1.set_xlabel('Number of Defects')
    ax1.set_ylabel('Probability')
    ax1.set_title(f'Binomial(n={n}, p={p})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Poisson Distribution
    ax2 = axes[0, 1]
    lambda_val = 10
    x = np.arange(0, 25)
    pmf = poisson.pmf(x, lambda_val)
    ax2.bar(x, pmf, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(lambda_val, color='red', linestyle='--', label=f'Mean = {lambda_val}')
    ax2.set_xlabel('Number of Events')
    ax2.set_ylabel('Probability')
    ax2.set_title(f'Poisson(λ={lambda_val})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Normal Distribution
    ax3 = axes[0, 2]
    mean_val, std_val = 250, 15
    x = np.linspace(mean_val - 4*std_val, mean_val + 4*std_val, 100)
    pdf = norm.pdf(x, mean_val, std_val)
    ax3.plot(x, pdf, linewidth=2, color='purple')
    ax3.fill_between(x, pdf, alpha=0.3, color='purple')
    ax3.axvline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val}')
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Density')
    ax3.set_title(f'Normal(μ={mean_val}, σ={std_val})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Exponential Distribution
    ax4 = axes[1, 0]
    mean_val = 1000
    x = np.linspace(0, mean_val*3, 100)
    pdf = expon.pdf(x, scale=mean_val)
    ax4.plot(x, pdf, linewidth=2, color='orange')
    ax4.fill_between(x, pdf, alpha=0.3, color='orange')
    ax4.axvline(mean_val, color='red', linestyle='--', label=f'Mean = {mean_val}')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Exponential(mean={mean_val})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Uniform Distribution
    ax5 = axes[1, 1]
    a, b = 0, 100
    x = np.linspace(a-10, b+10, 100)
    pdf = uniform.pdf(x, a, b-a)
    ax5.plot(x, pdf, linewidth=2, color='teal')
    ax5.fill_between(x, pdf, alpha=0.3, color='teal')
    ax5.axvline((a+b)/2, color='red', linestyle='--', label=f'Mean = {(a+b)/2}')
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Density')
    ax5.set_title(f'Uniform(a={a}, b={b})')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. CDFs Comparison
    ax6 = axes[1, 2]
    x_norm = np.linspace(200, 300, 100)
    cdf_norm = norm.cdf(x_norm, 250, 15)
    ax6.plot(x_norm, cdf_norm, label='Normal', linewidth=2)

    x_exp = np.linspace(0, 3000, 100)
    cdf_exp = expon.cdf(x_exp, scale=1000)
    ax6.plot(x_exp, cdf_exp, label='Exponential', linewidth=2)

    ax6.set_xlabel('Value')
    ax6.set_ylabel('Cumulative Probability')
    ax6.set_title('CDF Comparison')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {save_path}")

    plt.close()


def plot_bayes_tree(prior, sensitivity, specificity, save_path=None):
    """
    Create probability tree diagram for Bayes' theorem.

    Parameters:
    -----------
    prior : float
        Prior probability
    sensitivity : float
        Sensitivity
    specificity : float
        Specificity
    save_path : str, optional
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate probabilities
    p_disease = prior
    p_no_disease = 1 - prior
    p_test_pos_disease = sensitivity
    p_test_neg_disease = 1 - sensitivity
    p_test_neg_no_disease = specificity
    p_test_pos_no_disease = 1 - specificity

    # Joint probabilities
    p_disease_and_pos = p_disease * p_test_pos_disease
    p_disease_and_neg = p_disease * p_test_neg_disease
    p_no_disease_and_pos = p_no_disease * p_test_pos_no_disease
    p_no_disease_and_neg = p_no_disease * p_test_neg_no_disease

    # Calculate posterior
    bayes_result = apply_bayes_theorem(prior, sensitivity, specificity)
    posterior = bayes_result['posterior']

    # Create text representation
    ax.text(0.5, 0.95, 'Bayes\' Theorem: Structural Damage Detection',
            ha='center', va='top', fontsize=16, fontweight='bold')

    tree_text = f"""
    Prior Probabilities:
        P(Damage) = {p_disease:.4f} ({p_disease*100:.1f}%)
        P(No Damage) = {p_no_disease:.4f} ({p_no_disease*100:.1f}%)

    Conditional Probabilities (Likelihood):
        P(Test+ | Damage) = {sensitivity:.4f} (Sensitivity)
        P(Test- | No Damage) = {specificity:.4f} (Specificity)
        P(Test+ | No Damage) = {p_test_pos_no_disease:.4f} (False Positive)

    Joint Probabilities:
        P(Damage AND Test+) = {p_disease_and_pos:.4f}
        P(Damage AND Test-) = {p_disease_and_neg:.4f}
        P(No Damage AND Test+) = {p_no_disease_and_pos:.4f}
        P(No Damage AND Test-) = {p_no_disease_and_neg:.4f}

    Posterior Probability (Using Bayes' Theorem):
        P(Damage | Test+) = {posterior:.4f} ({posterior*100:.1f}%)

    Interpretation:
        If the test is positive, there is a {posterior*100:.1f}% probability
        that the structure actually has damage.

        This is much higher than the prior probability of {p_disease*100:.1f}%,
        demonstrating how the test result updates our belief.
    """

    ax.text(0.1, 0.85, tree_text, ha='left', va='top', fontsize=11,
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot: {save_path}")

    plt.close()


def create_statistical_report(concrete_data, material_data, output_file='lab4_statistical_report.txt'):
    """
    Create a statistical report summarizing findings.

    Parameters:
    -----------
    concrete_data : pd.DataFrame
        Concrete strength dataset
    material_data : pd.DataFrame
        Material properties dataset
    output_file : str
        Output file name
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LAB 4: STATISTICAL ANALYSIS REPORT\n")
        f.write("Descriptive Statistics and Probability Distributions\n")
        f.write("="*80 + "\n\n")

        # Part 1: Concrete Strength Analysis
        f.write("PART 1: CONCRETE STRENGTH ANALYSIS\n")
        f.write("-"*80 + "\n\n")

        stats_dict = calculate_descriptive_stats(concrete_data, 'strength_mpa')

        f.write("Descriptive Statistics:\n")
        f.write(f"  Count:              {stats_dict['count']}\n")
        f.write(f"  Mean:               {stats_dict['mean']:.4f} MPa\n")
        f.write(f"  Median:             {stats_dict['median']:.4f} MPa\n")
        f.write(f"  Mode:               {stats_dict['mode']:.4f} MPa\n")
        f.write(f"  Standard Deviation: {stats_dict['std_dev']:.4f} MPa\n")
        f.write(f"  Variance:           {stats_dict['variance']:.4f} MPa²\n")
        f.write(f"  Range:              {stats_dict['range']:.4f} MPa\n")
        f.write(f"  IQR:                {stats_dict['iqr']:.4f} MPa\n")
        f.write(f"  Skewness:           {stats_dict['skewness']:.4f}\n")
        f.write(f"  Kurtosis:           {stats_dict['kurtosis']:.4f}\n\n")

        f.write("Five-Number Summary:\n")
        f.write(f"  Minimum: {stats_dict['min']:.4f} MPa\n")
        f.write(f"  Q1:      {stats_dict['q1']:.4f} MPa\n")
        f.write(f"  Median:  {stats_dict['median']:.4f} MPa\n")
        f.write(f"  Q3:      {stats_dict['q3']:.4f} MPa\n")
        f.write(f"  Maximum: {stats_dict['max']:.4f} MPa\n\n")

        f.write("Interpretation:\n")
        if abs(stats_dict['skewness']) < 0.5:
            f.write("  - Distribution is approximately symmetric\n")
        elif stats_dict['skewness'] > 0.5:
            f.write("  - Distribution is positively skewed (right tail)\n")
        else:
            f.write("  - Distribution is negatively skewed (left tail)\n")

        if abs(stats_dict['kurtosis']) < 0.5:
            f.write("  - Kurtosis is near normal (mesokurtic)\n")
        elif stats_dict['kurtosis'] > 0.5:
            f.write("  - Distribution is heavy-tailed (leptokurtic)\n")
        else:
            f.write("  - Distribution is light-tailed (platykurtic)\n")

        cv = (stats_dict['std_dev'] / stats_dict['mean']) * 100
        f.write(f"  - Coefficient of Variation: {cv:.2f}%\n")
        if cv < 15:
            f.write("  - Low variability (good quality control)\n")
        elif cv < 30:
            f.write("  - Moderate variability\n")
        else:
            f.write("  - High variability (quality control needed)\n")
        f.write("\n")

        # Part 2: Material Comparison
        f.write("PART 2: MATERIAL PROPERTIES COMPARISON\n")
        f.write("-"*80 + "\n\n")

        grouped = material_data.groupby('material_type')['yield_strength_mpa'].agg(['mean', 'std', 'min', 'max'])
        f.write("Summary by Material Type:\n")
        f.write(grouped.to_string())
        f.write("\n\n")

        # Part 3: Probability Applications
        f.write("PART 3: PROBABILITY APPLICATIONS\n")
        f.write("-"*80 + "\n\n")

        f.write("3.1 Binomial Distribution (Quality Control):\n")
        f.write("    Scenario: 100 components tested, 5% defect rate\n")
        p_exactly_3 = calculate_probability_binomial(100, 0.05, 3)
        p_le_5 = sum([calculate_probability_binomial(100, 0.05, k) for k in range(6)])
        f.write(f"    P(exactly 3 defects) = {p_exactly_3:.4f}\n")
        f.write(f"    P(≤ 5 defects) = {p_le_5:.4f}\n\n")

        f.write("3.2 Poisson Distribution (Bridge Loads):\n")
        f.write("    Scenario: Average 10 heavy trucks per hour\n")
        p_exactly_8 = calculate_probability_poisson(10, 8)
        p_gt_15 = 1 - sum([calculate_probability_poisson(10, k) for k in range(16)])
        f.write(f"    P(exactly 8 trucks) = {p_exactly_8:.4f}\n")
        f.write(f"    P(> 15 trucks) = {p_gt_15:.4f}\n\n")

        f.write("3.3 Normal Distribution (Steel Yield Strength):\n")
        f.write("    Scenario: Mean = 250 MPa, Std = 15 MPa\n")
        p_exceeds_280 = calculate_probability_normal(250, 15, x_lower=280)
        percentile_95 = norm.ppf(0.95, 250, 15)
        f.write(f"    P(X > 280 MPa) = {p_exceeds_280:.4f} ({p_exceeds_280*100:.2f}%)\n")
        f.write(f"    95th percentile = {percentile_95:.4f} MPa\n\n")

        f.write("3.4 Exponential Distribution (Component Lifetime):\n")
        f.write("    Scenario: Mean lifetime = 1000 hours\n")
        p_fail_500 = calculate_probability_exponential(1000, 500)
        p_survive_1500 = 1 - calculate_probability_exponential(1000, 1500)
        f.write(f"    P(failure before 500h) = {p_fail_500:.4f}\n")
        f.write(f"    P(survive beyond 1500h) = {p_survive_1500:.4f}\n\n")

        f.write("3.5 Bayes' Theorem (Damage Detection):\n")
        f.write("    Scenario: Base rate = 5%, Sensitivity = 95%, Specificity = 90%\n")
        bayes_result = apply_bayes_theorem(0.05, 0.95, 0.90)
        f.write(f"    Prior: P(Damage) = {bayes_result['prior']:.4f}\n")
        f.write(f"    Posterior: P(Damage | Test+) = {bayes_result['posterior']:.4f}\n")
        f.write(f"    This means if test is positive, there is {bayes_result['posterior']*100:.1f}% chance of actual damage\n\n")

        # Engineering Implications
        f.write("ENGINEERING IMPLICATIONS\n")
        f.write("-"*80 + "\n\n")
        f.write("1. Quality Control:\n")
        f.write("   - Descriptive statistics help identify variability in concrete strength\n")
        f.write("   - High CV suggests need for better quality control procedures\n\n")

        f.write("2. Design Specifications:\n")
        f.write("   - Material property distributions inform safety factors\n")
        f.write("   - Percentile values used for characteristic strength calculations\n\n")

        f.write("3. Risk Assessment:\n")
        f.write("   - Probability distributions model uncertainty in loads and strengths\n")
        f.write("   - Bayes' theorem helps update risk estimates with new information\n\n")

        f.write("4. Decision Making:\n")
        f.write("   - Statistical analysis provides quantitative basis for engineering decisions\n")
        f.write("   - Probability calculations inform inspection and maintenance strategies\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")

    print(f"✓ Statistical report saved: {output_file}")


def main():
    """Main execution function."""
    print("="*80)
    print("LAB 4: STATISTICAL ANALYSIS")
    print("Descriptive Statistics and Probability Distributions")
    print("="*80)
    print()

    # ========== PART 1: DESCRIPTIVE STATISTICS ==========
    print("\n" + "="*80)
    print("PART 1: DESCRIPTIVE STATISTICS")
    print("="*80)

    # Load concrete strength data
    print("\n1.1 Loading Concrete Strength Data:")
    concrete_df = load_data('concrete_strength.csv')
    if concrete_df is None:
        return

    print("\nFirst few rows:")
    print(concrete_df.head())
    print("\nData info:")
    print(concrete_df.info())

    # Calculate descriptive statistics
    print("\n1.2 Descriptive Statistics:")
    stats_dict = calculate_descriptive_stats(concrete_df, 'strength_mpa')

    print("\nMeasures of Central Tendency:")
    print(f"  Mean:   {stats_dict['mean']:.4f} MPa")
    print(f"  Median: {stats_dict['median']:.4f} MPa")
    print(f"  Mode:   {stats_dict['mode']:.4f} MPa")

    print("\nMeasures of Spread:")
    print(f"  Variance:  {stats_dict['variance']:.4f} MPa²")
    print(f"  Std Dev:   {stats_dict['std_dev']:.4f} MPa")
    print(f"  Range:     {stats_dict['range']:.4f} MPa")
    print(f"  IQR:       {stats_dict['iqr']:.4f} MPa")

    print("\nShape Measures:")
    print(f"  Skewness:  {stats_dict['skewness']:.4f}")
    print(f"  Kurtosis:  {stats_dict['kurtosis']:.4f}")

    print("\nFive-Number Summary:")
    print(f"  Min: {stats_dict['min']:.4f} MPa")
    print(f"  Q1:  {stats_dict['q1']:.4f} MPa")
    print(f"  Q2:  {stats_dict['median']:.4f} MPa")
    print(f"  Q3:  {stats_dict['q3']:.4f} MPa")
    print(f"  Max: {stats_dict['max']:.4f} MPa")

    # Visualize concrete strength distribution
    print("\n1.3 Visualizing Concrete Strength Distribution:")
    plot_distribution(concrete_df, 'strength_mpa',
                     'Concrete Strength Distribution Analysis',
                     save_path='concrete_strength_distribution.png')

    # ========== PART 2: PROBABILITY DISTRIBUTIONS ==========
    print("\n" + "="*80)
    print("PART 2: PROBABILITY DISTRIBUTIONS")
    print("="*80)

    print("\n2.1 Discrete Distributions:")

    # Binomial
    print("\n  Binomial Distribution - Quality Control Scenario:")
    print("    100 components tested, 5% defect rate")
    p_exactly_3 = calculate_probability_binomial(100, 0.05, 3)
    p_le_5 = sum([calculate_probability_binomial(100, 0.05, k) for k in range(6)])
    print(f"    P(exactly 3 defects) = {p_exactly_3:.6f}")
    print(f"    P(≤ 5 defects) = {p_le_5:.6f}")

    # Poisson
    print("\n  Poisson Distribution - Bridge Load Events:")
    print("    Average 10 heavy trucks per hour")
    p_exactly_8 = calculate_probability_poisson(10, 8)
    p_gt_15 = 1 - sum([calculate_probability_poisson(10, k) for k in range(16)])
    print(f"    P(exactly 8 trucks) = {p_exactly_8:.6f}")
    print(f"    P(> 15 trucks) = {p_gt_15:.6f}")

    print("\n2.2 Continuous Distributions:")

    # Normal
    print("\n  Normal Distribution - Steel Yield Strength:")
    print("    Mean = 250 MPa, Std = 15 MPa")
    p_exceeds_280 = calculate_probability_normal(250, 15, x_lower=280)
    percentile_95 = norm.ppf(0.95, 250, 15)
    print(f"    P(X > 280 MPa) = {p_exceeds_280:.6f} ({p_exceeds_280*100:.2f}%)")
    print(f"    95th percentile = {percentile_95:.4f} MPa")

    # Exponential
    print("\n  Exponential Distribution - Component Lifetime:")
    print("    Mean lifetime = 1000 hours")
    p_fail_500 = calculate_probability_exponential(1000, 500)
    p_survive_1500 = 1 - calculate_probability_exponential(1000, 1500)
    print(f"    P(failure before 500h) = {p_fail_500:.6f}")
    print(f"    P(survive beyond 1500h) = {p_survive_1500:.6f}")

    # Plot all distributions
    print("\n2.3 Visualizing Probability Distributions:")
    plot_probability_distributions(save_path='probability_distributions.png')

    # Distribution Fitting
    print("\n2.4 Distribution Fitting:")
    fitted_params = fit_distribution(concrete_df, 'strength_mpa', 'normal')
    plot_distribution_fitting(concrete_df, 'strength_mpa', fitted_params,
                             save_path='distribution_fitting.png')

    # ========== PART 3: PROBABILITY APPLICATIONS ==========
    print("\n" + "="*80)
    print("PART 3: PROBABILITY APPLICATIONS")
    print("="*80)

    print("\n3.1 Bayes' Theorem - Structural Damage Detection:")
    print("    Base rate: 5% of structures have damage")
    print("    Test sensitivity: 95% (true positive rate)")
    print("    Test specificity: 90% (true negative rate)")

    bayes_result = apply_bayes_theorem(0.05, 0.95, 0.90)

    print(f"\n    Prior: P(Damage) = {bayes_result['prior']:.4f}")
    print(f"    Posterior: P(Damage | Test+) = {bayes_result['posterior']:.4f}")
    print(f"\n    Interpretation:")
    print(f"    If the test is positive, there is a {bayes_result['posterior']*100:.1f}% probability")
    print(f"    that the structure actually has damage.")
    print(f"    This is a {(bayes_result['posterior']/bayes_result['prior']):.1f}x increase from the prior!")

    # Visualize Bayes' theorem
    print("\n3.2 Visualizing Bayes' Theorem:")
    plot_bayes_tree(0.05, 0.95, 0.90, save_path='bayes_theorem_tree.png')

    # ========== PART 4: MATERIAL COMPARISON ==========
    print("\n" + "="*80)
    print("PART 4: MATERIAL PROPERTIES COMPARISON")
    print("="*80)

    print("\n4.1 Loading Material Properties Data:")
    material_df = load_data('material_properties.csv')
    if material_df is None:
        return

    print("\n4.2 Comparative Statistics:")
    grouped = material_df.groupby('material_type')['yield_strength_mpa'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(grouped)

    print("\n4.3 Visualizing Material Comparison:")
    plot_material_comparison(material_df, 'yield_strength_mpa', 'material_type',
                            save_path='material_comparison_boxplot.png')

    # ========== PART 5: STATISTICAL REPORT ==========
    print("\n" + "="*80)
    print("PART 5: GENERATING STATISTICAL REPORT")
    print("="*80)

    create_statistical_report(concrete_df, material_df, 'lab4_statistical_report.txt')

    print("\n" + "="*80)
    print("LAB 4 COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Outputs:")
    print("  1. concrete_strength_distribution.png")
    print("  2. probability_distributions.png")
    print("  3. distribution_fitting.png")
    print("  4. bayes_theorem_tree.png")
    print("  5. material_comparison_boxplot.png")
    print("  6. lab4_statistical_report.txt")
    print("\nAll analyses completed. Review the plots and report for detailed findings.")


if __name__ == "__main__":
    main()
