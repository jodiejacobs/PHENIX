import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Single cell analysis with linear regression for infected vs uninfected cells')
parser.add_argument('-i', '--input', required=True, help='Input file path (e.g., object.head.txt)')
parser.add_argument('-o', '--output', required=True, help='Output directory for plots and results')
parser.add_argument('-n', '--num_features', type=int, default=20, 
                    help='Number of features to analyze (default: 20, use -1 for all)')
args = parser.parse_args()

# Create output directory if it doesn't exist
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Read the data
df = pd.read_csv(args.input, sep='\t', skiprows=8)

# Clean up cell type column
df['Cell Type'] = df['Cell Type'].str.strip()

# Filter to only include rows with cell type data
df_filtered = df[df['Cell Type'].isin(['JW18 uninf.', 'JW18 wMel'])].copy()

# Encode cell types for regression (0 = uninfected, 1 = wMel)
df_filtered['Cell_Type_Numeric'] = (df_filtered['Cell Type'] == 'JW18 wMel').astype(int)

# Define columns to analyze (excluding metadata)
exclude_cols = ['Row', 'Column', 'Plane', 'Timepoint', 'Field', 'Object No', 'X', 'Y', 
                'Bounding Box', 'Position X [µm]', 'Position Y [µm]', 'Compound', 
                'Concentration', 'Cell Type', 'Cell Count', 'Cell_Type_Numeric',
                'Cells Selected Selected - Object No in Cells',
                'Cells Selected Selected - Object No in Cells Selected',
                'Cells Selected Selected - Class']

# Get numeric measurement columns
numeric_cols = [col for col in df_filtered.columns if col not in exclude_cols and 
                pd.api.types.is_numeric_dtype(df_filtered[col]) and 
                col.startswith('Cells Selected Selected')]

print(f"Analyzing {len(numeric_cols)} cellular measurements...")
print(f"Total cells: {len(df_filtered)}")
print(f"  - Uninfected: {(df_filtered['Cell Type'] == 'JW18 uninf.').sum()}")
print(f"  - wMel infected: {(df_filtered['Cell Type'] == 'JW18 wMel').sum()}")
print("\n" + "="*80 + "\n")

# Function to perform regression and create plot
def analyze_measurement(data, measurement_col, output_prefix='cell'):
    """
    Analyze a single measurement with scatter plot and linear regression
    """
    # Prepare data
    df_clean = data[[measurement_col, 'Cell Type', 'Cell_Type_Numeric']].dropna()
    
    if len(df_clean) < 10:
        return None
    
    # Separate by cell type
    uninf = df_clean[df_clean['Cell Type'] == 'JW18 uninf.'][measurement_col]
    wmel = df_clean[df_clean['Cell Type'] == 'JW18 wMel'][measurement_col]
    
    # Perform t-test
    t_stat, p_value_ttest = stats.ttest_ind(uninf, wmel)
    
    # Linear regression with cell type as predictor
    X = df_clean['Cell_Type_Numeric'].values.reshape(-1, 1)
    y = df_clean[measurement_col].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate R-squared and p-value for regression
    y_pred = model.predict(X)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # F-statistic and p-value for regression
    n = len(y)
    f_stat = (r_squared / 1) / ((1 - r_squared) / (n - 2))
    p_value_reg = 1 - stats.f.cdf(f_stat, 1, n - 2)
    
    # Create figure with scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot individual cells
    for cell_type, color, label in [('JW18 uninf.', 'lightgreen', 'Uninfected'),
                                     ('JW18 wMel', 'lightblue', 'wMel')]:
        mask = df_clean['Cell Type'] == cell_type
        cell_data = df_clean[mask]
        ax.scatter(np.random.normal(int(cell_type == 'JW18 wMel'), 0.1, len(cell_data)),
                   cell_data[measurement_col],
                   alpha=0.5, s=30, color=color, label=label, edgecolors='black', linewidth=0.5)
    
    # Add mean lines
    ax.plot([-0.3, 0.3], [uninf.mean(), uninf.mean()], 'k-', linewidth=2, label='Mean')
    ax.plot([0.7, 1.3], [wmel.mean(), wmel.mean()], 'k-', linewidth=2)
    
    # Customize plot
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Uninfected', 'wMel'])
    ax.set_ylabel(measurement_col.replace('Cells Selected Selected - ', ''), 
                  fontsize=10, fontweight='bold')
    ax.set_title(f'{measurement_col.replace("Cells Selected Selected - ", "")}\n'
                 f'Regression: β={model.coef_[0]:.4f}, p={p_value_reg:.2e} | '
                 f't-test: p={p_value_ttest:.2e}',
                 fontsize=10)
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, 1.5)
    
    plt.tight_layout()
    
    # Save plot
    safe_filename = measurement_col.replace('/', '_').replace(' ', '_').replace('[', '').replace(']', '')
    output_path = output_dir / f'{output_prefix}_{safe_filename}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'measurement': measurement_col,
        'coefficient': model.coef_[0],
        'intercept': model.intercept_,
        'r_squared': r_squared,
        'p_value_regression': p_value_reg,
        'p_value_ttest': p_value_ttest,
        'mean_uninf': uninf.mean(),
        'std_uninf': uninf.std(),
        'n_uninf': len(uninf),
        'mean_wmel': wmel.mean(),
        'std_wmel': wmel.std(),
        'n_wmel': len(wmel),
        'effect_size': model.coef_[0] / np.std(y)  # Standardized coefficient
    }

# Analyze all measurements
results = []

# Determine how many features to analyze
if args.num_features == -1:
    features_to_analyze = numeric_cols
else:
    features_to_analyze = numeric_cols[:args.num_features]

for i, col in enumerate(features_to_analyze):
    print(f"Processing {i+1}/{len(features_to_analyze)}: {col}")
    result = analyze_measurement(df_filtered, col, output_prefix='single_cell')
    if result:
        results.append(result)

# Create summary dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('p_value_regression')

# Save results
results_path = output_dir / 'single_cell_regression_results.csv'
results_df.to_csv(results_path, index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80 + "\n")

# Display top significant results
print("Top 10 most significant differences (by regression p-value):")
print("-" * 80)
for idx, row in results_df.head(10).iterrows():
    print(f"\n{row['measurement'].replace('Cells Selected Selected - ', '')}")
    print(f"  Coefficient (β): {row['coefficient']:.4f}")
    print(f"  R²: {row['r_squared']:.4f}")
    print(f"  Regression p-value: {row['p_value_regression']:.2e}")
    print(f"  t-test p-value: {row['p_value_ttest']:.2e}")
    print(f"  Uninfected: {row['mean_uninf']:.3f} ± {row['std_uninf']:.3f} (n={row['n_uninf']})")
    print(f"  wMel: {row['mean_wmel']:.3f} ± {row['std_wmel']:.3f} (n={row['n_wmel']})")
    print(f"  Effect size: {row['effect_size']:.4f}")

print("\n" + "="*80)
print(f"\nResults saved to: {results_path}")
print(f"Plots saved in: {output_dir}")
