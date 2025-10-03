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

# Read the data - note: object files have different structure than plate files
# First, let's find where [Data] is located
with open(args.input, 'r') as f:
    for i, line in enumerate(f):
        if line.strip() == '[Data]':
            skip_rows = i + 1  # Skip [Data] line and use next line as header
            break
    else:
        skip_rows = 8  # Default fallback

print(f"Skipping {skip_rows} rows to read data...")

# Use low_memory=False to avoid dtype warnings with large files
df = pd.read_csv(args.input, sep='\t', skiprows=skip_rows, low_memory=False)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

print(f"Read {len(df)} rows with {len(df.columns)} columns")
print(f"First few column names: {df.columns.tolist()[:5]}")

# Clean up cell type column
if 'Cell Type' in df.columns:
    df['Cell Type'] = df['Cell Type'].str.strip()
    print(f"Cell types found: {df['Cell Type'].unique()}")
else:
    print("ERROR: 'Cell Type' column not found!")
    print("Available columns:", df.columns.tolist())
    exit(1)

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # LEFT PANEL: Traditional comparison with distributions
    for cell_type, color, label in [('JW18 uninf.', 'lightgreen', 'Uninfected'),
                                     ('JW18 wMel', 'lightblue', 'wMel')]:
        mask = df_clean['Cell Type'] == cell_type
        cell_data = df_clean[mask]
        # Violin plot
        parts = ax1.violinplot([cell_data[measurement_col].values], 
                               positions=[int(cell_type == 'JW18 wMel')],
                               widths=0.5, showmeans=False, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
        # Scatter individual points
        ax1.scatter(np.random.normal(int(cell_type == 'JW18 wMel'), 0.08, len(cell_data)),
                   cell_data[measurement_col],
                   alpha=0.3, s=10, color=color, edgecolors='none')
    
    # Add mean lines
    ax1.plot([-0.3, 0.3], [uninf.mean(), uninf.mean()], 'k-', linewidth=2)
    ax1.plot([0.7, 1.3], [wmel.mean(), wmel.mean()], 'k-', linewidth=2)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Uninfected', 'wMel'])
    ax1.set_ylabel(measurement_col.replace('Cells Selected Selected - ', ''), 
                   fontsize=10, fontweight='bold')
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_title('Distribution Comparison', fontsize=11, fontweight='bold')
    
    # RIGHT PANEL: Regression plot showing actual linear relationship
    # Plot all points
    colors_scatter = df_clean['Cell_Type_Numeric'].map({0: 'lightgreen', 1: 'lightblue'})
    ax2.scatter(df_clean['Cell_Type_Numeric'], 
               df_clean[measurement_col],
               alpha=0.4, s=20, c=colors_scatter, edgecolors='black', linewidth=0.3)
    
    # Plot regression line
    x_line = np.array([0, 1])
    y_line = model.predict(x_line.reshape(-1, 1))
    ax2.plot(x_line, y_line, 'r-', linewidth=3, label=f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x')
    
    # Add confidence interval
    residual_std = np.std(residuals)
    y_err = 1.96 * residual_std  # 95% CI
    ax2.fill_between(x_line, y_line - y_err, y_line + y_err, alpha=0.2, color='red')
    
    # Add group means as larger points
    ax2.scatter([0, 1], [uninf.mean(), wmel.mean()], 
               s=200, c=['darkgreen', 'darkblue'], 
               marker='D', edgecolors='black', linewidth=2, 
               zorder=10, label='Group means')
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Uninfected (0)', 'wMel (1)'])
    ax2.set_xlabel('Cell Type (Coded)', fontsize=10, fontweight='bold')
    ax2.set_ylabel(measurement_col.replace('Cells Selected Selected - ', ''), 
                   fontsize=10, fontweight='bold')
    ax2.set_xlim(-0.1, 1.1)
    ax2.legend(loc='best', fontsize=8)
    ax2.set_title(f'Linear Regression\nR² = {r_squared:.4f}, p = {p_value_reg:.2e}', 
                 fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{measurement_col.replace("Cells Selected Selected - ", "")}\n'
                 f'Coefficient: β = {model.coef_[0]:.4f} | t-test p = {p_value_ttest:.2e}',
                 fontsize=12, fontweight='bold')
    
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

# CREATE SUMMARY VISUALIZATION
print("\nCreating summary plots...")

# Calculate percent change and fold change
results_df['percent_change'] = ((results_df['mean_wmel'] - results_df['mean_uninf']) / 
                                 results_df['mean_uninf'].abs()) * 100
results_df['fold_change'] = results_df['mean_wmel'] / results_df['mean_uninf']
results_df['log2_fold_change'] = np.log2(results_df['fold_change'])

# Create a comprehensive summary figure
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Volcano plot (effect size vs significance)
ax1 = fig.add_subplot(gs[0, 0])
significant = results_df['p_value_regression'] < 0.05
ax1.scatter(results_df.loc[~significant, 'coefficient'], 
           -np.log10(results_df.loc[~significant, 'p_value_regression']),
           alpha=0.5, s=40, color='gray', label='Not significant')
ax1.scatter(results_df.loc[significant, 'coefficient'], 
           -np.log10(results_df.loc[significant, 'p_value_regression']),
           alpha=0.7, s=60, color='red', label='p < 0.05')
ax1.axhline(-np.log10(0.05), linestyle='--', color='black', linewidth=1)
ax1.axvline(0, linestyle='-', color='black', linewidth=0.5)
ax1.set_xlabel('Effect Size (β coefficient)', fontsize=11, fontweight='bold')
ax1.set_ylabel('-log10(p-value)', fontsize=11, fontweight='bold')
ax1.set_title('Volcano Plot: Magnitude vs Significance', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Top changes by absolute coefficient
ax2 = fig.add_subplot(gs[0, 1])
top_changes = results_df.nlargest(15, 'coefficient')
colors_bar = ['lightblue' if x > 0 else 'lightcoral' for x in top_changes['coefficient']]
y_pos = np.arange(len(top_changes))
ax2.barh(y_pos, top_changes['coefficient'], color=colors_bar, edgecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels([x.replace('Cells Selected Selected - ', '')[:40] 
                      for x in top_changes['measurement']], fontsize=8)
ax2.set_xlabel('Coefficient (wMel effect)', fontsize=11, fontweight='bold')
ax2.set_title('Top 15 Measurements Most Increased by Infection', fontsize=12, fontweight='bold')
ax2.axvline(0, color='black', linewidth=1)
ax2.grid(True, alpha=0.3, axis='x')

# 3. Percent change
ax3 = fig.add_subplot(gs[1, 0])
top_pct = results_df.nlargest(15, 'percent_change')
y_pos = np.arange(len(top_pct))
colors_bar = ['lightblue' if x > 0 else 'lightcoral' for x in top_pct['percent_change']]
ax3.barh(y_pos, top_pct['percent_change'], color=colors_bar, edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels([x.replace('Cells Selected Selected - ', '')[:40] 
                      for x in top_pct['measurement']], fontsize=8)
ax3.set_xlabel('Percent Change (%)', fontsize=11, fontweight='bold')
ax3.set_title('Top 15 by Percent Change (wMel vs Uninf)', fontsize=12, fontweight='bold')
ax3.axvline(0, color='black', linewidth=1)
ax3.grid(True, alpha=0.3, axis='x')

# 4. R-squared distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(results_df['r_squared'], bins=30, edgecolor='black', color='skyblue', alpha=0.7)
ax4.axvline(results_df['r_squared'].median(), color='red', linestyle='--', 
           linewidth=2, label=f'Median = {results_df["r_squared"].median():.3f}')
ax4.set_xlabel('R² (Variance Explained)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
ax4.set_title('Distribution of Model Fit (R²)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Effect size vs percent change
ax5 = fig.add_subplot(gs[2, 0])
scatter = ax5.scatter(results_df['percent_change'], results_df['coefficient'],
                     c=-np.log10(results_df['p_value_regression']), 
                     cmap='RdYlBu_r', s=60, alpha=0.6, edgecolors='black', linewidth=0.5)
ax5.set_xlabel('Percent Change (%)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Coefficient (β)', fontsize=11, fontweight='bold')
ax5.set_title('Effect Size vs Percent Change', fontsize=12, fontweight='bold')
ax5.axhline(0, color='black', linewidth=0.5)
ax5.axvline(0, color='black', linewidth=0.5)
plt.colorbar(scatter, ax=ax5, label='-log10(p-value)')
ax5.grid(True, alpha=0.3)

# 6. Summary statistics table
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')
summary_stats = [
    ['Total Measurements', f'{len(results_df)}'],
    ['Significant (p<0.05)', f'{(results_df["p_value_regression"] < 0.05).sum()}'],
    ['Highly Sig (p<0.001)', f'{(results_df["p_value_regression"] < 0.001).sum()}'],
    ['', ''],
    ['Increased by wMel', f'{(results_df["coefficient"] > 0).sum()}'],
    ['Decreased by wMel', f'{(results_df["coefficient"] < 0).sum()}'],
    ['', ''],
    ['Median R²', f'{results_df["r_squared"].median():.4f}'],
    ['Median |coefficient|', f'{results_df["coefficient"].abs().median():.4f}'],
    ['Median % change', f'{results_df["percent_change"].median():.2f}%'],
]
table = ax6.table(cellText=summary_stats, cellLoc='left',
                 colWidths=[0.6, 0.4], loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)
for i in range(len(summary_stats)):
    if i in [0, 4]:
        for j in range(2):
            table[(i, j)].set_facecolor('#CCE5FF')
            table[(i, j)].set_text_props(weight='bold')
ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

fig.suptitle('Infection Effect Summary: How wMel Changes Cellular Measurements', 
            fontsize=14, fontweight='bold', y=0.995)

# Save summary plot
summary_path = output_dir / 'infection_effect_summary.png'
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
print(f"Summary plot saved to: {summary_path}")
plt.close()

# Save enhanced results with percent change
enhanced_results_path = output_dir / 'single_cell_regression_results_enhanced.csv'
results_df.to_csv(enhanced_results_path, index=False)
print(f"Enhanced results saved to: {enhanced_results_path}")
