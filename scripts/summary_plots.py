import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import argparse
import os
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Compare cell measurements between infected and uninfected cells')
parser.add_argument('-i', '--input', required=True, help='Input file path (e.g., PlateResults.txt)')
parser.add_argument('-o', '--output', required=True, help='Output directory for plots and results')
args = parser.parse_args()

# Create output directory if it doesn't exist
output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

# Set style - no grid
sns.set_style("white")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 6
plt.rcParams['axes.labelsize'] = 6
plt.rcParams['axes.titlesize'] = 6
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['legend.fontsize'] = 6

# Define color mapping
COLOR_MAP = {
    'JW18DOX': '#8fcb84',
    'JW18 uninf.': '#8fcb84',
    'JW18 uninf': '#8fcb84',
    'JW18wMel': '#09aa4b',
    'JW18 wMel': '#09aa4b',
    'S2DOX': '#fab280',
    'S2 uninf.': '#fab280',
    'S2 uninf': '#fab280',
    'S2wMel': '#d25727',
    'S2 wMel': '#d25727'
}

# Read the data - skip metadata lines and [Data] line
df = pd.read_csv(args.input, sep='\t', skiprows=8)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check if Cell Type column exists
if 'Cell Type' not in df.columns:
    print("Error: 'Cell Type' column not found!")
    print("Available columns:", df.columns.tolist())
    exit(1)

# Clean up cell type column
df['Cell Type'] = df['Cell Type'].str.strip()

# Identify numeric columns to analyze (excluding metadata columns)
exclude_cols = ['Row', 'Column', 'Plane', 'Timepoint', 'Number of Analyzed Fields', 
                'Height [µm]', 'Time [s]', 'Compound', 'Concentration', 'Cell Type', 
                'Cell Count', 'Cells - Number of Objects', 'wMel - Number of Objects',
                'uninfected - Number of Objects', 'Cells Selected - Number of Objects']

numeric_cols = [col for col in df.columns if col not in exclude_cols and 
                pd.api.types.is_numeric_dtype(df[col])]

# Filter to only include rows with cell type data (handles both JW18 and S2)
valid_cell_types = [ct for ct in df['Cell Type'].unique() if ct in COLOR_MAP.keys()]
df_filtered = df[df['Cell Type'].isin(valid_cell_types)]

# Function to perform t-test and create plot for each column
def analyze_and_plot(data, column_name, output_prefix='plot'):
    """
    Analyze a single measurement column and create a bar plot with significance
    """
    # Determine which cell line we're working with
    cell_types = data['Cell Type'].unique()
    
    # Try to identify infected vs uninfected pairs
    uninf_type = None
    wmel_type = None
    
    for ct in cell_types:
        if 'uninf' in ct.lower() or 'dox' in ct.lower():
            uninf_type = ct
        elif 'wmel' in ct.lower():
            wmel_type = ct
    
    if uninf_type is None or wmel_type is None:
        return None
    
    # Group by cell type
    uninf_data = data[data['Cell Type'] == uninf_type][column_name].dropna()
    wmel_data = data[data['Cell Type'] == wmel_type][column_name].dropna()
    
    # Skip if insufficient data
    if len(uninf_data) < 2 or len(wmel_data) < 2:
        return None
    
    # Calculate statistics
    uninf_mean = uninf_data.mean()
    uninf_std = uninf_data.std()
    wmel_mean = wmel_data.mean()
    wmel_std = wmel_data.std()
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(uninf_data, wmel_data)
    
    # Create plot with figure size adjusted so plot area is 2x2 inches
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    
    x_pos = [0, 1]
    means = [uninf_mean, wmel_mean]
    stds = [uninf_std, wmel_std]
    
    # Keep full labels with infection status
    labels = [uninf_type, wmel_type]
    
    # Get colors from mapping
    colors = [COLOR_MAP.get(uninf_type, '#8fcb84'), 
              COLOR_MAP.get(wmel_type, '#09aa4b')]
    
    # Create bar plot with error bars
    bars = ax.bar(x_pos, means, yerr=stds, capsize=3, 
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
    
    # Customize plot - 6pt fonts, show p-value
    ax.set_ylabel('Value', fontsize=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=6)
    ax.tick_params(axis='both', labelsize=6, width=0.5)
    
    # # Add p-value to plot
    # ax.text(0.02, 0.98, f'p={p_value:.3e}', 
    #         transform=ax.transAxes, fontsize=6, 
    #         ha='left', va='top')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add significance indicator
    y_max = max(means[0] + stds[0], means[1] + stds[1])
    y_pos = y_max * 1.1
    
    if p_value < 0.001:
        sig_text = '***'
    elif p_value < 0.01:
        sig_text = '**'
    elif p_value < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    
    # Draw significance line
    ax.plot([x_pos[0], x_pos[0], x_pos[1], x_pos[1]], 
            [y_pos, y_pos*1.02, y_pos*1.02, y_pos], 
            'k-', linewidth=0.5)
    ax.text((x_pos[0] + x_pos[1])/2, y_pos*1.03, sig_text, 
            ha='center', fontsize=6)
    
    # Add p-value text
    ax.text(0.98, 0.98, f'p={p_value:.2e}', 
            transform=ax.transAxes, fontsize=6, 
            ha='right', va='top')
    
    # # Add sample sizes
    # ax.text(x_pos[0], -y_max*0.15, f'n={len(uninf_data)}', 
    #         ha='center', fontsize=6)
    # ax.text(x_pos[1], -y_max*0.15, f'n={len(wmel_data)}', 
    #         ha='center', fontsize=6)
    
    plt.tight_layout()
    
    # Save plot
    safe_filename = column_name.replace('/', '_').replace(' ', '_').replace('[', '').replace(']', '')
    output_path = output_dir / f'{output_prefix}_{safe_filename}.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    
    return {
        'column': column_name,
        'uninf_mean': uninf_mean,
        'uninf_std': uninf_std,
        'uninf_n': len(uninf_data),
        'wmel_mean': wmel_mean,
        'wmel_std': wmel_std,
        'wmel_n': len(wmel_data),
        't_statistic': t_stat,
        'p_value': p_value
    }

# Analyze all numeric columns
results = []
print("Analyzing columns and generating plots...")
for i, col in enumerate(numeric_cols):
    print(f"Processing {i+1}/{len(numeric_cols)}: {col}")
    result = analyze_and_plot(df_filtered, col, output_prefix='comparison')
    if result:
        results.append(result)

# Create summary dataframe
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('p_value')

# Save results to CSV
results_path = output_dir / 'statistical_results.csv'
results_df.to_csv(results_path, index=False)
print(f"\nAnalysis complete!")
print(f"Generated {len(results)} plots in: {output_dir}")
print(f"Results saved to: {results_path}")

# Display top significant results
print("\nTop 10 most significant differences (p < 0.05):")
sig_results = results_df[results_df['p_value'] < 0.05].head(10)
for idx, row in sig_results.iterrows():
    print(f"{row['column']}: p={row['p_value']:.2e}")