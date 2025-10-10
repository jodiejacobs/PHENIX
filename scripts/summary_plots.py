import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import os
from pathlib import Path
from scipy import stats

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

df['Cell Type'] = df['Cell Type'].astype(str).str.strip()

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
    # Separate into JW18 and S2 groups
    jw18_uninf = None
    jw18_wmel = None
    s2_uninf = None
    s2_wmel = None
    
    for ct in data['Cell Type'].unique():
        if 'jw18' in ct.lower():
            if 'uninf' in ct.lower() or 'dox' in ct.lower():
                jw18_uninf = ct
            elif 'wmel' in ct.lower():
                jw18_wmel = ct
        elif 's2' in ct.lower():
            if 'uninf' in ct.lower() or 'dox' in ct.lower():
                s2_uninf = ct
            elif 'wmel' in ct.lower():
                s2_wmel = ct
    
    # Check if we have at least one complete pair
    has_jw18 = jw18_uninf is not None and jw18_wmel is not None
    has_s2 = s2_uninf is not None and s2_wmel is not None
    
    if not has_jw18 and not has_s2:
        return None
    
    # Prepare data for plotting
    x_pos = []
    means = []
    stds = []
    labels = []
    colors = []
    results = {}
    
    position = 0
    
    # Add JW18 data if available
    if has_jw18:
        jw18_uninf_data = data[data['Cell Type'] == jw18_uninf][column_name].dropna()
        jw18_wmel_data = data[data['Cell Type'] == jw18_wmel][column_name].dropna()
        
        if len(jw18_uninf_data) >= 2 and len(jw18_wmel_data) >= 2:
            x_pos.extend([position, position + 1])
            means.extend([jw18_uninf_data.mean(), jw18_wmel_data.mean()])
            stds.extend([jw18_uninf_data.std(), jw18_wmel_data.std()])
            labels.extend([jw18_uninf, jw18_wmel])
            colors.extend([COLOR_MAP.get(jw18_uninf, '#8fcb84'), 
                          COLOR_MAP.get(jw18_wmel, '#09aa4b')])
            
            # Perform t-test for JW18
            t_stat, p_value = stats.ttest_ind(jw18_uninf_data, jw18_wmel_data)
            results['jw18'] = {
                'uninf_mean': jw18_uninf_data.mean(),
                'uninf_std': jw18_uninf_data.std(),
                'uninf_n': len(jw18_uninf_data),
                'wmel_mean': jw18_wmel_data.mean(),
                'wmel_std': jw18_wmel_data.std(),
                'wmel_n': len(jw18_wmel_data),
                't_statistic': t_stat,
                'p_value': p_value,
                'x_pos': [position, position + 1]
            }
            position += 3  # Leave gap between groups
    
    # Add S2 data if available
    if has_s2:
        s2_uninf_data = data[data['Cell Type'] == s2_uninf][column_name].dropna()
        s2_wmel_data = data[data['Cell Type'] == s2_wmel][column_name].dropna()
        
        if len(s2_uninf_data) >= 2 and len(s2_wmel_data) >= 2:
            x_pos.extend([position, position + 1])
            means.extend([s2_uninf_data.mean(), s2_wmel_data.mean()])
            stds.extend([s2_uninf_data.std(), s2_wmel_data.std()])
            labels.extend([s2_uninf, s2_wmel])
            colors.extend([COLOR_MAP.get(s2_uninf, '#fab280'), 
                          COLOR_MAP.get(s2_wmel, '#d25727')])
            
            # Perform t-test for S2
            t_stat, p_value = stats.ttest_ind(s2_uninf_data, s2_wmel_data)
            results['s2'] = {
                'uninf_mean': s2_uninf_data.mean(),
                'uninf_std': s2_uninf_data.std(),
                'uninf_n': len(s2_uninf_data),
                'wmel_mean': s2_wmel_data.mean(),
                'wmel_std': s2_wmel_data.std(),
                'wmel_n': len(s2_wmel_data),
                't_statistic': t_stat,
                'p_value': p_value,
                'x_pos': [position, position + 1]
            }
    
    if len(means) == 0:
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    # Create bar plot
    bars = ax.bar(x_pos, means, yerr=stds, capsize=3, 
                   color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
    
    # Customize plot
    ax.set_ylabel('Value', fontsize=6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=6, width=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add significance indicators for each comparison
    y_max = max([m + s for m, s in zip(means, stds)])
    
    for comparison, data_dict in results.items():
        x_positions = data_dict['x_pos']
        p_val = data_dict['p_value']
        
        y_pos = y_max * 1.1
        
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        # Draw significance line
        ax.plot([x_positions[0], x_positions[0], x_positions[1], x_positions[1]], 
                [y_pos, y_pos*1.02, y_pos*1.02, y_pos], 
                'k-', linewidth=0.5)
        ax.text((x_positions[0] + x_positions[1])/2, y_pos*1.03, sig_text, 
                ha='center', fontsize=6)
        
        # Add p-value text
        ax.text((x_positions[0] + x_positions[1])/2, y_pos*1.08, f'p={p_val:.2e}', 
                ha='center', fontsize=5)
        
        y_max = y_pos * 1.15  # Adjust for next comparison
    
    plt.tight_layout()
    
    # Save plot
    safe_filename = column_name.replace('/', '_').replace(' ', '_').replace('[', '').replace(']', '')
    output_path = output_dir / f'{output_prefix}_{safe_filename}.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.close()
    
    # Return combined results
    return {
        'column': column_name,
        'results': results
    }
# Analyze all numeric columns
results = []
print("Analyzing columns and generating plots...")
for i, col in enumerate(numeric_cols):
    # print(f"Processing {i+1}/{len(numeric_cols): {col}")
    result = analyze_and_plot(df_filtered, col, output_prefix='comparison')
    if result:
        # Flatten results for CSV
        for cell_line, cell_stats in result['results'].items():
            results.append({
                'column': result['column'],
                'cell_line': cell_line,
                **cell_stats
            })

# Create summary dataframe
if len(results) > 0:
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')
    
    # Save results to CSV
    results_path = output_dir / 'statistical_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nAnalysis complete!")
    print(f"Generated plots in: {output_dir}")
    print(f"Results saved to: {results_path}")
    
    # Display top significant results
    print("\nTop 10 most significant differences (p < 0.05):")
    sig_results = results_df[results_df['p_value'] < 0.05].head(10)
    for idx, row in sig_results.iterrows():
        print(f"{row['column']} ({row['cell_line']}): p={row['p_value']:.2e}")
else:
    print("\nNo valid comparisons found!")