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

# Function to perform t-test and create boxplot for each column
def analyze_and_plot(data, column_name, output_prefix='plot'):
    """
    Analyze a single measurement column and create a box plot with colored dots
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
    plot_data = []
    plot_order = []
    results = {}
    
    # Add JW18 data if available
    if has_jw18:
        jw18_uninf_data = data[data['Cell Type'] == jw18_uninf][column_name].dropna()
        jw18_wmel_data = data[data['Cell Type'] == jw18_wmel][column_name].dropna()
        
        if len(jw18_uninf_data) >= 2 and len(jw18_wmel_data) >= 2:
            # Add to plot data
            for val in jw18_uninf_data:
                plot_data.append({'Cell Type': jw18_uninf, 'Value': val})
            for val in jw18_wmel_data:
                plot_data.append({'Cell Type': jw18_wmel, 'Value': val})
            
            plot_order.extend([jw18_uninf, jw18_wmel])
            
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
                'groups': [jw18_uninf, jw18_wmel]
            }
    
    # Add S2 data if available
    if has_s2:
        s2_uninf_data = data[data['Cell Type'] == s2_uninf][column_name].dropna()
        s2_wmel_data = data[data['Cell Type'] == s2_wmel][column_name].dropna()
        
        if len(s2_uninf_data) >= 2 and len(s2_wmel_data) >= 2:
            # Add to plot data
            for val in s2_uninf_data:
                plot_data.append({'Cell Type': s2_uninf, 'Value': val})
            for val in s2_wmel_data:
                plot_data.append({'Cell Type': s2_wmel, 'Value': val})
            
            plot_order.extend([s2_uninf, s2_wmel])
            
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
                'groups': [s2_uninf, s2_wmel]
            }
    
    if len(plot_data) == 0:
        return None
    
    # Convert to DataFrame for plotting
    plot_df = pd.DataFrame(plot_data)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    
    # Create boxplot
    bp = ax.boxplot([plot_df[plot_df['Cell Type'] == ct]['Value'].values for ct in plot_order],
                     positions=range(len(plot_order)),
                     widths=0.6,
                     patch_artist=True,
                     showfliers=False,  # Don't show outliers as we'll plot all points
                     medianprops=dict(color='black', linewidth=1),
                     boxprops=dict(linewidth=0.5),
                     whiskerprops=dict(linewidth=0.5),
                     capprops=dict(linewidth=0.5))
    
    # Color the boxes and outlines
    for patch, ct in zip(bp['boxes'], plot_order):
        color = COLOR_MAP.get(ct, '#gray')
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
    
    # Color the whiskers and caps to match
    for i, ct in enumerate(plot_order):
        color = COLOR_MAP.get(ct, '#gray')
        bp['whiskers'][i*2].set_color(color)
        bp['whiskers'][i*2+1].set_color(color)
        bp['caps'][i*2].set_color(color)
        bp['caps'][i*2+1].set_color(color)
    
    # Add colored dots for each data point
    for i, ct in enumerate(plot_order):
        values = plot_df[plot_df['Cell Type'] == ct]['Value'].values
        # Add jitter to x positions for visibility
        x_positions = np.random.normal(i, 0.04, size=len(values))
        ax.scatter(x_positions, values, 
                  color=COLOR_MAP.get(ct, '#gray'), 
                  alpha=0.6, 
                  s=8, 
                  edgecolors='none',
                  zorder=3)
    
    # Customize plot
    ax.set_ylabel('Value', fontsize=6)
    ax.set_xticks(range(len(plot_order)))
    ax.set_xticklabels(plot_order, fontsize=6, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=6, width=0.5)
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Add significance indicators for each comparison
    y_max = plot_df['Value'].max()
    y_range = plot_df['Value'].max() - plot_df['Value'].min()
    
    sig_height = y_max
    for comparison, data_dict in results.items():
        groups = data_dict['groups']
        p_val = data_dict['p_value']
        
        x1 = plot_order.index(groups[0])
        x2 = plot_order.index(groups[1])
        
        sig_height += y_range * 0.08
        
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        # Draw significance line
        ax.plot([x1, x1, x2, x2], 
                [sig_height, sig_height + y_range*0.02, sig_height + y_range*0.02, sig_height], 
                'k-', linewidth=0.5)
        ax.text((x1 + x2)/2, sig_height + y_range*0.03, sig_text, 
                ha='center', fontsize=6)
        
        # Add p-value text
        ax.text((x1 + x2)/2, sig_height + y_range*0.06, f'p={p_val:.2e}', 
                ha='center', fontsize=5)
        
        sig_height += y_range * 0.12
    
    # Adjust y-axis limits to accommodate significance bars
    ax.set_ylim(plot_df['Value'].min() * 0.95, sig_height + y_range * 0.05)
    
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
    result = analyze_and_plot(df_filtered, col, output_prefix='boxplot')
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
    
    # Specifically look for Lysotracker results
    print("\n" + "="*60)
    print("LYSOTRACKER INTENSITY ANALYSIS:")
    print("="*60)
    lyso_results = results_df[results_df['column'].str.contains('Lyso', case=False, na=False)]
    
    if len(lyso_results) > 0:
        for idx, row in lyso_results.iterrows():
            cell_line = row['cell_line'].upper()
            print(f"\n{cell_line} Comparison:")
            print(f"  Column: {row['column']}")
            print(f"  Uninfected mean ± std: {row['uninf_mean']:.2f} ± {row['uninf_std']:.2f} (n={row['uninf_n']})")
            print(f"  wMel mean ± std: {row['wmel_mean']:.2f} ± {row['wmel_std']:.2f} (n={row['wmel_n']})")
            print(f"  Fold change: {row['wmel_mean']/row['uninf_mean']:.2f}x")
            print(f"  t-statistic: {row['t_statistic']:.3f}")
            print(f"  p-value: {row['p_value']:.2e}")
            
            if row['p_value'] < 0.001:
                sig = "highly significant (***)"
            elif row['p_value'] < 0.01:
                sig = "very significant (**)"
            elif row['p_value'] < 0.05:
                sig = "significant (*)"
            else:
                sig = "not significant"
            
            if row['wmel_mean'] > row['uninf_mean']:
                direction = "INCREASED"
            else:
                direction = "DECREASED"
            
            print(f"  Conclusion: Lysotracker intensity {direction} in wMel-infected cells ({sig})")
    else:
        print("\nNo Lysotracker measurements found in the data.")
        print("Columns containing 'lyso' (case-insensitive):")
        lyso_cols = [col for col in numeric_cols if 'lyso' in col.lower()]
        if lyso_cols:
            for col in lyso_cols:
                print(f"  - {col}")
        else:
            print("  None found")
else:
    print("\nNo valid comparisons found!")
