import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Read the data
df = pd.read_csv('PlateResults.txt', sep='\t', skiprows=7)

# Clean up cell type column
df['Cell Type'] = df['Cell Type'].str.strip()

# Identify numeric columns to analyze (excluding metadata columns)
exclude_cols = ['Row', 'Column', 'Plane', 'Timepoint', 'Number of Analyzed Fields', 
                'Height [µm]', 'Time [s]', 'Compound', 'Concentration', 'Cell Type', 
                'Cell Count', 'Cells - Number of Objects', 'wMel - Number of Objects',
                'uninfected - Number of Objects', 'Cells Selected - Number of Objects']

numeric_cols = [col for col in df.columns if col not in exclude_cols and 
                pd.api.types.is_numeric_dtype(df[col])]

# Filter to only include rows with cell type data
df_filtered = df[df['Cell Type'].isin(['JW18 uninf.', 'JW18 wMel'])]

# Function to perform t-test and create plot for each column
def analyze_and_plot(data, column_name, output_prefix='plot'):
    """
    Analyze a single measurement column and create a bar plot with significance
    """
    # Group by cell type
    uninf_data = data[data['Cell Type'] == 'JW18 uninf.'][column_name].dropna()
    wmel_data = data[data['Cell Type'] == 'JW18 wMel'][column_name].dropna()
    
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
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_pos = [0, 1]
    means = [uninf_mean, wmel_mean]
    stds = [uninf_std, wmel_std]
    labels = ['JW18 uninf', 'JW18 wMel']
    colors = ['lightgreen', 'lightblue']
    
    # Create bar plot with error bars
    bars = ax.bar(x_pos, means, yerr=stds, capsize=10, 
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Customize plot
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'{column_name}\np-value: {p_value:.4e}', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    
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
            'k-', linewidth=1.5)
    ax.text((x_pos[0] + x_pos[1])/2, y_pos*1.03, sig_text, 
            ha='center', fontsize=14, fontweight='bold')
    
    # Add sample sizes
    ax.text(x_pos[0], -y_max*0.15, f'n={len(uninf_data)}', 
            ha='center', fontsize=9)
    ax.text(x_pos[1], -y_max*0.15, f'n={len(wmel_data)}', 
            ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    safe_filename = column_name.replace('/', '_').replace(' ', '_').replace('[', '').replace(']', '')
    plt.savefig(f'{output_prefix}_{safe_filename}.png', dpi=300, bbox_inches='tight')
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
results_df.to_csv('statistical_results.csv', index=False)
print(f"\nAnalysis complete!")
print(f"Generated {len(results)} plots")
print(f"Results saved to 'statistical_results.csv'")

# Display top significant results
print("\nTop 10 most significant differences (p < 0.05):")
sig_results = results_df[results_df['p_value'] < 0.05].head(10)
for idx, row in sig_results.iterrows():
    print(f"{row['column']}: p={row['p_value']:.2e}")
