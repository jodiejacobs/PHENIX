# Single-Cell Analysis of Wolbachia (wMel) Infection Effects

## Overview

This analysis quantifies the cellular and morphological changes induced by *Wolbachia* wMel infection in *Drosophila melanogaster* cells using high-content imaging data. Linear regression models were used to assess the magnitude and significance of infection-associated changes across multiple cellular phenotypes.

## Data

- **Sample sizes:** 632,335 uninfected cells; 589,936 wMel-infected cells
- **Cell lines:** JW18 uninf. (uninfected control), JW18 wMel (wMel-infected)
- **Measurements:** 20 cellular features including morphology, texture, and nuclear staining

## Methods

### Statistical Analysis

For each cellular measurement, we performed:

1. **Linear regression** with infection status as the binary predictor (0 = uninfected, 1 = wMel)
2. **Independent t-tests** to confirm group differences
3. **Effect size calculation** using standardized coefficients

The regression coefficient (β) represents the absolute change in measurement value associated with wMel infection. All measurements showed p < 0.001 due to large sample sizes; effect sizes and R² values were used to assess biological significance.

## Key Findings

### 1. Cell-Cell Interaction

**Cell Contact Area with Neighbors: -16.46% (p < 0.001, R² = 0.101, effect size = -0.635)**

The most substantial and biologically significant effect of wMel infection is a reduction in cell-cell contact area. Infected cells show approximately 16.5 percentage points less contact with neighboring cells compared to uninfected controls (mean uninf: 33.67%, mean wMel: 17.21%). This represents a large effect size (-0.635) and is the most predictive single feature of infection status among all measurements tested.

**Biological interpretation:** This finding suggests that wMel infection alters cell adhesion properties or cytoskeletal organization, leading to reduced intercellular contact. This could result from changes in adhesion molecule expression, altered membrane properties, or mechanical changes to the cell cortex.

### 2. Cell Morphology

**Cell Width: +0.97 µm (p < 0.001, R² = 0.041, effect size = 0.404)**
- Infected cells are significantly wider (mean uninf: 8.85 µm, mean wMel: 9.82 µm)

**Cell Roundness: +0.030 (p < 0.001, R² = 0.036, effect size = 0.379)**
- Infected cells exhibit increased roundness on a 0-1 scale

**Width-to-Length Ratio: +0.065 (p < 0.001, R² = 0.055, effect size = 0.469)**
- Infected cells have higher aspect ratios, indicating more circular geometry

**Cell Length: -0.24 µm (p < 0.001, R² = 0.0007, effect size = -0.051)**
- Small but significant decrease in cell length

**Biological interpretation:** The coordinated changes in width, roundness, and aspect ratio indicate that wMel infection induces a shift from elongated to more spherical cell morphology. This morphological transition may reflect cytoskeletal reorganization, altered cell spreading, or changes in intracellular pressure. The effect is moderate (effect sizes 0.38-0.47) and consistent across the population.

### 3. Brightfield Texture Features

Multiple brightfield image texture features showed significant changes:

- **Ridge detection: +0.0017 (effect size = 0.509, R² = 0.065)**
- **Bright regions: +0.0023 (effect size = 0.502, R² = 0.063)**
- **Spot detection: +0.0010 (effect size = 0.417, R² = 0.043)**
- **Edge detection: +0.0061 (effect size = 0.399, R² = 0.040)**
- **Valley detection: +0.0013 (effect size = 0.274, R² = 0.019)**
- **Saddle detection: +0.0010 (effect size = 0.250, R² = 0.016)**
- **Dark regions: +0.0012 (effect size = 0.230, R² = 0.013)**

**Biological interpretation:** These texture features capture spatial intensity variations in brightfield microscopy. The coordinated increases in ridge, spot, and edge detection suggest that infected cells contain additional internal structures or density variations. These could represent:
1. Wolbachia bacteria themselves (though typically requiring specific staining)
2. Cellular reorganization in response to infection
3. Changes in organelle distribution or cytoplasmic density
4. Modified membrane ruffling or cell surface topology

The moderate effect sizes (0.23-0.51) indicate these are consistent, detectable changes across the infected population.

### 4. Cell Size and Nuclear Staining

**Cell Area: +9.27 µm² (p < 0.001, R² = 0.005, effect size = 0.147)**
- Small increase in total cell area (mean uninf: 113.0 µm², mean wMel: 122.3 µm²)
- Low R² indicates high variability; not a strong predictor

**HOECHST Mean Intensity: -4.10 (p < 0.001, R² = 0.0002, effect size = -0.025)**
- Minimal change in nuclear staining intensity
- Very small effect size indicates negligible biological significance

**HOECHST Sum: +7,903 (p < 0.001, R² = 0.001, effect size = 0.076)**
- Small increase in total nuclear signal
- Low predictive value

**Biological interpretation:** Changes in cell area and nuclear staining show statistical significance but minimal biological importance (low R² and small effect sizes). The slight increase in cell area is consistent with the morphological changes but explains little variance in the population.

## Summary of Effect Magnitudes

| Feature Category | Representative Measurement | Effect Size | Biological Significance |
|-----------------|---------------------------|-------------|------------------------|
| Cell-cell interaction | Contact area | -0.635 | Large |
| Cell shape | Width-to-length ratio | 0.469 | Medium |
| Cell width | Width | 0.404 | Medium |
| Internal texture | Ridge detection | 0.509 | Medium |
| Cell roundness | Roundness | 0.379 | Medium |
| Cell area | Total area | 0.147 | Small |
| Nuclear staining | HOECHST mean | -0.025 | Negligible |

## Interpretation

The primary cellular consequence of wMel infection is a dramatic reduction in cell-cell contact area (16.5 percentage point decrease), accompanied by morphological changes toward a rounder, less elongated cell shape. These findings suggest that Wolbachia infection fundamentally alters cell adhesion and/or mechanical properties.

Secondary effects include changes in brightfield texture features, which may reflect the physical presence of bacterial symbionts or host cell reorganization. Changes in cell size and nuclear staining are minimal and unlikely to be biologically significant despite statistical significance.

The high sample sizes (>500,000 cells per condition) provide exceptional statistical power, but biological interpretation should prioritize effect sizes and R² values over p-values alone.

## Technical Notes

### Regression Model

For each measurement Y:
```
Y = β₀ + β₁(infection status) + ε
```

Where:
- β₀ = intercept (predicted value for uninfected cells)
- β₁ = coefficient (change associated with wMel infection)
- infection status: 0 = uninfected, 1 = wMel

### Effect Size Calculation

Standardized effect size = β₁ / SD(Y), where SD(Y) is the standard deviation of the measurement across all cells.

### Model Fit

R² values indicate the proportion of variance in each measurement explained by infection status alone. Low R² values (even with significant p-values) indicate that factors other than infection status contribute substantially to measurement variability.

## Files Generated

- `single_cell_regression_results.csv` - Complete statistical results for all measurements
- `single_cell_regression_results_enhanced.csv` - Results with additional metrics (percent change, fold change)
- `infection_effect_summary.png` - Six-panel summary visualization
- Individual plots for each measurement (distribution comparison and regression)

## Citation

Analysis performed using Python 3.6 with pandas, scipy, scikit-learn, matplotlib, and seaborn.
