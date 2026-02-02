# Correlation Heatmap – A Visual Introduction to Asset Correlations

This project explores **asset return correlations** using correlation heatmaps, with a focus on **clarity, intuition, and visual understanding** rather than heavy mathematical abstraction.

I initially encountered correlation heatmaps without fully understanding what they represented or how to interpret them.  
The goal of this project was therefore simple: **build something visual enough that the concept of correlation becomes intuitive**, and then gradually layer in more quantitative structure.

---

##  Project Overview

The project:
- Downloads historical price data for major **technology stocks**
- Converts prices into **daily log returns**
- Computes **correlation matrices**
- Visualises correlations using **heatmaps**
- Enhances interpretability through:
  - clustering
  - masking redundant information
  - rolling time windows
  - summary statistics

The result is a beginner-friendly but professional-grade **quantitative exploratory analysis**.

---

##  What This Project Demonstrates

- Practical understanding of **correlation vs diversification**
- Why returns (not prices) are used in quantitative finance
- How correlations **change over time**
- How visually simple tools can reveal market structure
- Clean Python project structure and reproducibility

---

##  Features

### 1. Clustered Correlation Heatmap
- Assets are reordered using **hierarchical clustering**
- Upper triangle is masked for readability
- Reveals natural groupings (e.g. semiconductors vs platforms)

### 2. Time-Window Comparison
- Correlations computed over:
  - 60 trading days (short-term)
  - 252 trading days (long-term)
- Highlights regime-dependent behaviour

### 3. Correlation Pair Analysis
- Identifies:
  - Most correlated asset pairs (redundant exposure)
  - Least correlated pairs (diversification benefits)

### 4. Diversification Score
- Computes the **average absolute correlation**
- Provides a simple, interpretable diversification metric

### 5. Automated Outputs
Each run generates:
- Correlation matrix (`.csv`)
- Heatmap images (`.png`)
- Pair analysis summary (`.txt`)
- Clustering dendrogram (`.png`)

---

##  Mathematical Background (High-Level)

- **Log returns** are used:
  \[
  r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
  \]

- **Pearson correlation** measures linear co-movement:
  \[
  \rho_{X,Y} = \frac{\mathrm{Cov}(X,Y)}{\sigma_X \sigma_Y}
  \]

- Correlation values lie in \([-1, 1]\):
  - +1: perfect co-movement
  - 0: no linear relationship
  - −1: perfect inverse movement

The focus of the project is **interpretation**, not optimisation.

---

##  Project Structure

