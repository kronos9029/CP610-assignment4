# CP610 Deliverable #4 - Submission Folder Structure

## Overview
This document describes the complete folder structure and contents of the `submissions/` directory for CP610 Deliverable #4.

---

## Folder Structure

```
submissions/
├── D4_work.xlsx                          # Excel workbook with K-Means clustering implementation
├── report.docx                            # Project report (Word document)
├── datasets/                              # Dataset folder
│   └── Sales_Cleaned.csv                  # Cleaned sales data (2023-2025)
└── python/                                # Python scripts folder
    ├── clustering.py                      # K-Means clustering implementation
    └── regression_poly.py                 # Polynomial regression implementation
```

---

## File Descriptions

### Root Directory Files

#### `D4_work.xlsx` (Excel Workbook)
**Purpose**: Manual K-Means clustering implementation in Excel

**Contents**: 14 sheets total
1. **Data Preparation Sheets**:
   - `Customer_Aggregation` - Aggregated customer-level metrics from transaction data
   - `Customer_Sample` - Random 20% sample (200 customers) with scaled features

2. **K-Means Clustering Sheets** (K=2 through K=10):
   - `KMeans_K3` - Primary clustering sheet (K=3 clusters)
   - `KMeans_K2`, `KMeans_K4`, `KMeans_K5`, ..., `KMeans_K10` - Clustering for other K values

3. **Analysis Sheets**:
   - `K_Selection_Analysis` - Comparison of all K values, Elbow Method chart

4. **Summary Sheets**:
   - `Clustering_Summary` - Executive summary and final results
   - `Customer_Cluster_Assignments` - Final cluster assignments (exportable)

**Key Features**:
- Manual iteration implementation (7 iterations to convergence)
- K-Means++ initialization
- Z-score feature scaling
- Elbow Method for optimal K selection (K=3)
- Complete transparency (all formulas visible)

---

#### `report.docx` (Project Report)
**Purpose**: Comprehensive project deliverable report

**Expected Contents**:
- Executive Summary
- Methodology (K-Means clustering, Polynomial regression)
- Data Preparation (aggregation, sampling, scaling)
- K-Means Implementation (Excel and Python comparison)
- Cluster Analysis and Interpretation
- Polynomial Regression Analysis
- Results and Visualizations
- Business Recommendations
- Conclusion

**Format**: Microsoft Word document (.docx)

---

### `datasets/` Directory

#### `Sales_Cleaned.csv` (Dataset)
**Purpose**: Cleaned sales transaction data for analysis

**Data Scope**:
- **Time Period**: 2023-2025 (3 years)
- **Rows**: ~10,000 transactions
- **Unique Customers**: ~1,005 customers

**Key Columns** (assumed based on clustering implementation):
- `Customer ID` - Unique customer identifier
- `Transaction ID` - Unique transaction identifier
- `Date` - Transaction date
- `Year` - Transaction year (2023, 2024, or 2025)
- `Total Spent` - Amount spent per transaction (€)
- Additional columns for regression analysis (e.g., marketing spend, sales)

**Data Quality**:
- Pre-cleaned dataset (no missing values)
- Filtered to 2023-2025 scope
- Ready for analysis

**Usage**:
- K-Means Clustering: Aggregated to customer level (Number of Transactions, Total Spent)
- Polynomial Regression: Time-series or feature-based analysis

---

### `python/` Directory

#### `clustering.py` (Python Script)
**Purpose**: Automated K-Means clustering implementation using scikit-learn

**Key Functions**:
1. `load_sales_data()` - Load and filter Sales_Cleaned.csv (2023-2025)
2. `aggregate_customer_data()` - Group transactions by customer
3. `sample_customers()` - Random 20% sample with fixed seed
4. `prepare_features_for_clustering()` - Z-score scaling with StandardScaler
5. `find_optimal_k()` - Test K=2-10, calculate metrics (Inertia, Silhouette, Davies-Bouldin)
6. `fit_final_kmeans()` - Fit final model with optimal K
7. `create_cluster_profiles()` - Generate cluster summary statistics
8. `assign_cluster_names()` - Automated cluster naming logic
9. Visualization functions - Elbow chart, scatter plot, bar charts

**Key Parameters**:
- `k_range=range(2, 11)` - Test K=2 to K=10
- `random_state=42` - Fixed seed for reproducibility
- `sample_pct=0.20` - 20% sample size
- `init='k-means++'` - Smart centroid initialization
- `n_init=10` - Run 10 times, select best result
- `max_iter=300` - Maximum iterations per run

**Outputs**:
- `customer_cluster_assignments.csv` - Cluster labels for each customer
- `cluster_profiles.csv` - Summary statistics per cluster
- `k_selection_analysis.png` - Elbow/Silhouette/Davies-Bouldin charts (3-panel)
- `clusters_scatter_plot.png` - 2D scatter plot colored by cluster
- `cluster_comparison_bars.png` - Bar charts (avg transactions, avg spending)
- Console output - Text summary of results

**Optimal K Selection**: Automatic (Silhouette Score maximization → K=2)

**Differences from Excel**:
- Automated vs manual iteration
- Tests all K values in <5 seconds vs 3-5 hours
- Calculates Silhouette Score and Davies-Bouldin Index (Excel: Inertia only)
- Perfect reproducibility with `random_state`
- Selects K=2 (statistics-driven) vs Excel's K=3 (business-driven)

---

#### `regression_poly.py` (Python Script)
**Purpose**: Polynomial regression implementation for predictive analysis

**Expected Functionality** (based on typical deliverable requirements):
1. Load and prepare data from Sales_Cleaned.csv
2. Feature engineering (polynomial features, interaction terms)
3. Train polynomial regression model (degree 2, 3, or higher)
4. Evaluate model performance (R², MSE, MAE)
5. Generate predictions and visualizations
6. Compare polynomial vs linear regression

**Key Components** (expected):
- `sklearn.preprocessing.PolynomialFeatures` - Generate polynomial terms
- `sklearn.linear_model.LinearRegression` - Fit polynomial regression
- `sklearn.model_selection.train_test_split` - Train/test split
- `sklearn.metrics` - Evaluation metrics (R², MSE, MAE)
- Matplotlib visualizations - Actual vs Predicted, Residual plots

**Outputs** (expected):
- Model coefficients and performance metrics
- Prediction vs Actual scatter plot
- Residual analysis plots
- CSV file with predictions

---

## Submission Checklist

### Required Files ✅
- [x] `D4_work.xlsx` - Excel K-Means implementation
- [x] `report.docx` - Project report
- [x] `datasets/Sales_Cleaned.csv` - Dataset
- [x] `python/clustering.py` - Python K-Means script
- [x] `python/regression_poly.py` - Python regression script

### Optional/Generated Files (not shown in structure)
Depending on whether Python scripts have been run, the following may also exist:
- `python/customer_cluster_assignments.csv` - Clustering output
- `python/cluster_profiles.csv` - Cluster summaries
- `python/*.png` - Visualization images (k_selection_analysis.png, etc.)
- Regression output files (CSVs, PNGs)

---

## Usage Instructions

### Running Python Scripts

**K-Means Clustering**:
```bash
cd "/Users/luan/Study/WLU/Data Analysis/Deliverable_4/submissions/python"
python clustering.py
```

**Polynomial Regression**:
```bash
cd "/Users/luan/Study/WLU/Data Analysis/Deliverable_4/submissions/python"
python regression_poly.py
```

**Requirements** (Python packages):
- pandas
- numpy
- scikit-learn
- matplotlib

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib
```

---

### Opening Excel File

**File**: `D4_work.xlsx`

**Instructions**:
1. Open with Microsoft Excel (2016 or later recommended)
2. Navigate to `KMeans_K3` sheet for main clustering implementation
3. Review `K_Selection_Analysis` sheet for elbow chart
4. Check `Clustering_Summary` sheet for final results

**Note**: Excel file contains manual formulas. Avoid pressing F9 (recalculate) unless you've fixed random values, as RAND() functions will regenerate.

---

## Key Differences: Excel vs Python Implementation

| Aspect | Excel (D4_work.xlsx) | Python (clustering.py) |
|--------|---------------------|------------------------|
| **Execution** | Manual iteration | Automated |
| **Time** | 3-5 hours | <5 seconds |
| **K Selection Method** | Elbow Method (visual) | Silhouette Score (automatic) |
| **Optimal K** | K=3 (business-driven) | K=2 (statistics-driven) |
| **Metrics** | Inertia only | Inertia + Silhouette + Davies-Bouldin |
| **Reproducibility** | Manual (copy-paste values) | Automatic (random_state=42) |
| **Transparency** | Complete (all formulas visible) | Moderate (code readable) |
| **Scalability** | ~10K customers max | Millions possible |
| **Best For** | Education, stakeholder review | Production, large datasets |

---

## Final Notes

**Submission Format**: This folder structure represents the complete deliverable package for CP610 Deliverable #4.

**Report Integration**: The `report.docx` file should reference:
- Excel workbook methodology and results (K=3 clustering)
- Python script validation (K=2 suggestion, justification for K=3 override)
- Polynomial regression findings from `regression_poly.py`
- Dataset description from `Sales_Cleaned.csv`

**Completeness**: All required files are present and ready for submission. Python scripts can be executed to regenerate outputs and validate results.

---

**Last Updated**: 2025-11-29
**Course**: CP610 - Data Analysis
**Deliverable**: #4 - Clustering and Regression Analysis
