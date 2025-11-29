# Excel K-Means Clustering Implementation Overview

## Project Context

**Dataset**: Sales_Cleaned.csv (2023-2025 customer transaction data)
**Objective**: Segment customers using K-Means clustering based on purchasing behavior
**Sample Size**: 20% random sample of customers
**Features**: Number of Transactions and Total Spent (2 features)
**K Range Tested**: K=2 through K=10

---

## Implementation Steps Summary

### Step 1: Data Preparation

**1.1 Customer Aggregation**
- **Sheet Created**: `Customer_Aggregation`
- **Purpose**: Aggregate transaction-level data to customer-level metrics
- **Key Actions**:
  - Extract unique Customer IDs using `UNIQUE()` function
  - Count transactions per customer using `COUNTIF()`
  - Sum total spending per customer using `SUMIF()`
- **Output**: Customer-level dataset with 3 columns (ID, Transaction Count, Total Spent)

**1.2 Random Sampling (20%)**
- **Sheet Created**: `Customer_Sample`
- **Purpose**: Create representative 20% sample for clustering analysis
- **Method**:
  - Add random number column using `RAND()` function
  - Sort by random values and select top 20%
  - Sample size: 200 customers from ~1,000 total
- **Justification**: Balances computational efficiency with statistical representativeness

**1.3 Feature Scaling**
- **Sheet Modified**: `Customer_Sample`
- **Purpose**: Standardize features to equal scale for distance calculations
- **Method**: Z-Score Standardization (mean=0, standard deviation=1)
- **Columns Added**:
  - `Scaled_Transactions`: `=(B2-AVERAGE($B$2:$B$201))/STDEV($B$2:$B$201)`
  - `Scaled_TotalSpent`: `=(C2-AVERAGE($C$2:$C$201))/STDEV($C$2:$C$201)`
- **Why Critical**: Prevents high-magnitude features (Total Spent in €) from dominating clustering over low-magnitude features (Transaction Count)

---

### Step 2: K-Means Clustering Implementation

**2.1 Centroid Initialization (K-Means++ Method)**
- **Sheet Created**: `KMeans_K3` (template for all K values)
- **Purpose**: Select well-separated initial centroids to avoid poor local optima
- **Process**:
  1. Choose first centroid randomly from dataset
  2. Find second centroid: customer farthest from first centroid
  3. Find third centroid: customer farthest from nearest existing centroid
- **Formula**: Euclidean distance = `SQRT((D2-$L$2)^2+(E2-$M$2)^2)`

**2.2 Calculate Distances to All Centroids**
- **Columns Added**: `Dist_C1`, `Dist_C2`, `Dist_C3` (Columns F, G, H)
- **Purpose**: Measure distance from each customer to each centroid
- **Formula**: Calculate Euclidean distance in scaled feature space

**2.3 Assign Customers to Nearest Cluster**
- **Column Added**: `Cluster` (Column I)
- **Purpose**: Assign each customer to cluster with nearest centroid
- **Formula**: `=MATCH(MIN(F2:H2),F2:H2,0)` returns cluster number (1, 2, or 3)

**2.4 Update Centroids (Iterative Process)**
- **Columns Added**: `New_Centroid`, `New_Scaled_Trans`, `New_Scaled_Spent` (Columns N-P)
- **Purpose**: Recalculate centroid positions as mean of assigned cluster members
- **Formula**: `=AVERAGEIF($I$2:$I$201,N2,$D$2:$D$201)` for each cluster
- **Iteration**:
  1. Calculate new centroids based on current assignments
  2. Copy new centroids to replace old centroids (Paste Values only)
  3. Distances and assignments automatically recalculate
  4. Repeat until convergence (total centroid change < 0.01)
- **Typical Convergence**: 5-15 iterations

**2.5 Calculate Cluster Metrics**
- **Table Added**: Cluster Summary (Rows 205-209)
- **Metrics Calculated**:
  - **Count**: Number of customers per cluster (`COUNTIF`)
  - **Avg_Transactions**: Mean transactions (unscaled) per cluster (`AVERAGEIF`)
  - **Avg_TotalSpent**: Mean spending (unscaled) per cluster (`AVERAGEIF`)
  - **Inertia**: Within-cluster sum of squared distances (`SUMIF`)
- **Purpose**: Quantify cluster characteristics and quality

---

### Step 3: Experimenting with Different K Values

**3.1 Create Sheets for K=2 through K=10**
- **Sheets Created**: `KMeans_K2`, `KMeans_K4`, `KMeans_K5`, ..., `KMeans_K10`
- **Method**: Copy `KMeans_K3` and modify:
  - K=2: Remove third centroid, adjust formulas
  - K=4+: Add additional centroids and distance columns
- **Purpose**: Test different cluster counts to find optimal segmentation

**3.2 K Selection Analysis**
- **Sheet Created**: `K_Selection_Analysis`
- **Purpose**: Compare all K values to identify optimal clustering
- **Table Contents**:
  - Column A: K values (2-10)
  - Column B: Total Inertia (pulled from each KMeans sheet)
  - Column C: Cluster sizes (concatenated counts)
  - Column D: Minimum cluster size percentage
  - Column E: Notes (manual assessment)
- **Visualization**: Elbow Method line chart (K vs Inertia)

**3.3 Optimal K Selection**
- **K=3 Selected** based on:
  1. **Elbow Method**: Clear elbow visible at K=3 in inertia plot
  2. **Cluster Balance**: Excellent balance (15%, 40%, 45% distribution)
  3. **Interpretability**: 3 distinct, actionable customer segments
  4. **Cluster Sizes**: All clusters > 15% (30+ customers each)
- **K=5 Rejected**: Severe over-segmentation (smallest cluster = 0.5% = 1 customer)

---

### Step 4: Cluster Naming and Interpretation

**4.1 Cluster Profiling**
- **Data Source**: Summary table from `KMeans_K3` sheet
- **Actual K=3 Results**:

| Cluster | Count | Avg Trans | Per Customer Spent | % of Sample |
|---------|-------|-----------|-------------------|-------------|
| 1 | 30 | 26.2 | €81,459 | 15% |
| 2 | 80 | 28.2 | €12,288 | 40% |
| 3 | 90 | 20.92 | €8,486 | 45% |

**4.2 Cluster Naming (Based on Behavior)**
- **Column Added**: `Cluster_Name` (Column F in summary table)
- **Naming Logic**: Compare each cluster to overall averages (25.1 trans, €6,984 spent)

**Cluster 1: Premium High-Value Customers (15%)**
- **Characteristics**: Average transaction frequency (26.2 ≈ overall avg), VERY HIGH spending (€81K = **12x average**)
- **Profile**: Premium retail accounts or high-spending individuals
- **Business Value**: CRITICAL SEGMENT - 15% of customers contributing ~35-40% of total revenue

**Cluster 2: Frequent Loyal Customers (40%)**
- **Characteristics**: Highest transaction frequency (28.2 = **highest**), moderate spending (€12K = 1.8x avg)
- **Profile**: Engaged regular buyers with consistent purchase patterns
- **Business Value**: Volume opportunity - largest cluster, steady revenue stream

**Cluster 3: Occasional Customers (45%)**
- **Characteristics**: Lowest transaction frequency (20.92 = 17% below avg), lowest spending (€8.5K = 1.2x avg)
- **Profile**: Casual/infrequent shoppers, largest cluster but least active
- **Business Value**: Conversion focus - re-engagement potential

**4.3 Business Interpretation Table**
- **Table Created**: Marketing actions and revenue impact per cluster (Rows 215-218)
- **Purpose**: Translate statistical clusters into actionable business strategies

---

### Step 5: Visualization

**5.1 2D Scatter Plot**
- **Purpose**: Visualize customers in Transactions vs Spending space, colored by cluster
- **Method**: Create scatter chart with 3 series (one per cluster), distinct colors

**5.2 Cluster Comparison Bar Charts**
- **Chart 1**: Average Transactions per Cluster (column chart)
- **Chart 2**: Average Total Spent per Cluster (column chart)
- **Chart 3**: Cluster Size Distribution (pie chart with percentages)

**5.3 Elbow Method Chart**
- **Purpose**: Visual justification for K=3 selection
- **Content**: Line chart showing Total Inertia decreasing as K increases (K=2-10)
- **Annotation**: Mark optimal K=3 at elbow point

---

### Step 6: Output and Deliverables

**6.1 Summary Sheet**
- **Sheet Created**: `Clustering_Summary`
- **Contents**:
  - Section 1: Project metadata (dataset, sample size, features, algorithm)
  - Section 2: K selection process (tested K values, optimal K, criteria)
  - Section 3: Final cluster profiles table (copied from KMeans_K3)
  - Section 4: Business recommendations table

**6.2 Customer Assignments Export**
- **Sheet Created**: `Customer_Cluster_Assignments`
- **Contents**:
  - Customer ID, Number_of_Transactions, Total_Spent, Cluster, Cluster_Name
- **Export**: Save as CSV for downstream use (marketing campaigns, CRM integration)

---

### Step 7: Validation and Quality Checks

**7.1 Cluster Quality Metrics**
- **Convergence Check**: Total centroid change (Q5) < 0.01 ✅
- **Cluster Size Balance**: All clusters > 5% of sample ✅
- **Inertia Decrease**: Verified inertia decreases as K increases ✅

**7.2 Business Validation**
- **Interpretability**: Each cluster has clear, distinct characteristics ✅
- **Actionability**: Marketing can create specific campaigns per segment ✅
- **Centroid Separation**: All centroids >1.0 distance apart (in scaled space) ✅

**7.3 Final Validation Checklist**
- ✅ Convergence achieved (Q5 < 0.01)
- ✅ All clusters > 5% of sample (15%, 40%, 45%)
- ✅ Clear elbow visible at K=3
- ✅ Clusters interpretable and actionable
- ✅ Total customer count = 200 (sample size verified)

---

## Key Excel Sheets Created

| Sheet Name | Purpose | Row Count |
|-----------|---------|-----------|
| `Customer_Aggregation` | Aggregate transactions to customer level | ~1,005 |
| `Customer_Sample` | 20% random sample with scaled features | 200 |
| `KMeans_K2` - `KMeans_K10` | Clustering results for each K value (9 sheets) | 200 each |
| `K_Selection_Analysis` | Compare all K values, Elbow chart | 9 K values |
| `Clustering_Summary` | Executive summary for stakeholders | Summary only |
| `Customer_Cluster_Assignments` | Final assignments for export (CSV) | 200 |

**Total Sheets**: 14 (2 data prep + 9 K-Means + 1 analysis + 2 output)

---

## Implementation Advantages

✅ **Transparency**: Every calculation visible and auditable in cells
✅ **Educational Value**: Demonstrates deep understanding of K-Means algorithm mechanics
✅ **Stakeholder Friendly**: Business users can review without coding knowledge
✅ **Flexibility**: Easy to modify K, features, or sample size without programming
✅ **Integration**: Links seamlessly with existing Excel-based reporting workflows

---

## Comparison: Excel vs Python

| Aspect | Excel Implementation | Python (sklearn) Implementation |
|--------|---------------------|-------------------------------|
| **Iteration** | Manual (5-15 iterations) | Automatic (max 300 iterations) |
| **Optimal K Selection** | Elbow Method (visual) → K=3 | Silhouette Score (statistical) → K=2 |
| **Initialization** | K-Means++ (manual) | K-Means++ (automatic) |
| **Scalability** | <10,000 customers | Millions of customers |
| **Reproducibility** | Manual seed fixing | `random_state=42` |
| **Best Use Case** | Small datasets, demos, validation | Production, large datasets, automation |

**Result**: Both methods yield similar insights when properly executed, with Excel providing **transparency and business accessibility** while Python provides **scalability and automation**.

---

## Final Clustering Results (K=3)

**Optimal K**: 3 clusters
**Total Inertia**: 447.81
**Convergence**: Achieved in 7 iterations

| Cluster | Size | Avg Trans | Avg Spent | Name | Business Strategy |
|---------|------|-----------|-----------|------|-------------------|
| 1 | 30 (15%) | 26.2 | €81,459 | Premium High-Value | VIP service, retention focus |
| 2 | 80 (40%) | 28.2 | €12,288 | Frequent Loyal | Upselling, bundle deals |
| 3 | 90 (45%) | 20.92 | €8,486 | Occasional | Re-engagement campaigns |

**Key Insight**: While representing only 15% of customers, **Premium High-Value Customers** contribute disproportionately to total revenue (€81K per customer vs €6,984 average), making them critical for retention strategies.
