# Python Implementation Analysis - Clustering

## Dataset Overview

**Dataset**: Sales_Cleaned.csv
**Scope**: Customer transaction data from 2023-2025
**Key Features**: Customer demographics, purchase behavior, location, membership level, and transaction details

---

## Project Requirements (CP610 Deliverable #4)

**Per PDF Specification:**
- ✅ Cluster **20% of randomly selected customers**
- ✅ Based on **Number of Transactions** and **Total Spent** only
- ✅ Using **K-Means** algorithm
- ✅ Experiment with different K values (2-10)
- ✅ **Explain what clusters represent (give them names)**

---

## Clustering Implementation Plan

### 1. Objective

Perform customer segmentation using K-Means clustering on 20% of customers to identify distinct purchasing patterns based on transaction frequency and total spending. This simplified approach focuses on the two most critical behavioral metrics as specified in the project requirements.

### 2. Feature Selection Rationale (Per Requirements)

#### Selected Features:

**As per PDF requirements, we use ONLY 2 features:**

1. **Number of Transactions** (Transaction Frequency)
   - **Why**: Direct measure of customer engagement and purchase frequency
   - **Business Insight**: Differentiates one-time buyers from repeat customers
   - **Range**: Integer count (1 to N transactions per customer)

2. **Total Spent** (Monetary Value)
   - **Why**: Direct measure of customer value and spending power
   - **Business Insight**: Identifies high-value vs. low-value customers
   - **Range**: Sum of all transaction amounts per customer

#### Why ONLY These 2 Features?

**Project Requirements Compliance:**
- The PDF explicitly states "based on Number of Transactions and Total Spent"
- This focuses clustering on the most actionable behavioral metrics
- Simpler = more interpretable for stakeholders

**Business Rationale:**
- These two metrics form the **FM (Frequency-Monetary)** component of RFM analysis
- Combined, they create a 2D behavioral space that clearly segments:
  - **High Frequency + High Spending** → VIP customers
  - **High Frequency + Low Spending** → Frequent small buyers
  - **Low Frequency + High Spending** → Occasional big spenders
  - **Low Frequency + Low Spending** → At-risk/inactive customers

**Statistical Rationale:**
- 2 features are sufficient for clear visualization (no dimensionality reduction needed)
- Avoids curse of dimensionality
- Reduces noise from correlated or redundant features
- K-Means performs well in low-dimensional spaces

#### Features Excluded (Not Part of Requirements):

- **Recency**: Not specified in requirements; can be analyzed post-clustering
- **Customer Age, Tenure**: Demographic factors; focus is on behavior only
- **Membership Level, Region**: Not part of core clustering requirements
- **Gross Profit/Margin**: Derived metrics; focus on raw spending behavior

---

### 3. Clustering Algorithm Selection

#### Primary Algorithm: **K-Means Clustering**

**Rationale:**

1. **Scalability**: Efficient with large datasets (10,000+ customer records expected)
2. **Interpretability**: Produces clear, well-separated clusters with centroid-based interpretation
3. **Industry Standard**: Widely used in customer segmentation; established best practices
4. **Numerical Suitability**: Works well with continuous numerical features (our feature set)

**Hyperparameter Tuning:**
- **Optimal K Selection**: Use Elbow Method + Silhouette Score analysis
- **Test Range**: k = 2 to 10 clusters
- **Initialization**: k-means++ for robust centroid initialization
- **Iterations**: max_iter=300, n_init=10 for stability

#### Alternative Considered: Hierarchical Clustering

**Why Not Primary Choice:**
- **Computational Cost**: O(n²) time complexity; slower on large datasets
- **Storage**: Requires full distance matrix (memory intensive)
- **Advantage**: Useful for dendrogram visualization to validate K-Means cluster count

**Usage**: Will be applied as secondary validation method on a sample subset

---

### 4. Data Preprocessing Pipeline

#### Step 1: Load Sales Data
```python
# Read CSV and filter for 2023-2025
df = pd.read_csv('datasets/Sales_Cleaned.csv', sep=';')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
df = df[df['Year'].between(2023, 2025)]
```

#### Step 2: Customer-Level Aggregation
```python
# Aggregate transaction data to customer level (ONLY 2 features needed)
customer_df = df.groupby('Customer ID').agg({
    'Transaction ID': 'count',  # Number of Transactions
    'Total Spent': 'sum',       # Total Spent
}).reset_index()

customer_df.rename(columns={
    'Transaction ID': 'Number_of_Transactions',
    'Total Spent': 'Total_Spent',
}, inplace=True)
```

#### Step 3: Random Sampling (20% of Customers)
**Why 20%?** Per project requirements

```python
# Randomly sample 20% of customers with fixed seed for reproducibility
sample_size = int(len(customer_df) * 0.20)
sampled_df = customer_df.sample(n=sample_size, random_state=42)
```

**Rationale for Random Sampling:**
- **Computational Efficiency**: Faster clustering on smaller dataset
- **Representative Sample**: 20% is statistically sufficient for pattern discovery
- **Reproducibility**: Fixed random_state=42 ensures consistent results
- **Project Compliance**: Explicitly required in PDF

#### Step 4: Feature Scaling
**Method**: StandardScaler (Z-score normalization)

**Why Standardization is Critical:**
- K-Means uses **Euclidean distance** - features must be on same scale
- Number of Transactions (range: 1-100s) vs. Total Spent (range: $10-$10,000s)
- Without scaling, Total Spent would dominate clustering due to larger magnitude
- StandardScaler: (x - mean) / std → mean=0, std=1

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sampled_df[['Number_of_Transactions', 'Total_Spent']])
```

**Why StandardScaler vs. MinMaxScaler?**
- StandardScaler preserves distribution shape (important for normal-ish data)
- Less sensitive to outliers than MinMaxScaler (which squashes to [0,1])
- Industry standard for K-Means preprocessing

#### Step 5: No Outlier Removal
**Decision**: Keep all customers (no winsorization or removal)

**Rationale:**
- High-value customers are **not outliers** - they're our most important segment
- Removing them would lose critical business insights
- K-Means is reasonably robust to outliers (compared to hierarchical methods)
- Outliers often form their own meaningful cluster (e.g., VIP tier)

---

### 5. Model Evaluation Strategy

#### Metric 1: **Elbow Method**
- Plot inertia (within-cluster sum of squares) vs. k
- Identify "elbow point" where marginal improvement diminishes
- **Interpretation**: Balance between model complexity and fit

#### Metric 2: **Silhouette Score**
- Measures cluster cohesion and separation
- Range: [-1, 1], higher is better
- **Threshold**: Aim for score > 0.5 (strong structure)

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, labels)
```

#### Metric 3: **Davies-Bouldin Index**
- Lower values indicate better separation
- Ratio of within-cluster to between-cluster distances
- **Interpretation**: Validates Silhouette findings

#### Metric 4: **Business Validation**
- **Cluster Size Distribution**: Avoid clusters with <5% of customers (too small to action)
- **Centroid Interpretation**: Each cluster must have distinct, actionable characteristics
- **Stability**: Test on train/test splits; clusters should be reproducible

---

### 6. Implementation Steps (Simplified)

```python
# 1. Load data
df = pd.read_csv('datasets/Sales_Cleaned.csv', sep=';')
df = df[df['Year'].between(2023, 2025)]

# 2. Aggregate to customer level (2 features only)
customer_df = df.groupby('Customer ID').agg({
    'Transaction ID': 'count',
    'Total Spent': 'sum'
}).reset_index()

# 3. Sample 20% of customers randomly
sampled_df = customer_df.sample(frac=0.20, random_state=42)

# 4. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sampled_df[['Number_of_Transactions', 'Total_Spent']])

# 5. Experiment with K values (2-10)
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    # Calculate metrics: inertia, silhouette, davies-bouldin

# 6. Select optimal K (based on silhouette score)
optimal_k = k_with_highest_silhouette

# 7. Fit final K-Means model
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
sampled_df['Cluster'] = final_kmeans.fit_predict(X_scaled)

# 8. Profile clusters and assign names
profiles = sampled_df.groupby('Cluster').agg({
    'Number_of_Transactions': ['mean', 'median', 'min', 'max'],
    'Total_Spent': ['mean', 'median', 'min', 'max']
})

# 9. Name clusters based on characteristics
# e.g., "VIP High-Value", "Frequent Buyers", "Occasional Low-Value"

# 10. Visualize
# - Elbow/Silhouette/Davies-Bouldin plots
# - 2D scatter plot (no PCA needed - already 2D!)
# - Bar charts comparing clusters
```

---

### 7. Expected Outputs

1. **customer_cluster_assignments.csv**: Customer ID, Number of Transactions, Total Spent, Cluster
2. **cluster_profiles.csv**: Statistical summary (mean, median, min, max) per cluster
3. **clustering_summary.txt**:
   - Optimal K value
   - Silhouette & Davies-Bouldin scores
   - Cluster names and interpretations
   - Business recommendations per segment
4. **Visualizations**:
   - **k_selection_analysis.png**: Elbow + Silhouette + Davies-Bouldin plots
   - **clusters_scatter_plot.png**: 2D scatter (no PCA needed - already 2D!)
   - **cluster_comparison_bars.png**: Bar charts comparing avg transactions & spending

---

### 8. Why This Approach Over Alternatives?

| Alternative | Why Not Chosen |
|------------|---------------|
| **DBSCAN** | Requires density-based assumptions; customer data is often uniform in density; difficult to interpret "noise" points as customers |
| **Gaussian Mixture Models (GMM)** | Assumes Gaussian distributions; customer data often multi-modal and skewed; added complexity not justified |
| **Spectral Clustering** | Computationally expensive; requires similarity graph; overkill for customer segmentation |
| **Agglomerative Clustering** | Used as validation only; O(n²) complexity; dendrograms useful for small samples but impractical at scale |

---

### 9. Success Criteria

✅ **Technical**:
- Silhouette Score > 0.3 (acceptable cluster structure)
- Davies-Bouldin Index < 2.0 (good separation)
- Clusters represent >5% of sample each (large enough to be actionable)
- Clear elbow point visible in inertia plot

✅ **Business**:
- Each cluster has distinct, interpretable characteristics
- Cluster names accurately reflect behavioral patterns
- Actionable marketing strategies can be derived per segment

---

### 10. Cluster Naming Strategy

**Examples of Expected Cluster Types:**

| Cluster Profile | Name | Business Action |
|----------------|------|-----------------|
| High Transactions + High Spending | **VIP High-Value Customers** | Loyalty rewards, exclusive offers |
| High Transactions + Low Spending | **Frequent Buyers (Low Value)** | Upselling, bundle deals |
| Low Transactions + High Spending | **Big Spenders (Low Frequency)** | Re-engagement campaigns |
| Low Transactions + Low Spending | **Occasional Low-Value Customers** | Activation promotions |

**Naming Methodology:**
1. Calculate mean transactions & spending for entire sample
2. Compare each cluster's averages to overall mean
3. Assign name based on quadrant (High/Low × High/Low)
4. Validate with business stakeholders

---

## Summary

This clustering approach **strictly follows project requirements** while maintaining analytical rigor:

**✅ Requirements Compliance:**
- Uses only 2 features (Number of Transactions, Total Spent)
- Samples 20% of customers randomly
- Experiments with K=2 to K=10
- Provides named, interpretable clusters

**✅ Methodology Strengths:**
- **Simplicity**: 2D feature space → easy visualization and interpretation
- **Scalability**: K-Means handles large datasets efficiently
- **Reproducibility**: Fixed random seed ensures consistent results
- **Business Value**: Directly actionable customer segments

**✅ Why This Approach Works:**
- Focus on behavioral metrics (what customers DO, not who they ARE)
- Avoids overfitting from too many features
- Clear trade-off analysis using multiple metrics (Elbow + Silhouette + Davies-Bouldin)
- Named clusters enable immediate strategic implementation

The 20% sampling requirement actually provides a **benefit**: faster iteration during K selection while still capturing representative customer patterns. The final model can be scaled to 100% of customers if needed for production deployment.
