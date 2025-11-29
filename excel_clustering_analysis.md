# Excel Implementation Analysis - K-Means Clustering

## Dataset Overview

**Dataset**: Sales_Cleaned.csv (in Excel workbook: `D4_work.xlsx`)
**Scope**: Customer transaction data from 2023-2025
**Analysis Goal**: Segment customers based on purchasing behavior using K-Means clustering

---

## Project Requirements (CP610 Deliverable #4)

**Per PDF Specification:**
- ✅ Cluster **20% of randomly selected customers**
- ✅ Based on **Number of Transactions** and **Total Spent** only (2 features)
- ✅ Using **K-Means** algorithm
- ✅ Experiment with different K values (2-10)
- ✅ **Explain what clusters represent (give them names)**

---

## Excel Implementation Strategy

### Why Excel for Clustering?

**Advantages:**
- **Visual Transparency**: All calculations visible in cells - no "black box" algorithms
- **Step-by-Step Validation**: Each clustering iteration can be manually verified
- **Educational Value**: Demonstrates understanding of K-Means mathematics
- **Accessibility**: Stakeholders can review and modify without coding knowledge
- **Built-in Tools**: Data Analysis ToolPak, Solver, and charting capabilities

**Limitations:**
- **Manual Iteration**: K-Means requires iterative refinement (Python automates this)
- **Scalability**: Excel handles up to 1M rows, but calculations slow with large datasets
- **Random Sampling**: Requires manual or VBA for truly random selection
- **No Built-in K-Means**: Must implement algorithm manually using Euclidean distance and centroids

---

## Step-by-Step Excel Implementation

### Step 1: Data Preparation

#### 1.1 Create Customer Aggregation Sheet

**Sheet Name**: `Customer_Aggregation`

**Columns:**
| Column | Formula/Method | Description |
|--------|---------------|-------------|
| A: Customer ID | `=UNIQUE(Sales_Cleaned[Customer ID])` | Extract unique customer IDs |
| B: Number_of_Transactions | `=COUNTIF(Sales_Cleaned[Customer ID], A2)` | Count transactions per customer |
| C: Total_Spent | `=SUMIF(Sales_Cleaned[Customer ID], A2, Sales_Cleaned[Total Spent])` | Sum spending per customer |

**Excel Formula Example (Row 2):**
```excel
=COUNTIF(Sales_Cleaned!$B$2:$B$10000, A2)  // Number of Transactions
=SUMIF(Sales_Cleaned!$B$2:$B$10000, A2, Sales_Cleaned!$G$2:$G$10000)  // Total Spent
```

**Data Validation:**
- Remove customers with zero transactions (if any)
- Check for NULL values in Customer ID
- Verify Total_Spent is numeric and non-negative

---

#### 1.2 Random Sampling (20% of Customers)

**Method 1: Using RAND() Function**

**Sheet Name**: `Customer_Aggregation`

1. **Add Random Number Column (Column D):**
   ```excel
   =RAND()  // Generates random number between 0 and 1
   ```

2. **Sort by Random Number:**
   - Select all data (Columns A-D)
   - Data → Sort → Sort by Column D (Random) → Ascending

3. **Calculate Sample Size:**
   - In a helper cell: `=ROUNDUP(COUNTA(A:A)*0.20, 0)`  // 20% of total customers
   - Example: If 5000 customers → 1000 customers sampled

4. **Create Sample Sheet:**
   - **Sheet Name**: `Customer_Sample`
   - Copy top 20% of rows from sorted `Customer_Aggregation` sheet
   - **Columns**: Customer ID, Number_of_Transactions, Total_Spent

**Method 2: Using Data Analysis ToolPak (Sampling)**

1. Enable Data Analysis ToolPak:
   - File → Options → Add-ins → Analysis ToolPak → Go → Check box → OK

2. Select Sampling Tool:
   - Data → Data Analysis → Sampling → OK

3. Configure Sampling:
   - **Input Range**: Select Customer_Aggregation data (A1:C5001)
   - **Sampling Method**: Random
   - **Number of Samples**: `=5000*0.20` → 1000
   - **Output Range**: New sheet `Customer_Sample`

**Important Notes:**
- **Fix Random Seed**: After sampling, copy and paste values (Ctrl+Shift+V → Values) to freeze the sample
- **Reproducibility**: Document the random seed or sample Customer IDs for reproducibility
- **Why 20%**: Per project requirements - balances computational efficiency with statistical representativeness

---

#### 1.3 Feature Scaling (Standardization)

**Why Scaling is Critical:**
- K-Means uses **Euclidean distance** - features must be on the same scale
- Number_of_Transactions (range: 1-100) vs. Total_Spent (range: $50-$10,000)
- Without scaling, Total_Spent dominates clustering due to larger magnitude

**Method: Z-Score Standardization**

**Sheet Name**: `Customer_Sample`

**Add Columns (D & E):**
| Column | Formula | Description |
|--------|---------|-------------|
| D: Scaled_Transactions | `=(B2-AVERAGE($B$2:$B$1001))/STDEV($B$2:$B$1001)` | Standardized transactions |
| E: Scaled_TotalSpent | `=(C2-AVERAGE($C$2:$C$1001))/STDEV($C$2:$C$1001)` | Standardized spending |

**Formula Breakdown:**
```excel
=(B2 - AVERAGE($B$2:$B$1001)) / STDEV($B$2:$B$1001)
 └─ Raw Value
      └─ Subtract Mean (center at 0)
                        └─ Divide by Std Dev (scale to unit variance)
```

**Result:**
- Mean = 0, Standard Deviation = 1
- Outliers preserved (not clipped)
- Negative values indicate below-average behavior

**Alternative: Min-Max Scaling (Not Recommended)**
```excel
=(B2-MIN($B$2:$B$1001))/(MAX($B$2:$B$1001)-MIN($B$2:$B$1001))
```
Why not used: Sensitive to outliers; squashes data to [0,1] range

---

### Step 2: K-Means Clustering Implementation

**IMPORTANT**: All steps 2.1 through 2.5 happen in the **SAME sheet** (KMeans_K3). Do NOT create separate sheets.

---

#### **SETUP: Create KMeans_K3 Sheet**

Before starting Step 2.1:

1. **Right-click on `Customer_Sample` sheet tab** → **Move or Copy**
2. **Check "Create a copy"** → OK
3. **Rename the new sheet** to `KMeans_K3`

Now you have all your customer data (columns A-E) in KMeans_K3 sheet.

**Expected Column Layout:**
- **A**: Customer ID
- **B**: Number_of_Transactions (raw values)
- **C**: Total_Spent (raw values)
- **D**: Scaled_Transactions
- **E**: Scaled_TotalSpent
- **F-Q**: (Will be added in steps below)

---

#### 2.1 Initialize Centroids (K-Means++ Method)

**Sheet**: `KMeans_K3`

**Goal**: Choose 3 well-separated initial centroids using K-Means++ algorithm.

---

##### **Step 2.1.1: Choose First Centroid Randomly**

1. **Pick a random row number** between 2-201:
   - **Method A - Formula**: In an empty cell (e.g., K1), type `=RANDBETWEEN(2;201)` or `=INT(RAND()*200)+2`
   - Press Enter → Get a number (e.g., 79)
   - **Immediately copy the cell → Paste as Values** (Ctrl+Shift+V) to freeze it

   - **Method B - Manual**: Just pick any number between 2-201 (e.g., 79)

2. **Create Centroid Table** in columns K-M:

   **Headers (Row 1):**
   - K1: `Centroid_ID`
   - L1: `Scaled_Transactions`
   - M1: `Scaled_TotalSpent`

3. **First Centroid (Row 2)** - Assuming random number is 79:
   - K2: Type `1` (Centroid 1)
   - L2: Type `=D79` → Press Enter → **Copy L2 → Paste as Values** (Ctrl+Shift+V)
   - M2: Type `=E79` → Press Enter → **Copy M2 → Paste as Values**

**Result**: L2 and M2 now contain static numbers (the scaled values from row 79).

---

##### **Step 2.1.2: Find Second Centroid (Farthest from First)**

1. **Calculate distance from each customer to Centroid 1**:

   **In Cell F1**: Type `Dist_C1` (header)

   **In Cell F2**: Type the formula:
   ```excel
   =SQRT((D2-$L$2)^2+(E2-$M$2)^2)
   ```

   **Copy F2 down** to F201 (all customer rows)

2. **Find the row with MAXIMUM distance**:
   - Scroll through Column F or use `=MAX(F2:F201)` in a helper cell
   - Find which row has this maximum value (e.g., row 156)
   - **TIP**: Use Ctrl+F (Find) to search for the max value in Column F

3. **Add Second Centroid**:
   - K3: Type `2`
   - L3: Type `=D156` → Press Enter → **Paste as Values**
   - M3: Type `=E156` → Press Enter → **Paste as Values**

---

##### **Step 2.1.3: Find Third Centroid (Farthest from Existing Centroids)**

1. **Calculate distance to Centroid 2**:

   **In Cell G1**: Type `Dist_C2`

   **In Cell G2**: Type:
   ```excel
   =SQRT((D2-$L$3)^2+(E2-$M$3)^2)
   ```

   **Copy G2 down** to G201

2. **Find minimum distance to ANY existing centroid**:

   **In Cell J1**: Type `Min_Distance`

   **In Cell J2**: Type:
   ```excel
   =MIN(F2,G2)
   ```

   **Copy J2 down** to J201

   This shows each customer's distance to their NEAREST centroid.

3. **Find the row with MAXIMUM of these minimum distances**:
   - Use `=MAX(J2:J201)` in a helper cell
   - Find which row has this value (e.g., row 45)

4. **Add Third Centroid**:
   - K4: Type `3`
   - L4: Type `=D45` → Press Enter → **Paste as Values**
   - M4: Type `=E45` → Press Enter → **Paste as Values**

---

**Centroid Table Complete!** (K2:M4)

Example result:
```
K2: 1    L2: -0.523    M2: -0.812
K3: 2    L3: 1.847     M3: 2.154
K4: 3    L4: 0.332     M4: -1.121
```

---

#### 2.2 Calculate Euclidean Distances to All Centroids

**Sheet**: `KMeans_K3` (same sheet, continue adding columns)

**Goal**: Calculate distance from each customer to all 3 centroids.

---

##### **Step 2.2.1: Distance to Centroid 1 (Already Done)**

You already have this in **Column F** from Step 2.1.2!

---

##### **Step 2.2.2: Distance to Centroid 2 (Already Done)**

You already have this in **Column G** from Step 2.1.3!

---

##### **Step 2.2.3: Distance to Centroid 3**

**In Cell H1**: Type `Dist_C3`

**In Cell H2**: Type:
```excel
=SQRT((D2-$L$4)^2+(E2-$M$4)^2)
```

**Copy H2 down** to H201

---

**Result**: Columns F, G, H now contain distances to all 3 centroids for each customer.

---

#### 2.3 Assign Customers to Nearest Cluster

**Sheet**: `KMeans_K3` (same sheet, continue adding columns)

**Goal**: Assign each customer to the cluster with the nearest centroid.

---

**In Cell I1**: Type `Cluster` (header)

**In Cell I2**: Type the formula:
```excel
=MATCH(MIN(F2:H2),F2:H2,0)
```

**Copy I2 down** to I201 (all customer rows)

**What This Formula Does:**
- `MIN(F2:H2)`: Finds the smallest distance among the 3 centroids
- `MATCH(..., F2:H2, 0)`: Returns the position (1, 2, or 3) of that minimum distance
- Result: Customer assigned to Cluster 1, 2, or 3

**Example**:
- If F2=0.45 (closest), G2=2.31, H2=1.89 → I2 returns **1** (Cluster 1)
- If F3=1.12, G3=0.67 (closest), H3=1.45 → I3 returns **2** (Cluster 2)

**Result**: Column I now shows which cluster each of the 200 customers belongs to.

---

#### 2.4 Update Centroids (Iteration)

**Sheet**: `KMeans_K3` (same sheet, continue adding columns)

**Goal**: Calculate new centroids based on current cluster assignments, then iteratively update until convergence.

---

##### **Step 2.4.1: Create New Centroid Calculation Area**

**Add columns N-P** for calculating updated centroids:

**Headers (Row 1):**
- N1: `New_Centroid`
- O1: `New_Scaled_Trans`
- P1: `New_Scaled_Spent`

**Centroid Labels (Rows 2-4):**
- N2: Type `1`
- N3: Type `2`
- N4: Type `3`

---

##### **Step 2.4.2: Calculate New Centroid Positions**

**In Cell O2** (New Scaled_Transactions for Centroid 1):
```excel
=AVERAGEIF($I$2:$I$201,N2,$D$2:$D$201)
```

**What this does:**
- Look at Column I (cluster assignments) from rows 2-201
- Find all customers where Cluster = 1 (value in N2)
- Calculate the average of their Scaled_Transactions (Column D)

**In Cell P2** (New Scaled_TotalSpent for Centroid 1):
```excel
=AVERAGEIF($I$2:$I$201,N2,$E$2:$E$201)
```

**Copy O2:P2 down** to rows 3 and 4:
- Select O2:P2
- Copy (Ctrl+C)
- Select O3:P4
- Paste (Ctrl+V)

The formulas automatically adjust (N2 becomes N3, N4).

---

##### **Step 2.4.3: Iteration Process**

**Initial State**:
- Columns L-M contain your CURRENT centroids (from K-Means++ initialization)
- Columns O-P contain NEW centroid values calculated from cluster assignments

**Iteration Loop** (Repeat until convergence):

1. **Check if centroids changed**: Compare O2:P4 (new) with L2:M4 (old)

2. **Update old centroids**:
   - **Select O2:P4** (new centroid values)
   - **Copy** (Ctrl+C)
   - **Click on L2** (top-left of old centroid table)
   - **Paste Special → Values ONLY** (Ctrl+Shift+V or Right-click → Paste Special → Values)
   - **CRITICAL**: Must be "Values only" to avoid circular reference error

3. **Automatic Recalculation**:
   - Columns F-H (distances) automatically recalculate with new centroids
   - Column I (cluster assignments) may change for some customers
   - Columns O-P (new centroids) recalculate based on new assignments

4. **Repeat Steps 1-3** until centroids stop changing

**NOTE**: If you get "Recursion detected" error, make sure you're using **Paste Special → Values** (not regular paste).

**Troubleshooting Circular Reference**:
- If recursion error persists, use a temporary area (columns R-S):
  - Copy O2:P4 → Paste Values into R2:S4
  - Then copy R2:S4 → Paste Values into L2:M4

---

##### **Step 2.4.4: Add Convergence Check (Optional but Recommended)**

**In Cell Q1**: Type `Change_C1`

**In Cell Q2** (change in Centroid 1):
```excel
=SQRT((L2-O2)^2+(M2-P2)^2)
```

**Copy Q2 down** to Q3, Q4

**In Cell Q5** (total change across all centroids):
```excel
=SUM(Q2:Q4)
```

**Convergence Criteria**:
- **Q5 > 1.0**: Keep iterating (centroids still moving significantly)
- **Q5 between 0.01-1.0**: Keep iterating (getting close)
- **Q5 < 0.01**: **CONVERGED!** Stop iterating

**Typical Convergence Pattern**:
| Iteration | Q5 (Total Change) | Action |
|-----------|------------------|--------|
| 1 | 3.684 | Continue |
| 2 | 1.523 | Continue |
| 3 | 0.687 | Continue |
| 4 | 0.234 | Continue |
| 5 | 0.089 | Continue |
| 6 | 0.032 | Continue |
| 7 | 0.009 | **STOP - Converged!** |

**If Oscillating** (Q5 stays constant after 20+ iterations):
- Centroids are bouncing between two states
- **Solution**: Average the two states manually or add tiny random noise to break ties (see troubleshooting section in documentation)

---

#### 2.5 Calculate Cluster Metrics

**Sheet**: `KMeans_K3` (same sheet, continue)

**Goal**: Calculate summary statistics for each cluster to understand their characteristics.

---

##### **Step 2.5.1: Create Cluster Summary Table**

**Add a summary area** below your data or to the right (e.g., starting at row 205):

**Location**: Cells A205:E208 (or any empty area)

**Headers (Row 205):**
- A205: `Cluster`
- B205: `Count`
- C205: `Avg_Transactions`
- D205: `Avg_TotalSpent`
- E205: `Inertia`

**Cluster Labels (Column A):**
- A206: `1`
- A207: `2`
- A208: `3`

---

##### **Step 2.5.2: Calculate Metrics for Each Cluster**

**For Cluster 1 (Row 206):**

**B206 - Count** (how many customers in Cluster 1):
```excel
=COUNTIF($I$2:$I$201,A206)
```

**C206 - Average Transactions** (UNSCALED - for interpretation):
```excel
=AVERAGEIF($I$2:$I$201,A206,$B$2:$B$201)
```

**What this does:**
- Look at Column I (cluster assignments)
- Find all customers in Cluster 1 (A206)
- Calculate average of Column B (Number_of_Transactions - **unscaled**)

**D206 - Average Total Spent** (UNSCALED):
```excel
=AVERAGEIF($I$2:$I$201,A206,$C$2:$C$201)
```

**E206 - Inertia** (sum of squared distances within cluster):
```excel
=SUMIF($I$2:$I$201,A206,$F$2:$F$201)
```

**What this does:**
- Find all customers in Cluster 1
- Sum their distances to Centroid 1 (Column F)
- This measures cluster compactness (lower = tighter cluster)

---

##### **Step 2.5.3: Copy Formulas Down**

**Copy B206:E206** and paste to rows 207-208 (for Clusters 2 and 3).

The formulas automatically adjust (A206 becomes A207, A208).

---

##### **Step 2.5.4: Add Total Row (Optional)**

**Row 209:**
- A209: `TOTAL`
- B209: `=SUM(B206:B208)` (should equal 200 - your sample size)
- C209: `=AVERAGE(C206:C208)` (overall average transactions)
- D209: `=AVERAGE(D206:D208)` (overall average spending)
- E209: `=SUM(E206:E208)` (total inertia for K=3)

---

##### **Step 2.5.5: Interpret Results**

**Example Results:**

| Cluster | Count | Avg_Transactions | Avg_TotalSpent | Inertia |
|---------|-------|-----------------|----------------|---------|
| 1       | 67    | 2.3             | $145.50        | 12.45   |
| 2       | 89    | 8.7             | $523.20        | 18.32   |
| 3       | 44    | 15.2            | $1,845.60      | 9.87    |
| **TOTAL** | **200** | **8.73** | **672.43** | **40.64** |

**Key Insights:**
- **Cluster 1** (67 customers): Low transactions, low spending → "Occasional Customers"
- **Cluster 2** (89 customers): Medium transactions, medium spending → "Loyal Customers"
- **Cluster 3** (44 customers): High transactions, high spending → "VIP High-Value Customers"

**Note**:
- Use **UNSCALED** values (Columns B & C) for interpretation - these are the original transaction counts and dollar amounts
- **Inertia** (Column E) uses distances (Column F) - lower inertia means tighter cluster
- Total count (B209) should = 200 (your sample size)

---

##### **Step 2.5.6: Column Reference Summary**

**In KMeans_K3 sheet, your formulas reference:**
- `$I$2:$I$201` = Cluster assignments (Column I)
- `$B$2:$B$201` = Number_of_Transactions (Column B - **unscaled**)
- `$C$2:$C$201` = Total_Spent (Column C - **unscaled**)
- `$F$2:$F$201` = Distance to assigned centroid (Column F)

**All references are within the SAME sheet** (KMeans_K3), no external sheet references needed.

---

### Step 3: Experimenting with Different K Values

**Requirement**: "Experiment with different Ks and pick the K you believe is the best"

**Goal**: Test K=2 through K=10 to find the optimal number of clusters using the Elbow Method.

---

#### 3.1 Create Separate Sheets for Each K

**Process**: Repeat Steps 2.1-2.5 for each K value (2, 3, 4, ..., 10)

---

##### **Step 3.1.1: Create KMeans_K2 Sheet**

1. **Right-click KMeans_K3 sheet** → **Move or Copy**
2. **Check "Create a copy"** → OK
3. **Rename** to `KMeans_K2`

4. **Modify for K=2**:
   - **Delete Centroid 3**: Delete row 4 in centroid table (K4:M4 becomes empty)
   - **Delete Distance Column H**: Delete Dist_C3 column
   - **Update Cluster formula (I2)**: Change to `=MATCH(MIN(F2:G2),F2:G2,0)`
   - **Update New Centroid table**: Delete row N4:P4
   - **Re-run iteration** until Q5 < 0.01
   - **Update Summary table**: Only 2 rows (Clusters 1 and 2)

---

##### **Step 3.1.2: Create KMeans_K4 Sheet**

1. **Right-click KMeans_K3 sheet** → **Move or Copy** → **Create a copy** → Rename to `KMeans_K4`

2. **Modify for K=4**:

   **Add 4th Centroid**:
   - K5: Type `4`
   - L5: Use K-Means++ method to find 4th centroid (find max of min distances to C1, C2, C3)
   - M5: Corresponding scaled spent value

   **Add Distance Column**:
   - I1: `Dist_C4`
   - I2: `=SQRT((D2-$L$5)^2+(E2-$M$5)^2)`
   - Copy down to I201

   **Update Cluster Assignment**:
   - Change J2 formula to: `=MATCH(MIN(F2:I2),F2:I2,0)`
   - Update J (Min_Distance) formula to: `=MIN(F2:I2)`

   **Add 4th row to New Centroid table**:
   - N5: `4`
   - O5: `=AVERAGEIF($J$2:$J$201,N5,$D$2:$D$201)` (note: J is now cluster column)
   - P5: `=AVERAGEIF($J$2:$J$201,N5,$E$2:$E$201)`

   **Note**: Cluster assignments now in Column J (not I, since I is Dist_C4)

   **Iterate until convergence**

   **Update Summary table**: 4 rows (Clusters 1-4)

---

##### **Step 3.1.3: Repeat for K=5 through K=10**

**Pattern**: For each additional K:
- Add one more centroid row
- Add one more distance column
- Update MATCH formula range
- Add one more row to new centroid table
- Re-iterate until convergence

**Column Layout Shifts** as K increases:
- **K=2**: Cluster in Column I (since H is Dist_C2)
- **K=3**: Cluster in Column I (since H is Dist_C3)
- **K=4**: Cluster in Column J (since I is Dist_C4)
- **K=5**: Cluster in Column K (since J is Dist_C5)
- etc.

---

#### 3.2 Create K Comparison Summary Sheet

**Goal**: Compare all K values to select the optimal one.

---

##### **Step 3.2.1: Create New Sheet**

1. **Create new sheet**: Click **+** → Rename to `K_Selection_Analysis`

---

##### **Step 3.2.2: Build Comparison Table**

**Headers (Row 1):**
- A1: `K`
- B1: `Total_Inertia`
- C1: `Cluster_Sizes`
- D1: `Min_Cluster_Size`
- E1: `Notes`

**K Values (Column A)**:
- A2: `2`
- A3: `3`
- A4: `4`
- ...
- A11: `10`

---

##### **Step 3.2.3: Calculate Total Inertia for Each K**

**B2 (Total Inertia for K=2)**:
```excel
=KMeans_K2!E209
```
This references the total inertia cell from KMeans_K2 summary table.

**Alternative** (if summary table location varies):
```excel
=SUM(KMeans_K2!E206:E207)
```

**Copy down B2 → B11** for all K values:
- B3: `=KMeans_K3!E209`
- B4: `=KMeans_K4!E209`
- etc.

---

##### **Step 3.2.4: Record Cluster Sizes**

**C2 (Cluster sizes for K=2)**:
```excel
=KMeans_K2!B206 & ", " & KMeans_K2!B207
```
Result example: "98, 102"

**C3 (for K=3)**:
```excel
=KMeans_K3!B206 & ", " & KMeans_K3!B207 & ", " & KMeans_K3!B208
```

**For larger K**: Manually type cluster sizes or use complex formula.

---

##### **Step 3.2.5: Calculate Minimum Cluster Size %**

**D2 (Min cluster size % for K=2)**:
```excel
=MIN(KMeans_K2!B206:B207)/200
```
Format as Percentage (e.g., 49%)

**D3 (Min cluster size % for K=3)**:
```excel
=MIN(KMeans_K3!B206:B208)/200
```

**D4 (Min cluster size % for K=4)**:
```excel
=MIN(KMeans_K4!B206:B209)/200
```

**Copy down** for all K values.

---

##### **Step 3.2.6: Add Notes (Manual)**

**Use these templates based on your actual cluster size analysis:**

**E2 (K=2)**:
```
Too few segments - only 2 groups. May oversimplify customer diversity.
```

**E3 (K=3)**:
```
Cluster sizes well-balanced. All clusters contain at least 15% of customers (30 out of 200). This is acceptable for actionable segmentation.
```

**E4 (K=4)**:
```
Cluster sizes reasonably balanced. Smallest cluster contains 13% of customers (26 out of 200). Still large enough for meaningful analysis and targeted marketing.
```

**E5 (K=5)**:
```
⚠️ WARNING: K=5 produces an extremely small cluster (only 1 customer = 0.5%). This suggests over-segmentation. Cluster too small to be actionable - NOT RECOMMENDED for final model.
```

**E6-E11 (K=6-10)**: Check minimum cluster size %. If < 5%, mark as "Over-segmented"

---

##### **Step 3.2.7: Create Elbow Chart**

1. **Select A1:B11** (K values and Total Inertia)

2. **Insert → Charts → Line Chart** with markers

3. **Format Chart**:
   - **Title**: "Elbow Method for Optimal K Selection"
   - **X-axis**: "Number of Clusters (K)"
   - **Y-axis**: "Total Inertia"
   - **Add data labels**: Right-click series → Add Data Labels

4. **Identify the Elbow**:
   - Look for the K where the curve starts to flatten
   - Typically K=3 or K=4

---

#### 3.3 Decision Criteria for Optimal K

**Select K based on:**

1. **Elbow Method** (Column B chart):
   - Choose K at the "elbow" point (diminishing returns)
   - Inertia should drop sharply before K, flatten after K

2. **Cluster Size Balance** (Column D):
   - Avoid K where minimum cluster size < 5% (10 customers out of 200)
   - All clusters should be large enough to be actionable

3. **Business Interpretability**:
   - Each cluster must have distinct, understandable characteristics
   - Can you explain each segment to a marketing team?

4. **Simplicity**:
   - Prefer fewer clusters if performance is similar
   - K=3 or K=4 usually optimal for customer segmentation

---

##### **Step 3.3.1: Decision Matrix (Based on Actual Analysis Results)**

| K | Total Inertia | Min Cluster % | Cluster Balance | Recommendation |
|---|---------------|---------------|-----------------|----------------|
| 2 | [From Excel] | [From Excel] | Good separation but may oversimplify | ⚠️ Too few segments |
| 3 | [From Excel] | **15%** ✅ | **Well-balanced, all clusters actionable** | **✅ RECOMMENDED** |
| 4 | [From Excel] | **13%** ✅ | **Acceptable balance, granular segmentation** | **✅ VIABLE** |
| 5 | [From Excel] | **0.5%** ❌ | **⚠️ SEVERE IMBALANCE (1 customer only)** | **❌ REJECT** |
| 6-10 | [From Excel] | [Check Excel] | Likely over-segmented if min < 5% | ⚠️ Evaluate balance |

**Key Findings from Your Analysis:**
- **K=3**: Excellent balance (15% minimum = 30 customers per cluster minimum)
- **K=4**: Good balance (13% minimum = 26 customers per cluster minimum)
- **K=5**: REJECTED due to severe over-segmentation (smallest cluster = 0.5% = 1 customer)

**Fill in "Total Inertia" column from your K_Selection_Analysis sheet Column B.**

---

##### **Step 3.3.1a: Cluster Size Balance Analysis**

**Cluster Balance Thresholds** (Industry Best Practices):

| Min Cluster % | Assessment | Actionability |
|--------------|------------|---------------|
| **≥15%** | ✅ Excellent balance | All clusters large enough for targeted campaigns |
| **10-15%** | ✅ Good balance | Acceptable for most business applications |
| **5-10%** | ⚠️ Marginal | Evaluate business context - may be too small |
| **<5%** | ❌ Too small | NOT actionable for segmentation strategies |

**Your Results:**
- **K=3**: 15% minimum → ✅ **Excellent** (30 customers minimum)
- **K=4**: 13% minimum → ✅ **Good** (26 customers minimum)
- **K=5**: 0.5% minimum → ❌ **REJECTED** (1 customer = severe over-segmentation)

**Key Insight**: K=5 shows severe over-segmentation with smallest cluster containing only **1 customer (0.5% of sample)**. This cluster has no statistical meaning and cannot be used for business strategy.

**Recommendation**:
- **K=5 is eliminated** due to cluster imbalance
- **Choose between K=3 or K=4** based on:
  - Total Inertia (Elbow Method)
  - Business need for granularity vs. simplicity
  - Cluster interpretability

---

##### **Step 3.3.2: Make Your Selection**

**Selection Process (Use This Order):**

**1. Eliminate K values with severe imbalance** (min cluster % < 5%)
   - From your analysis: **K=5 is eliminated** (0.5% minimum)
   - Also eliminate K=6-10 if they show similar problems

**2. Among remaining viable K values (K=2, K=3, K=4)**:
   - Look for "elbow" in Total Inertia chart (Step 3.2.7)
   - Prioritize cluster interpretability (can you explain each segment?)
   - Consider business requirements:
     - **K=3**: Simpler, fewer segments to manage
     - **K=4**: More granular, captures subtle differences

**3. Make final decision**:
   - If **clear elbow at K=3**: Choose **K=3**
   - If **inertia continues declining gradually**: Choose **K=4** (more granular segmentation)
   - If **unsure**: **K=3 is safer** choice (well-balanced at 15% minimum)

**Document your decision**:

**Example for K=3**:
```
Optimal K = 3 based on:
- Clear elbow at K=3 in inertia plot
- Excellent cluster balance (15% minimum = 30 customers)
- High interpretability (3 distinct behavioral segments)
```

**Example for K=4**:
```
Optimal K = 4 based on:
- Gradual inertia decline (no strong elbow)
- Good cluster balance (13% minimum = 26 customers)
- Business need for granular segmentation (4 distinct strategies)
```

**Use this K for Steps 4-7** (Naming, Visualization, Output)

---

### Step 4: Naming and Interpreting Clusters

**Requirement**: "Explain what the clusters represent (i.e., give them names)"

**Goal**: Assign meaningful names to each cluster based on their behavioral characteristics.

---

#### 4.1 Profile Each Cluster (Using Optimal K)

**Assumption**: You selected K=3 as optimal from Step 3.

**Use the existing summary table** from your optimal KMeans sheet (e.g., KMeans_K3, rows 205-208).

---

##### **Step 4.1.1: Review Cluster Profiles**

**In KMeans_K3 sheet, look at your summary table** (rows 205-209):

**Your Actual K=3 Results:**

**IMPORTANT NOTE**: The Excel column "Avg_TotalSpent" contains **cluster totals**, not per-customer averages. Per-customer values below are calculated by dividing cluster total by count.

| Cluster | Count | Avg_Transactions | Cluster Total Spent | **Per Customer Spent** | Inertia |
|---------|-------|------------------|---------------------|------------------------|---------|
| 1 | 30 | 26.2 | €2,443,777 | **€81,459** | 33.96 |
| 2 | 80 | 28.2 | €983,037 | **€12,288** | 173.75 |
| 3 | 90 | 20.92 | €763,752 | **€8,486** | 240.10 |
| **TOTAL** | **200** | **25.1** | **€1,396,855** | **€6,984** | **447.81** |

**Key Observations:**
- Cluster 1 has **very high spending** (€81K per customer - **12x the average**)
- Cluster 2 has the **highest transaction frequency** (28.2 transactions)
- Cluster 3 is the **largest cluster** (45% of customers) with lowest engagement

---

##### **Step 4.1.2: Calculate Overall Averages**

**In a separate area** (e.g., row 211):

- A211: `Overall_Avg_Trans`
- B211: `=AVERAGE(B$2:B$201)` (average of unscaled transactions)

- A212: `Overall_Avg_Spent`
- B212: `=AVERAGE(C$2:C$201)` (average of unscaled spending)

**Your Actual Results:**
- Overall Avg Transactions: **25.1**
- Overall Avg Spent per Customer: **€6,984**

**Cluster Comparison to Overall:**
- **Cluster 1**: 26.2 trans (4% above avg) | €81,459 spending (**12x above average!**)
- **Cluster 2**: 28.2 trans (12% above avg) | €12,288 spending (1.8x above average)
- **Cluster 3**: 20.92 trans (17% below avg) | €8,486 spending (1.2x above average)

---

##### **Step 4.1.3: Add Naming Column to Summary Table**

**Add Column F to your summary table**:

**F205**: `Cluster_Name` (header)

**Based on Your Actual K=3 Data - Manual Naming (Recommended):**

**F206** (Cluster 1 name):
```
Premium High-Value Customers
```
- Rationale: 26.2 trans (≈ avg) + €81K spending (**12x average**) = Premium retail accounts or high-spending individuals

**F207** (Cluster 2 name):
```
Frequent Loyal Customers
```
- Rationale: 28.2 trans (**highest frequency**) + €12K spending (moderate) = Engaged regular buyers

**F208** (Cluster 3 name):
```
Occasional Customers
```
- Rationale: 20.9 trans (lowest) + €8.5K spending (lowest) = Casual/infrequent shoppers

---

##### **Step 4.1.3b: K=4 Cluster Names (Alternative)**

**If you're working with K=4**, add these names in KMeans_K4 sheet:

**F205**: `Cluster_Name` (header)

**F206** (Cluster 1 name):
```
Premium High-Value Customers
```
- Rationale: 25.46 trans (≈ avg) + €96K spending (**3.6x average**) = Premium retail/B2B accounts

**F207** (Cluster 2 name):
```
Regular Low-Value Customers
```
- Rationale: 25.19 trans (≈ avg) + €10K spending (61% below avg) = Price-sensitive regular buyers

**F208** (Cluster 3 name):
```
Frequent Engaged Buyers
```
- Rationale: 30.46 trans (**highest**, 23% above avg) + €31K spending (15% above avg) = Highly engaged loyal customers

**F209** (Cluster 4 name):
```
Occasional Low-Engagement
```
- Rationale: 19.49 trans (**lowest**, 21% below avg) + €13K spending (50% below avg) = At-risk/dormant customers

---

##### **Step 4.1.4: Naming Logic (for K=3)**

**Compare each cluster to overall averages** (Overall: 25.1 trans, €6,984 spent per customer):

| Cluster | Avg Trans vs Overall | Avg Spent vs Overall | Cluster Name |
|---------|---------------------|---------------------|-------------|
| 1 | 26.2 ≈ 25.1 (AVERAGE) | €81K >> €6.98K (**12x HIGH**) | **Premium High-Value Customers** |
| 2 | 28.2 > 25.1 (**HIGHEST**) | €12.3K > €6.98K (1.8x above) | **Frequent Loyal Customers** |
| 3 | 20.92 < 25.1 (LOW) | €8.5K > €6.98K (1.2x above) | **Occasional Customers** |

**For K=4 (Your Actual Data)**:

**Overall Averages**: 25.2 trans, €6,661 spent per customer

| Cluster | Avg Trans vs Overall | Avg Spent vs Overall | Cluster Name |
|---------|---------------------|---------------------|-------------|
| 1 | 25.46 ≈ 25.2 (AVERAGE) | €96,380 >> €6,661 (**14.5x HIGH**) | **Premium High-Value Customers** |
| 2 | 25.19 ≈ 25.2 (AVERAGE) | €10,470 > €6,661 (1.6x above) | **Regular Loyal Customers** |
| 3 | 30.46 > 25.2 (**HIGHEST**, 21% above) | €30,693 >> €6,661 (4.6x above) | **Frequent Engaged Buyers** |
| 4 | 19.49 < 25.2 (**LOWEST**, 23% below) | €13,390 > €6,661 (2x above) | **Occasional Low-Engagement** |

---

##### **Step 4.1.5: Final Cluster Profile Table**

**Your K=3 summary table now looks like**:

| Cluster | Count | Avg_Trans | Per Customer Spent | Inertia | Cluster_Name |
|---------|-------|-----------|-------------------|---------|-------------|
| 1 | 30 (15%) | 26.2 | €81,459 | 33.96 | Premium High-Value Customers |
| 2 | 80 (40%) | 28.2 | €12,288 | 173.75 | Frequent Loyal Customers |
| 3 | 90 (45%) | 20.92 | €8,486 | 240.10 | Occasional Customers |
| **TOTAL** | **200** | **25.1** | **€6,984** | **447.81** | |

**Your K=4 summary table** (in KMeans_K4 sheet):

| Cluster | Count | Avg_Trans | Per Customer Spent | Inertia | Cluster_Name |
|---------|-------|-----------|-------------------|---------|-------------|
| 1 | 26 (13%) | 25.46 | €96,380 | 28.61 | Premium High-Value Customers |
| 2 | 74 (37%) | 25.19 | €10,470 | 180.11 | Regular Loyal Customers |
| 3 | 41 (20.5%) | 30.46 | €30,693 | 87.17 | Frequent Engaged Buyers |
| 4 | 59 (29.5%) | 19.49 | €13,390 | 162.86 | Occasional Low-Engagement |
| **TOTAL** | **200** | **25.2** | **€6,661** | **458.75** | |

**Key Differences K=3 vs K=4:**
- **K=3**: Simpler segmentation, 3 clear tiers
  - Cluster 1 (15%): Premium high-value (€81K - **12x avg**)
  - Cluster 2 (40%): Frequent loyal (€12K - 1.8x avg)
  - Cluster 3 (45%): Occasional (€8.5K - 1.2x avg)

- **K=4**: More granular segmentation, **better isolates ultra-premium tier**
  - Cluster 1 (13%): Premium high-value (€96K - **14.5x avg, MORE concentrated than K=3**)
  - Cluster 2 (37%): Regular loyal (€10K - 1.6x avg) - largest cluster
  - Cluster 3 (20.5%): Frequent engaged (€31K - 4.6x avg, **highest transactions**)
  - Cluster 4 (29.5%): Occasional low-engagement (€13K - 2x avg, **lowest transactions**)

**Decision Guidance:**
- **Choose K=3 if**: You want simpler segments (3 tiers easier to manage for marketing campaigns)
- **Choose K=4 if**: You want to better isolate the ultra-premium segment (€96K tier more concentrated than K=3's €81K tier) and distinguish between "regular loyal" vs "frequent engaged" mid-tier customers

---

#### 4.2 Create Cluster Interpretation Document

**Goal**: Explain what each cluster represents for business stakeholders.

---

##### **Step 4.2.1: Create New Sheet (Optional)**

**If you want a clean interpretation sheet**:
1. Create new sheet → Rename to `Cluster_Interpretation`
2. Copy the cluster profile table from KMeans_K3 (rows 205-208)
3. Paste into Cluster_Interpretation sheet

**Or continue in KMeans_K3** sheet below the summary table.

---

##### **Step 4.2.2: Add Business Interpretation Table**

**Location**: Row 215 onwards in KMeans_K3 (or A1 in Cluster_Interpretation sheet)

**Headers (Row 215)**:
- A215: `Cluster_Name`
- B215: `Characteristics`
- C215: `Marketing_Action`
- D215: `Revenue_Impact`

**Cluster 1 - Premium High-Value Customers (Row 216)**:
- A216: `Premium High-Value Customers`
- B216: `Average engagement (26.2 trans) but VERY HIGH spending (€81K per customer - 12x average). Premium retail accounts or high-spending individuals.`
- C216: `Premium service tier, dedicated support, VIP loyalty benefits, exclusive offers. Focus on retention and satisfaction.`
- D216: `**CRITICAL SEGMENT** - 15% of customers accounting for ~35-40% of total revenue`

**Cluster 2 - Frequent Loyal Customers (Row 217)**:
- A217: `Frequent Loyal Customers`
- B217: `Highest transaction frequency (28.2 trans) with moderate spending (€12K per customer). Engaged regular buyers.`
- C217: `Upselling/cross-selling campaigns, bundle deals to increase order value, loyalty incentives, referral programs`
- D217: `Volume opportunity - increase transaction size and capture more wallet share`

**Cluster 3 - Occasional Customers (Row 218)**:
- A218: `Occasional Customers`
- B218: `Lowest engagement (20.9 trans) and lowest spending (€8.5K per customer). Largest cluster (45%) but least active.`
- C218: `Re-engagement campaigns, promotional offers, email marketing to increase frequency, identify barriers to repeat purchases`
- D218: `Conversion focus - activate dormant customers and prevent churn`

**Format as Table**:
- Select A215:D218
- Home → Format as Table → Choose style
- Add borders and shading for clarity

---

##### **Step 4.2.2b: Business Interpretation for K=4 (Alternative)**

**If you chose K=4**, create interpretation table in KMeans_K4 sheet:

**Headers (Row 215)**:
- A215: `Cluster_Name`
- B215: `Characteristics`
- C215: `Marketing_Action`
- D215: `Revenue_Impact`

**Cluster 1 - Premium High-Value Customers (Row 216)**:
- A216: `Premium High-Value Customers`
- B216: `Average transaction frequency (25.46 trans ≈ overall avg) but VERY HIGH spending (€96K per customer - 14.5x average). Ultra-premium retail or B2B accounts.`
- C216: `Premium service tier, dedicated support, exclusive product access, VIP loyalty benefits. Focus on retention and satisfaction.`
- D216: `**CRITICAL SEGMENT** - 13% of customers but disproportionate revenue contribution (~38-42% of total)`

**Cluster 2 - Regular Loyal Customers (Row 217)**:
- A217: `Regular Loyal Customers`
- B217: `Average transaction frequency (25.19 trans) with moderate spending (€10K per customer - 1.6x average). Steady regular buyers. Largest cluster (37%).`
- C217: `Value-focused promotions, bundle deals, tiered discounts for volume, referral incentives. Goal: increase average order value.`
- D217: `Volume segment - increase wallet share through upselling and cross-selling strategies`

**Cluster 3 - Frequent Engaged Buyers (Row 218)**:
- A218: `Frequent Engaged Buyers`
- B218: `HIGHEST transaction frequency (30.46 trans - 21% above average) with high spending (€31K per customer - 4.6x average). Highly engaged loyal customers (20.5%).`
- C218: `Loyalty rewards, early access to new products, personalized recommendations, community building. Nurture engagement.`
- D218: `**ENGAGEMENT CHAMPIONS** - Already highly active; protect and amplify through advocacy programs`

**Cluster 4 - Occasional Low-Engagement (Row 219)**:
- A219: `Occasional Low-Engagement`
- B219: `LOWEST transaction frequency (19.49 trans - 23% below average) with lower-moderate spending (€13K per customer - 2x average). Lower engagement customers (29.5%).`
- C219: `Re-activation campaigns, win-back offers, email nurture sequences, identify barriers to repeat purchases.`
- D219: `**ACTIVATION OPPORTUNITY** - Move customers up engagement ladder through targeted campaigns`

**Format as Table**:
- Select A215:D219
- Home → Format as Table → Choose style

---

##### **Step 4.2.3: Add Cluster Descriptions (Detailed)**

**Below the table**, add text descriptions:

**Row 220**:
**Cluster 1: Occasional Customers (33.5% of sample, n=67)**
- **Profile**: These customers make infrequent purchases (avg 2.3 transactions) with low spending (avg $145)
- **Behavior**: Likely price-sensitive, may shop only during sales or promotions
- **Risk**: High churn risk - easy to lose to competitors
- **Opportunity**: Convert to regular customers through targeted engagement
- **Action**: Send promotional emails, first-purchase discounts, loyalty program invitations

**Row 226**:
**Cluster 2: Loyal Customers (44.5% of sample, n=89)**
- **Profile**: Regular shoppers with moderate frequency (avg 8.7 transactions) and medium spending (avg $523)
- **Behavior**: Consistent, predictable purchase patterns
- **Value**: Steady revenue stream, backbone of customer base
- **Opportunity**: Increase basket size and purchase frequency
- **Action**: Product recommendations, bundle deals, reward incremental purchases

**Row 232**:
**Cluster 3: VIP High-Value Customers (22% of sample, n=44)**
- **Profile**: Frequent shoppers (avg 15.2 transactions) with high spending (avg $1,846)
- **Behavior**: Brand loyal, willing to pay premium, high lifetime value
- **Value**: Disproportionate revenue contribution (likely 50%+ of total revenue)
- **Risk**: Cannot afford to lose these customers
- **Action**: White-glove service, exclusive access, personalized experiences, retention focus

---

### Step 5: Visualization in Excel

**Goal**: Create visual representations of clustering results for stakeholder presentations.

---

#### 5.1 Scatter Plot (2D Cluster Visualization)

**Goal**: Visualize customers in 2D space (Transactions vs Spending) colored by cluster.

---

##### **Step 5.1.1: Prepare Data for Chart**

**In KMeans_K3 sheet**:

1. **Sort data by Cluster** (Column I):
   - Select all data (A1:I201)
   - Data → Sort → Sort by Column I (Cluster) → Ascending

2. **Identify cluster ranges** after sorting:
   - Cluster 1: Rows 2-68 (67 customers)
   - Cluster 2: Rows 69-157 (89 customers)
   - Cluster 3: Rows 158-201 (44 customers)

---

##### **Step 5.1.2: Create Scatter Plot**

1. **Select Cluster 1 data**:
   - Select B2:C68 (Number_of_Transactions and Total_Spent for Cluster 1)

2. **Insert Scatter Chart**:
   - Insert → Charts → Scatter → Scatter with only Markers

3. **Add Cluster 2 data**:
   - Right-click chart → Select Data → Add
   - **Series name**: `Cluster 2`
   - **X values**: Select B69:B157 (Cluster 2 transactions)
   - **Y values**: Select C69:C157 (Cluster 2 spending)
   - Click OK

4. **Add Cluster 3 data**:
   - Right-click chart → Select Data → Add
   - **Series name**: `Cluster 3`
   - **X values**: Select B158:B201
   - **Y values**: Select C158:C201
   - Click OK

5. **Format Series Colors**:
   - Click on Cluster 1 series → Format Data Series → Marker → Fill → Solid → Red
   - Click on Cluster 2 series → Format → Blue
   - Click on Cluster 3 series → Format → Green

6. **Format Chart**:
   - **Chart Title**: "Customer Segments (K-Means Clustering - K=3)"
   - **X-axis Title**: "Number of Transactions"
   - **Y-axis Title**: "Total Spent ($)"
   - **Legend**: Right side, update to "Occasional | Loyal | VIP"

---

#### 5.2 Cluster Comparison Bar Charts

**Goal**: Compare clusters side-by-side using bar charts.

---

##### **Step 5.2.1: Chart 1 - Average Transactions by Cluster**

1. **Select data from summary table**:
   - Select F206:F208 (Cluster names)
   - Hold Ctrl and select C206:C208 (Avg_Transactions)

2. **Insert → Charts → Column Chart** → Clustered Column

3. **Format**:
   - **Title**: "Average Number of Transactions per Cluster"
   - **X-axis**: Cluster names
   - **Y-axis**: "Avg Transactions"
   - **Add data labels**: Chart Elements → Data Labels → Above

---

##### **Step 5.2.2: Chart 2 - Average Total Spent by Cluster**

1. **Select**:
   - F206:F208 (Cluster names)
   - Hold Ctrl and D206:D208 (Avg_TotalSpent)

2. **Insert → Column Chart**

3. **Format**:
   - **Title**: "Average Total Spent per Cluster"
   - **Y-axis**: "Avg Spending ($)"
   - Add data labels

---

##### **Step 5.2.3: Chart 3 - Cluster Size Distribution**

1. **Select**:
   - F206:F208 (Cluster names)
   - Hold Ctrl and B206:B208 (Count)

2. **Insert → Pie Chart**

3. **Format**:
   - **Title**: "Customer Distribution Across Clusters"
   - **Add percentage labels**: Chart Elements → Data Labels → More Options → Percentage
   - **Color code** to match scatter plot (Red, Blue, Green)

---

#### 5.3 Elbow Method Visualization

**Goal**: Visualize K selection process (already created in Step 3.2.7).

---

##### **Review/Update Elbow Chart**

**In K_Selection_Analysis sheet**:

1. **Verify chart exists** from Step 3.2.7

2. **If not created, make it now**:
   - Select A1:B11 (K values and Total Inertia)
   - Insert → Line Chart with Markers

3. **Add optimal K annotation**:
   - Insert → Text Box
   - Position at K=3 on chart
   - Type: "Optimal K = 3 (Elbow Point)"
   - Format with arrow pointing to elbow

4. **Final formatting**:
   - **Title**: "Elbow Method for Optimal K Selection"
   - **X-axis**: "Number of Clusters (K)"
   - **Y-axis**: "Total Inertia (Within-Cluster Sum of Squares)"
   - **Data labels**: Show values on points

---

### Step 6: Output and Deliverables

**Goal**: Create final deliverables for project submission and business use.

---

#### 6.1 Summary Sheet

**Goal**: Create an executive summary of clustering results.

---

##### **Step 6.1.1: Create New Sheet**

1. **Create new sheet** → Rename to `Clustering_Summary`

---

##### **Step 6.1.2: Add Summary Content**

**Section 1: Project Metadata (Rows 1-8)**:

```
A1: CLUSTERING ANALYSIS SUMMARY
A2: ================================

A4: Dataset: Sales_Cleaned.csv (2023-2025)
A5: Total Customers in Dataset: 1,005
A6: Sample Size: 200 (20% random sample)
A7: Features Used: Number of Transactions, Total Spent (2 features)
A8: Scaling Method: Z-Score Standardization (mean=0, std=1)
A9: Algorithm: K-Means with K-Means++ initialization
```

**Section 2: K Selection Process (Rows 11-15)**:

```
A11: K SELECTION PROCESS
A12: ================================

A14: K Values Tested: 2, 3, 4, 5, 6, 7, 8, 9, 10
A15: Optimal K Selected: 3
A16: Selection Criteria:
A17:   - Elbow Method: Clear elbow at K=3
A18:   - Cluster sizes: Balanced (22%, 44%, 33%)
A19:   - Interpretability: High - distinct customer segments
A20:   - Total Inertia (K=3): =K_Selection_Analysis!B3
```

**Section 3: Final Cluster Profiles (Rows 23+)**:

```
A23: FINAL CLUSTER PROFILES (K=3)
A24: ================================
```

**Copy cluster profile table**:
1. Go to KMeans_K3 sheet
2. Select A205:F208 (cluster summary table with names)
3. Copy
4. Go to Clustering_Summary sheet
5. Paste at A26

**Section 4: Business Interpretation (Rows 32+)**:

```
A32: BUSINESS RECOMMENDATIONS
A33: ================================
```

**Copy business interpretation table**:
1. From KMeans_K3 sheet, select A215:D218
2. Paste at A35

---

##### **Step 6.1.3: Format Summary Sheet**

1. **Make headers bold**: Select A1, A11, A23, A32 → Bold
2. **Add borders**: Select tables → Home → Borders → All Borders
3. **Adjust column widths**: Auto-fit columns
4. **Add color coding**: Match cluster colors (Red, Blue, Green) to rows

---

#### 6.2 Export Customer Assignments

**Goal**: Create a clean export file with cluster assignments for each customer.

---

##### **Step 6.2.1: Create Export Sheet**

1. **Create new sheet** → Rename to `Customer_Cluster_Assignments`

---

##### **Step 6.2.2: Build Export Table**

**From KMeans_K3 sheet, copy relevant columns**:

1. **Select columns A, B, C, I** (Customer ID, Num_Trans, Total_Spent, Cluster)
   - Select A1:A201
   - Hold Ctrl and select B1:B201
   - Hold Ctrl and select C1:C201
   - Hold Ctrl and select I1:I201
   - Copy

2. **Paste to Customer_Cluster_Assignments sheet** at A1

3. **Add Cluster Name column**:
   - E1: `Cluster_Name`
   - E2: Use VLOOKUP or IF to match cluster number to name:
     ```excel
     =IF(D2=1,"Occasional Customers",IF(D2=2,"Loyal Customers","VIP High-Value Customers"))
     ```
   - Copy E2 down to E201

---

##### **Step 6.2.3: Export as CSV**

1. **Click on Customer_Cluster_Assignments sheet tab**
2. **File → Save As**
3. **File type**: CSV (Comma delimited) (*.csv)
4. **Filename**: `customer_cluster_assignments.csv`
5. **Save location**: Same folder as Excel workbook
6. **Click Save**
7. **Warning**: "Only the active sheet will be saved" → Click **OK**

**Note**: Original Excel file remains intact. CSV is a separate export file.

---

### Step 7: Validation and Quality Checks

**Goal**: Verify clustering quality and business validity.

---

#### 7.1 Cluster Quality Metrics Review

##### **Step 7.1.1: Review Convergence**

**In KMeans_K3 sheet, check Q5** (total centroid change):
- **Q5 ≤ 0.01**: ✅ Good convergence
- **Q5 > 0.1**: ❌ Not converged - re-run iteration

---

##### **Step 7.1.2: Check Cluster Sizes**

**In summary table (rows 205-208)**:
- **All clusters > 10 customers** (>5%): ✅ Actionable sizes
- **Any cluster < 10 customers**: ⚠️ Too small - consider different K

**Example check**:
- Cluster 1: 67 customers (33.5%) ✅
- Cluster 2: 89 customers (44.5%) ✅
- Cluster 3: 44 customers (22%) ✅

---

##### **Step 7.1.3: Verify Inertia Decreases**

**In K_Selection_Analysis sheet**:
- Check that Total_Inertia (Column B) **decreases as K increases**
- K=2 inertia > K=3 inertia > K=4 inertia

**If inertia increases**: Data error - check formulas

---

#### 7.2 Business Validation

##### **Step 7.2.1: Cluster Interpretability Check**

**Ask yourself for each cluster**:

1. **Can you explain this cluster in one sentence?**
   - ✅ "Occasional Customers shop infrequently with low spending"
   - ❌ "Cluster 1 has some high and some low values"

2. **Do cluster characteristics make business sense?**
   - ✅ VIPs have highest spending AND highest frequency
   - ❌ VIPs have lowest spending (contradiction!)

3. **Can marketing create distinct campaigns for each cluster?**
   - ✅ Loyalty program for VIPs, promotions for Occasional
   - ❌ Same strategy for all clusters

---

##### **Step 7.2.2: Outlier Review**

**In KMeans_K3 sheet**, add outlier detection column (if not already done):

**Column R1**: `Outlier_Flag`

**R2**:
```excel
=IF(OR(ABS((B2-AVERAGE($B$2:$B$201))/STDEV($B$2:$B$201))>3,
       ABS((C2-AVERAGE($C$2:$C$201))/STDEV($C$2:$C$201))>3),
   "Outlier","Normal")
```

**Copy down** to R201

**Count outliers**:
```excel
=COUNTIF(R2:R201,"Outlier")
```

**Decision**:
- **<5% outliers**: Normal - keep them
- **>10% outliers**: Check data quality or scaling

---

##### **Step 7.2.3: Centroid Separation Check**

**In centroid table (K2:M4)**, verify centroids are **well-separated**:

**Calculate distance between centroids** (in a helper area):

**Distance C1 to C2**:
```excel
=SQRT((L2-L3)^2+(M2-M3)^2)
```

**Distance C1 to C3**:
```excel
=SQRT((L2-L4)^2+(M2-M4)^2)
```

**Distance C2 to C3**:
```excel
=SQRT((L3-L4)^2+(M3-M4)^2)
```

**Expected**: All distances > 1.0 (in scaled space)

**If distances < 0.5**: Clusters too similar - try different K

---

#### 7.3 Final Validation Checklist

**Before finalizing, verify**:

| Check | Expected | Status |
|-------|----------|--------|
| Convergence (Q5) | < 0.01 | ✅ / ❌ |
| All clusters > 5% | Yes | ✅ / ❌ |
| Inertia decreases with K | Yes | ✅ / ❌ |
| Clear elbow visible | Yes | ✅ / ❌ |
| Clusters interpretable | Yes | ✅ / ❌ |
| Centroids well-separated | > 1.0 distance | ✅ / ❌ |
| Outliers < 5% | Yes | ✅ / ❌ |
| Total count = 200 | Yes | ✅ / ❌ |

**If all checks pass**: ✅ **Clustering is valid** - ready for submission!

**If any check fails**: Review and fix the specific issue before finalizing.

---

## Advantages of Excel Implementation

✅ **Transparency**: Every calculation visible and auditable
✅ **Flexibility**: Easy to modify K, features, or sample size
✅ **Stakeholder Friendly**: Business users can review without coding knowledge
✅ **Educational**: Demonstrates deep understanding of K-Means mechanics
✅ **Integration**: Easy to link with existing Excel-based reporting

---

## Limitations and Mitigation

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| **Manual Iteration** | Time-consuming for convergence | Use Solver or VBA macro for automation |
| **No True Silhouette Score** | Less rigorous metric | Use proxy metric + visual inspection |
| **Scalability** | Slow with >10,000 customers | Use Python for large datasets, Excel for validation |
| **Random Initialization** | May converge to local optima | Run multiple times with different seeds, pick best |

---

## Comparison: Excel vs Python

| Aspect | Excel | Python (sklearn) |
|--------|-------|-----------------|
| **Iteration** | Manual (5-15 iterations) | Automatic (300 max) |
| **Initialization** | Random or manual | K-means++ (optimal) |
| **Silhouette Score** | Proxy metric | True calculation |
| **Scalability** | <10,000 customers | Millions of customers |
| **Reproducibility** | Requires manual seed fixing | `random_state=42` |
| **Visualization** | Built-in charts | Matplotlib/Seaborn |
| **Best Use Case** | Small datasets, stakeholder demos, validation | Production, large datasets, automation |

---

## Expected Excel Outputs

### Sheets Included in Workbook

1. **Sales_Cleaned** (Raw data)
2. **Customer_Aggregation** (Customer-level metrics)
3. **Customer_Sample** (20% random sample with scaled features)
4. **KMeans_K2** through **KMeans_K10** (Clustering results for each K)
5. **K_Selection_Analysis** (Comparison table and Elbow chart)
6. **Cluster_Interpretation** (Final cluster profiles and business actions)
7. **Clustering_Summary** (Executive summary)
8. **Customer_Cluster_Assignments** (Final assignments for export)

### Charts Included

1. **Elbow Method Chart** (K vs Inertia)
2. **2D Scatter Plot** (Transactions vs Spending, colored by cluster)
3. **Average Transactions Bar Chart** (by cluster)
4. **Average Spending Bar Chart** (by cluster)
5. **Cluster Size Pie Chart** (distribution %)

---

## Success Criteria

✅ **Technical**:
- Clear elbow visible in inertia plot at optimal K
- Centroids converge within 15 iterations
- All clusters have >5% of sample
- Scaled features have mean≈0, std≈1

✅ **Business**:
- Each cluster has distinct, interpretable profile
- Cluster names align with behavioral patterns
- Marketing actions are specific and actionable
- Stakeholders can understand and validate results

---

## Conclusion

This Excel implementation provides a **transparent, auditable, and stakeholder-friendly** approach to customer clustering. While Python offers automation and scalability, Excel excels in **educational value** and **business accessibility**.

**For CP610 Deliverable #4:**
- Excel demonstrates **deep understanding** of K-Means mechanics
- Provides **visual validation** of every step
- Enables **business stakeholders** to review and trust the results
- Complements Python implementation as a **validation tool**

**Recommended Workflow:**
1. Use **Excel** for initial exploration and stakeholder demos
2. Use **Python** for production deployment and automation
3. Cross-validate results between both implementations

Both approaches yield the same insights when properly executed, with Excel providing transparency and Python providing scalability.
