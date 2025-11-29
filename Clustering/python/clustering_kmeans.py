import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# Suppress runtime warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def load_sales_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=';', decimal=',')
    # Convert Date column to datetime format for proper date handling
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
    # Filter only years 2023-2025 to match project scope
    df = df[df['Year'].between(2023, 2025)]
    return df


def aggregate_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    # Group by Customer ID and calculate the two required metrics
    customer_agg = df.groupby('Customer ID').agg({
        # Count of transactions per customer (Number of Transactions)
        'Transaction ID': 'count',
        # Sum of all spending per customer (Total Spent)
        'Total Spent': 'sum',
    }).reset_index()

    # Rename columns for clarity
    customer_agg.rename(columns={
        'Transaction ID': 'Number_of_Transactions',
        'Total Spent': 'Total_Spent',
    }, inplace=True)

    return customer_agg


def sample_customers(customer_df: pd.DataFrame, sample_pct=0.20, random_seed=42) -> pd.DataFrame:
    # Calculate sample size (20% of total customers)
    sample_size = int(len(customer_df) * sample_pct)

    # Randomly sample customers with fixed seed for reproducibility
    sampled_customers = customer_df.sample(n=sample_size, random_state=random_seed)

    return sampled_customers


def prepare_features_for_clustering(customer_df: pd.DataFrame) -> tuple:
    # Extract only the two required features as per PDF requirements
    feature_cols = ['Number_of_Transactions', 'Total_Spent']
    X = customer_df[feature_cols].copy()

    # Check for any invalid values (NaN, inf) and replace with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill any NaN values with column median (robust to outliers)
    X = X.fillna(X.median())

    # Standardize features (mean=0, std=1) - essential for K-Means distance calculations
    # This ensures both features contribute equally to clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Clip extreme values after scaling to prevent overflow in distance calculations
    # Keep values within reasonable range (-10 to 10 standard deviations)
    X_scaled = np.clip(X_scaled, -10, 10)

    return X_scaled, feature_cols, scaler


def find_optimal_k(X_scaled, k_range=range(2, 11)) -> tuple:
    # Initialize lists to store evaluation metrics for each K
    inertias = []           # Within-cluster sum of squares (Elbow method)
    silhouette_scores = []  # Cluster quality metric (higher is better)
    davies_bouldin_scores = []  # Cluster separation metric (lower is better)

    # Test different values of K (number of clusters)
    for k in k_range:
        # Initialize K-Means clustering algorithm
        kmeans = KMeans(
            n_clusters=k,           # Number of clusters to form
            init='k-means++',       # Smart initialization method
            n_init=10,              # Run algorithm 10 times with different seeds
            max_iter=300,           # Maximum iterations per run
            random_state=42,        # For reproducibility
        )

        # Fit the model and get cluster assignments
        labels = kmeans.fit_predict(X_scaled)

        # Store inertia (sum of squared distances to nearest cluster center)
        inertias.append(kmeans.inertia_)

        # Calculate and store silhouette score (measures cluster cohesion and separation)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

        # Calculate and store Davies-Bouldin index (ratio of within-cluster to between-cluster distances)
        davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels))

    # Determine optimal K using silhouette score (highest value indicates best clustering)
    optimal_k_idx = np.argmax(silhouette_scores)
    optimal_k = list(k_range)[optimal_k_idx]

    return optimal_k, inertias, silhouette_scores, davies_bouldin_scores, k_range


def fit_final_kmeans(X_scaled, optimal_k: int) -> tuple:
    # Initialize K-Means with optimal K
    kmeans = KMeans(
        n_clusters=optimal_k,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42,
    )

    # Fit model and get cluster labels for each customer
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Calculate final evaluation metrics
    silhouette = silhouette_score(X_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)

    return kmeans, cluster_labels, silhouette, davies_bouldin


def create_cluster_profiles(customer_df: pd.DataFrame, cluster_labels, scaler, feature_cols) -> pd.DataFrame:
    customer_df['Cluster'] = cluster_labels

    profiles = customer_df.groupby('Cluster').agg({
        'Customer ID': 'count',
        'Number_of_Transactions': ['mean', 'median', 'min', 'max'],
        'Total_Spent': ['mean', 'median', 'min', 'max'],
    }).round(2)

    profiles.columns = ['_'.join(col).strip('_') for col in profiles.columns.values]
    profiles.rename(columns={'Customer ID_count': 'Cluster_Size'}, inplace=True)

    cluster_names = assign_cluster_names(profiles)
    profiles['Cluster_Name'] = cluster_names

    return profiles


def assign_cluster_names(profiles: pd.DataFrame) -> list:
    cluster_names = []

    overall_avg_transactions = profiles['Number_of_Transactions_mean'].mean()
    overall_avg_spent = profiles['Total_Spent_mean'].mean()

    # Iterate through each cluster and assign name based on characteristics
    for idx in profiles.index:
        avg_transactions = profiles.loc[idx, 'Number_of_Transactions_mean']
        avg_spent = profiles.loc[idx, 'Total_Spent_mean']

        # High spending, high frequency
        if avg_spent > overall_avg_spent and avg_transactions > overall_avg_transactions:
            cluster_names.append('High Value Customers')
        # High spending, low frequency
        elif avg_spent > overall_avg_spent and avg_transactions <= overall_avg_transactions:
            cluster_names.append('Big Spenders')
        # Low spending, high frequency
        elif avg_spent <= overall_avg_spent and avg_transactions > overall_avg_transactions:
            cluster_names.append('Loyal Customers')
        # Low spending, low frequency
        else:
            cluster_names.append('Occasional Customers')

    return cluster_names


def visualize_elbow_silhouette(k_range, inertias, silhouette_scores, davies_bouldin_scores,
                                optimal_k, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(list(k_range), inertias, marker='o', linewidth=2, markersize=8)
    axes[0].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
    axes[0].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[0].set_ylabel('Inertia (Within-Cluster SS)', fontsize=12)
    axes[0].set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouette_scores, marker='o', color='green', linewidth=2, markersize=8)
    axes[1].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
    axes[1].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score Analysis', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(list(k_range), davies_bouldin_scores, marker='o', color='orange', linewidth=2, markersize=8)
    axes[2].axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal K={optimal_k}')
    axes[2].set_xlabel('Number of Clusters (K)', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
    axes[2].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'k_selection_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()


def visualize_clusters_scatter(customer_df: pd.DataFrame, optimal_k, out_dir: Path):
    plt.figure(figsize=(12, 8))

    for cluster_id in range(optimal_k):
        cluster_data = customer_df[customer_df['Cluster'] == cluster_id]
        plt.scatter(
            cluster_data['Number_of_Transactions'],
            cluster_data['Total_Spent'],
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            s=100,
            edgecolors='k',
            linewidths=0.5,
        )

    plt.xlabel('Number of Transactions', fontsize=12)
    plt.ylabel('Total Spent ($)', fontsize=12)
    plt.title(f'Customer Clusters (K={optimal_k})\nBased on Transaction Count and Total Spending',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'clusters_scatter_plot.png', dpi=200, bbox_inches='tight')
    plt.close()


def visualize_cluster_comparison(profiles: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cluster_labels = [f"C{i}" for i in profiles.index]

    axes[0].bar(cluster_labels, profiles['Number_of_Transactions_mean'],
                color='skyblue', edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Cluster', fontsize=12)
    axes[0].set_ylabel('Avg Number of Transactions', fontsize=12)
    axes[0].set_title('Average Transactions per Cluster', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Average Total Spent by Cluster
    axes[1].bar(cluster_labels, profiles['Total_Spent_mean'],
                color='lightcoral', edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('Cluster', fontsize=12)
    axes[1].set_ylabel('Avg Total Spent ($)', fontsize=12)
    axes[1].set_title('Average Spending per Cluster', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig(out_dir / 'cluster_comparison_bars.png', dpi=200, bbox_inches='tight')
    plt.close()


def save_outputs(customer_df: pd.DataFrame, profiles: pd.DataFrame, optimal_k: int,
                 silhouette: float, davies_bouldin: float, k_results: dict, out_dir: Path):
    # Create output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prepare customer output with cluster assignments
    customer_output = customer_df[['Customer ID', 'Number_of_Transactions', 'Total_Spent', 'Cluster']].copy()

    csv_path = out_dir / 'customer_cluster_assignments.csv'
    try:
        customer_output.to_csv(csv_path, index=False)
        print(f"Saved cluster to {csv_path}")
    except PermissionError:
        alt_path = out_dir / 'customer_cluster_out.csv'
        print(f"Could not write {csv_path} (file may be open). Saved to {alt_path}.")
        customer_output.to_csv(alt_path, index=False)

    profile_path = out_dir / 'cluster_profiles.csv'
    try:
        profiles.to_csv(profile_path)
        print(f"Saved cluster profiles to {profile_path}")
    except PermissionError:
        alt_path = out_dir / 'cluster_profiles_out.csv'
        print(f"Could not write {profile_path} (file may be open). Saved to {alt_path}.")
        profiles.to_csv(alt_path)

def main():
    """Execute the complete clustering pipeline as per project requirements."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    csv_path = base_dir / 'datasets' / 'Sales_Cleaned.csv'
    out_dir = base_dir / 'Clustering' / 'python'

    df = load_sales_data(csv_path)

    customer_df = aggregate_customer_data(df)

    sampled_df = sample_customers(customer_df, sample_pct=0.20, random_seed=42)

    X_scaled, feature_cols, scaler = prepare_features_for_clustering(sampled_df)


    print("EXPERIMENTING WITH DIFFERENT K VALUES (K=2 to K=10)")
    optimal_k, inertias, silhouette_scores, davies_bouldin_scores, k_range = find_optimal_k(X_scaled)

    print("\nResults for all K values tested:")
    for i, k in enumerate(k_range):
        print(f"{k:<5} {inertias[i]:<15.2f} {silhouette_scores[i]:<15.4f} {davies_bouldin_scores[i]:<15.4f}")

    print(f"BEST K SELECTED: {optimal_k}")
    print(f"Rationale: K={optimal_k} was chosen because it has the HIGHEST Silhouette Score")
    print(f"Silhouette Score: {silhouette_scores[optimal_k - 2]:.4f} (higher is better)")
    print(f"Davies-Bouldin Index: {davies_bouldin_scores[optimal_k - 2]:.4f} (lower is better)")

    kmeans, cluster_labels, silhouette, davies_bouldin = fit_final_kmeans(X_scaled, optimal_k)

    print("CLUSTER INTERPRETATIONS - EXPLAINING WHAT EACH CLUSTER REPRESENTS")
    profiles = create_cluster_profiles(sampled_df, cluster_labels, scaler, feature_cols)

    print("\nCluster Profiles with Names and Explanations:\n")
    for idx in profiles.index:
        size = int(profiles.loc[idx, 'Cluster_Size'])
        pct = (size / len(sampled_df)) * 100
        name = profiles.loc[idx, 'Cluster_Name']
        avg_trans = profiles.loc[idx, 'Number_of_Transactions_mean']
        avg_spent = profiles.loc[idx, 'Total_Spent_mean']

        print(f"Cluster {idx}: {name}")
        print(f"  Size: {size:,} customers ({pct:.1f}% of sample)")
        print(f"  Avg Transactions: {avg_trans:.1f}")
        print(f"  Avg Total Spent: ${avg_spent:,.2f}")

    visualize_elbow_silhouette(k_range, inertias, silhouette_scores, davies_bouldin_scores, optimal_k, out_dir)

    visualize_clusters_scatter(sampled_df, optimal_k, out_dir)


    visualize_cluster_comparison(profiles, out_dir)

    k_results = {
        'k_range': k_range,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores
    }
    save_outputs(sampled_df, profiles, optimal_k, silhouette, davies_bouldin, k_results, out_dir)


if __name__ == '__main__':
    main()
