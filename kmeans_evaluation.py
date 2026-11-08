
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the RFM table
rfm = pd.read_csv("outputs/rfm_table.csv")

# Use the original RFM columns (no log transformation)
features = rfm[["recency", "frequency", "monetary"]]

# Standardise the three RFM features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Evaluate K from 2 to 8
results = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    results.append({
        "k": k,
        "inertia": round(inertia, 4),
        "silhouette_score": round(silhouette, 4)
    })

# Display and save results
results_df = pd.DataFrame(results)

print("\nK-means evaluation results:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/kmeans_evaluation.csv",
    index=False
)

print("\nSaved to outputs/kmeans_evaluation.csv")
