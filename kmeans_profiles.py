
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load RFM table
rfm = pd.read_csv("outputs/rfm_table.csv")

# Use the original RFM columns
features = rfm[["recency", "frequency", "monetary"]]

# Standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# Fit K-means with the selected K=4
kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = kmeans.fit_predict(X_scaled)

# Profile the segments
profiles = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    )
)

# Print complete profile table
print("\nK=4 Customer Segment Profiles:")
print(profiles.round(2).to_string())

# Business names based on the observed cluster profiles
segment_names = {
    0: "Loyal Active Customers",
    1: "Developing Customers",
    2: "Premium Customers",
    3: "Low-Engagement Customers"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Premium = segment with the highest mean monetary value
premium_segment = profiles["mean_monetary"].idxmax()

rfm["is_premium"] = (
    rfm["segment_id"] == premium_segment
)

# Select required output columns
customer_segments = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

# Save output
customer_segments.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

print("\nPremium segment ID:", premium_segment)

print("\nSegment names:")
print(
    rfm.groupby("segment_id")["segment_name"]
    .first()
    .to_string()
)

print("\nSaved outputs/customer_segments.csv")

# PCA on the scaled RFM matrix
pca = PCA(n_components=2)
pca.fit(X_scaled)

print("\nPCA explained variance ratio:")
print(pca.explained_variance_ratio_.round(4))

print(
    "Total explained variance:",
    round(pca.explained_variance_ratio_.sum(), 4)
)
