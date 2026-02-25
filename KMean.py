import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Sample dataset (Age, Income, Spending)
data = {
    "Age": [25, 34, 45, 23, 35, 52, 46, 50],
    "Annual_Income": [40000, 60000, 80000, 30000, 50000, 90000, 75000, 85000],
    "Spending_Score": [60, 70, 40, 90, 65, 30, 45, 35]
}
df = pd.DataFrame(data)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply K-Means (K=2 clusters)
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

print("Cluster Centers (Scaled):\n", kmeans.cluster_centers_)
print("\nClustered Data:\n", df)
