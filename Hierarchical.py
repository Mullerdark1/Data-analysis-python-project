import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Sample dataset
data = {
    "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8],
    "Annual_Income": [15, 16, 17, 25, 26, 60, 62, 63],
    "Spending_Score": [39, 81, 6, 77, 40, 50, 49, 55]
}
df = pd.DataFrame(data)

# Create linkage matrix
Z = linkage(df[["Annual_Income", "Spending_Score"]], method='ward')

# Plot dendrogram
plt.figure(figsize=(6, 4))
dendrogram(Z, labels=df["CustomerID"].values)
plt.title("Customer Dendrogram")
plt.xlabel("Customer ID")
plt.ylabel("Distance")
plt.show()

# Cut tree into 2 clusters
clusters = fcluster(Z, 2, criterion="maxclust")
df["Cluster"] = clusters
print(df)
