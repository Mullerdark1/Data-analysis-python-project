import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Sample commercial dataset (customer data)
data = {
    "Age": [25, 30, 45, 22, 23, 36, 50, 48],
    "Annual_Income": [20000, 35000, 60000, 15000, 18000, 40000, 70000, 65000],
    "Spending_Score": [40, 60, 75, 20, 30, 65, 90, 85]
}

df = pd.DataFrame(data)

# Standardize the features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("PCA Result:\n", pca_result)
