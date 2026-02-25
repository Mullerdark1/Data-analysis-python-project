import pandas as pd
from sklearn.ensemble import IsolationForest

# Sample dataset (transaction amounts in $)
data = {
    "TransactionID": range(1, 11),
    "Amount": [50, 60, 55, 52, 58, 5000, 53, 59, 61, 45]  # 5000 is suspicious
}

df = pd.DataFrame(data)

# Train Isolation Forest
model = IsolationForest(contamination=0.1, random_state=42)
df["Anomaly"] = model.fit_predict(df[["Amount"]])

# -1 = anomaly, 1 = normal
print(df)
