import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

#dataset for customer purchase predictions
data = {
    "Age": [22, 25, 47, 52, 46, 56, 55, 60],
    "Income": [20000, 25000, 50000, 60000, 52000, 80000, 75000, 90000],
    "Buy_Product": [0, 0, 1, 1, 1, 1, 1, 1]  # 0 = No, 1 = Yes
}

df = pd.DataFrame(data)

# Features and target
X = df[["Age", "Income"]]
y = df["Buy_Product"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
