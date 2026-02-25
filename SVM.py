import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset
data = {
    "Hours_Study": [2, 4, 6, 8, 10, 1, 7, 5],
    "Attendance": [60, 70, 80, 90, 95, 50, 85, 75],
    "Pass": [0, 0, 1, 1, 1, 0, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

# Features and target
X = df[["Hours_Study", "Attendance"]]
y = df["Pass"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train SVM classifier
model = SVC(kernel="linear")   # You can also try "rbf" or "poly"
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
