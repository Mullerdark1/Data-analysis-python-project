import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Simulated dataset: Hours Studied vs Pass/Fail
hours = np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1)
results = np.array([0,0,0,0,1,1,1,1,1,1])  # 0 = Fail, 1 = Pass

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(hours, results, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities and classes
pred_probs = model.predict_proba(X_test)
pred_classes = model.predict(X_test)

print("Predicted probabilities:\n", pred_probs)
print("Predicted classes:", pred_classes)
print("Actual classes:", y_test)

# Plot
plt.scatter(hours, results, color="blue", label="Data")
plt.plot(hours, model.predict_proba(hours)[:,1], color="red", label="Pass Probability")
plt.xlabel("Hours Studied")
plt.ylabel("Pass Probability")
plt.title("Logistic Regression - Classification")
plt.legend()
plt.show()
