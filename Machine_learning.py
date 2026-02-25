import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Simulated dataset: Study Hours vs Exam Score
hours = np.array([1,2,3,4,5,6,7,8,9]).reshape(-1,1)
scores = np.array([35, 50, 55, 65, 70, 75, 80, 88, 95])

# Split into training and test
X_train, X_test, y_train, y_test = train_test_split(hours, scores, test_size=0.2, random_state=42)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Print results
print("Predicted:", predictions)
print("Actual:", y_test)

# Plot
plt.scatter(hours, scores, color="blue", label="Data")
plt.plot(hours, model.predict(hours), color="red", label="Best Fit Line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Simple Linear Regression with Scikit-Learn")
plt.legend()
plt.show()
