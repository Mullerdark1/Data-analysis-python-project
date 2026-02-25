import pandas as pd
import statsmodels.api as sm

# Sample dataset
data = {
    "Hours_Study": [2, 3, 4, 5, 6, 7, 7, 8, 8, 9, 10, 10, 11, 12, 13],
    "Attendance":  [55, 60, 58, 65, 68, 70, 60, 75, 55, 80, 85, 65, 88, 92, 97],
    "Pass":        [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Define X and y
X = df[["Hours_Study", "Attendance"]]
y = df["Pass"]

# Add constant
X = sm.add_constant(X)

# Logistic Regression
model = sm.Logit(y, X).fit()

print(model.summary())