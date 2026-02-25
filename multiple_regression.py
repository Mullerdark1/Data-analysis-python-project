import pandas as pd
import statsmodels.api as sm

# Example dataset
data = {
    "Hours_Sleep": [5, 6, 7, 8, 6, 7, 8, 9],
    "Hours_Study": [2, 3, 4, 5, 3, 4, 5, 6],
    "Productivity_Score": [60, 65, 72, 80, 68, 74, 82, 88]
}

df = pd.DataFrame(data)

# Define dependent (Y) and independent variables (X)
X = df[["Hours_Sleep", "Hours_Study"]]
y = df["Productivity_Score"]

# Add constant (intercept)
X = sm.add_constant(X)

# Fit regression model
model = sm.OLS(y, X).fit()

# Show results
print(model.summary())           