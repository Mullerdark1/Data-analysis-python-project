import pandas as pd
import statsmodels.api as sm

#Dataset of hours of sleep vs productivity
data = {
    "Hours_Sleep": [4, 5, 6, 7, 8, 9, 10],
    "Productivity_Score": [50, 55, 60, 68, 72, 78, 85]
}

df = pd.DataFrame(data)

#Define Independent variable (X) and dependent variable (y)
X = df["Hours_Sleep"]
y = df["Productivity_Score"]

# Add constant for intercept
X = sm.add_constant(X)

# Fit model
model = sm.OLS(y, X).fit()

# Show summary
print(model.summary())

