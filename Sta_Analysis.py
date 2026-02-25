import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Sta 264" : [88, 72, 75, 90, 92],
    "Sta 268" : [80, 90, 70, 75, 77],
    "Ent 211" : [82, 85, 88, 90, 92]
}
df = pd.DataFrame(data)

#print(df)

#Descriptive statistics
#print(df.describe())

#correlation
#print(df.corr())

#covariance
#print(df.cov())

#Variance and Standard Deviation
print("Sta 268 Variance:",df["Sta 268"].var())
print("Sta 268 STD:",df["Sta 268"].std())

#Custom range
#print(df.apply(lambda x: (x.max() - x.min())))