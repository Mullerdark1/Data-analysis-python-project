import pandas as pd

df = pd.read_csv("Fos_students.csv")

#last 5 rows
print(df.tail())

#print shapes
print(df.shape)

#quick stats
print(df.describe())

#check for missing values
print(df.isnull().sum())

#count students according to department
print(df["Department"].value_counts())

#check for maximum CGPA in Faculty of science
print(df["CGPA"].max())