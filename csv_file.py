import pandas as pd
df = pd.read_csv("Fos_students.csv")

#give basic statistics value
print(df.describe())

#select Age column or a single column only
print(df["Age"])

#print multiple columns
print(df[["Name", "Department"]])

#print first 5 rows
print(df.iloc[0:5])

#select students with age less than 25 and score greater than 85
print(df[(df["Age"]<25)& (df["Score"]>85)])

#check for missing values
print(df.isnull())

#Drop for rows with missing value
print(df.dropna())

#fill missing values with 0 instead of NaN
print(df.fillna(0))

#checking for duplicates
print(df.duplicated())

#remove duplicate 
df_no_dup = df.drop_duplicates()
print(df_no_dup)

#rename columns
df = df.rename(columns= {"Score" : "Exam_Score"})
print(df.head())

#covert age column to string
df["Age"] = df["Age"].astype(str)
print(df.dtypes)