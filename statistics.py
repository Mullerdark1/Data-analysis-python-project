import pandas as pd
print("Pandas version:", pd.__version__)

#create a series
data = [10, 20, 30, 40, 50]

s = pd.Series(data)
print("Series:\n", s)

#create a DataFrame
data = {
    "Name" : ["John", "Mary", "Olamilekan"],
    "Age" :[22, 25, 26],
    "Score" : [85, 90, 88]
}
df = pd.DataFrame(data)
print("DataFrame:\n",df)

print("Column Names:",df.columns)
print("First 2 rows\n", df.head(2))
print("Age Column:\n", df["Age"])
print("Average Age:", df["Age"].mean())