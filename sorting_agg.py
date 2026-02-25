import pandas as pd

df = pd.read_csv("Fos_students.csv")

#sort by CGPA (ASCENDING ORDER)
#print(df.sort_values("CGPA"))

#Sort CGPA (descending order)
#print(df.sort_values("CGPA", ascending=False)) 

#Average CGPA
#print("Average CGPA:",df["CGPA"].mean())

#Maximum and Minimum CGPA
#print("Maximum CGPA:",df["CGPA"].max()),
#print("Minimum CGPA:",df["CGPA"].min())

#Count number of students
#print("Number of Students:",df["CGPA"].count())

#Average CGPA per department 
print(df.groupby("Department")["CGPA"].mean())

#Number of students per department
print(df.groupby("Department")["Name"].count())