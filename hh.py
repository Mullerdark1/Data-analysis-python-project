import pandas as pd

data = {
    "Name" : ['Falahat', 'Kikelomo','Yemisi','Darasimi','Olamilekan','Olamilekan'],
    "Department" : ['Statistics', 'Statistics','Physics','Comp. Sci','Statistics','Statistics'],
    "Age" : [23, 21, 22, 19, 26, 26],
    "Score": [90, 85, 88, 92,90, 90]
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv("Fos_students.csv", index=False)

print("✅ Data saved to Fos_dtudents.csv")