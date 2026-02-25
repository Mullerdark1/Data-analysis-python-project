import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Fos_students.csv")

#Bar chart 
df.plot(kind="bar", x="Department", y="Score", color="skyblue")
plt.title("Number of Students in Each Department")
plt.ylabel("Score of students")
plt.xlabel("Departments")
plt.show()

#line chart
df.plot(kind="line", x="Name", y="Age", marker="o")
plt.title("Age per Students")
plt.ylabel("Age")
plt.xlabel("Departments")
plt.show()

#Pie Chart
df.set_index("Department")["Score"].plot(kind="pie", autopct="%1.1f%%")
plt.title("Student Distribution by Department")
plt.ylabel("")  # Hide y-axis
plt.show()
