import pandas as pd
import matplotlib.pyplot as plt

#Create date range
dates = pd.date_range("2025-09-01", periods=10, freq="D")

data = {
    "Date": dates,
    "Milo": [10,20,30,40,50,60,70,80,90,100],
    "Dano": [5,10,15,20,25,30,35,40,45,50]
}
df = pd.DataFrame(data)
print(df)

#set date as index
df.set_index("Date", inplace=True)

#plot daily sales
df.plot(title="Daily sales for products", marker="o")
plt.ylabel("Date")
plt.xlabel("Sales")
plt.show()

#Resample
#weekly_sales = df.resample("W").mean()
#print(weekly_sales)