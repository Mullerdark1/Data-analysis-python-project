import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Sample dataset
data = {
    "Hours_Study": [2, 4, 6, 8, 10, 1, 7, 5],
    "Attendance": [60, 70, 80, 90, 95, 50, 85, 75],
    "Pass": [0, 0, 1, 1, 1, 0, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

X = df[["Hours_Study", "Attendance"]]
y = df["Pass"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Build neural network
model = Sequential([
    Dense(8, activation="relu", input_shape=(2,)),
    Dense(4, activation="relu"),
    Dense(1, activation="sigmoid")  # Output: probability of passing
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=20, verbose=0)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy:", accuracy)
