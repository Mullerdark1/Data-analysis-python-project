import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Dataset
data = {
    'Hours_Study': [2, 4, 5, 6, 8, 9, 10, 12],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95],
    'Pass':       [0, 0, 0, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}
df = pd.DataFrame(data)

# Features and target
X = df[['Hours_Study', 'Attendance']]
y = df['Pass']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Accuracy
accuracy = tree.score(X_test, y_test)
print("Decision Tree Accuracy:", accuracy)

# Visualization
plt.figure(figsize=(8,6))
plot_tree(tree, feature_names=['Hours_Study', 'Attendance'], class_names=['Fail', 'Pass'], filled=True)
plt.show()
