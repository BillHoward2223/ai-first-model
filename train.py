
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Each row contains:
# [CPU usage percentage, RAM usage percentage]
X = [
    [10, 20], [15, 25], [20, 30], [25, 35],
    [30, 40], [35, 45], [40, 50], [45, 55],
    [50, 60], [55, 65], [60, 70], [65, 75],
    [70, 80], [75, 85], [80, 90], [90, 95],
    [20, 80], [30, 85], [40, 90], [50, 95],
    [85, 30], [90, 40], [95, 50], [80, 45],
]

# Labels: 0 = normal, 1 = overloaded
y = [
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    1, 1, 1, 1,
    0, 0, 0, 0,
    1, 1, 1, 1,
]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create and train the AI model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Evaluate on examples not used for training
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")
print(f"Test accuracy: {accuracy:.2%}")

# Save the trained model
joblib.dump(model, "server_model.joblib")
print("Saved model to server_model.joblib")
