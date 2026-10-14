
import csv
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

DATA_FILE = "training_data.csv"
MODEL_FILE = "server_model.joblib"

# 1. Load the entire dataset
X = []
y = []

with open(DATA_FILE, newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        X.append([
            float(row["cpu_pct"]),
            float(row["ram_pct"]),
        ])
        y.append(int(row["overloaded"]))

if len(X) < 8:
    raise ValueError("Collect more examples before training.")

if len(set(y)) != 2:
    raise ValueError("Training data must contain both labels: 0 and 1.")

# 2. Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

# 3. Train a fresh model on the combined dataset
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
)

model.fit(X_train, y_train)

# 4. Evaluate the model on held-out examples
predictions = model.predict(X_test)

print("Total examples:", len(X))
print("Training examples:", len(X_train))
print("Test examples:", len(X_test))
print("Accuracy:", f"{accuracy_score(y_test, predictions):.2%}")
print(classification_report(
    y_test,
    predictions,
    zero_division=0,
))

# 5. Save the newly trained model
joblib.dump(model, MODEL_FILE)

print("Updated model saved to:", MODEL_FILE)
