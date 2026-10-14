
import joblib

# Load the previously trained model
model = joblib.load("server_model.joblib")

print("Server Overload Predictor")
print("Enter CPU and RAM usage percentages.")
print("Type 'q' to quit.")

while True:
    cpu_input = input("\nCPU usage (%): ").strip()

    if cpu_input.lower() == "q":
        break

    ram_input = input("RAM usage (%): ").strip()

    try:
        cpu = float(cpu_input)
        ram = float(ram_input)

        if not (0 <= cpu <= 100 and 0 <= ram <= 100):
            print("Please enter percentages between 0 and 100.")
            continue

        # Predict using the saved model
        result = model.predict([[cpu, ram]])[0]

        if result == 1:
            print("Prediction: OVERLOADED")
        else:
            print("Prediction: NORMAL")

    except ValueError:
        print("Invalid input. Enter numbers between 0 and 100.")

print("Predictor stopped.")
