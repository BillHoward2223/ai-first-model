
import os
import joblib

MODEL_FILE = "incremental_model.joblib"

def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"{MODEL_FILE} not found. Train the initial model first."
        )

    return joblib.load(MODEL_FILE)


def main():
    model = load_model()

    print("=" * 45)
    print(" IPTV Server AI - Incremental Model Predictor")
    print("=" * 45)
    print("Enter CPU and RAM usage percentages.")
    print("Type 'q' to quit.\n")

    while True:
        cpu_input = input("CPU usage (%): ").strip()

        if cpu_input.lower() == "q":
            break

        ram_input = input("RAM usage (%): ").strip()

        try:
            cpu = float(cpu_input)
            ram = float(ram_input)

            if not (0 <= cpu <= 100 and 0 <= ram <= 100):
                print("Error: CPU and RAM must be between 0 and 100.\n")
                continue

            # Predict using the currently loaded model.
            prediction = model.predict([[cpu, ram]])[0]

            # Probability is available for this SGDClassifier.
            probabilities = model.predict_proba([[cpu, ram]])[0]
            class_probabilities = dict(
                zip(model.classes_, probabilities)
            )
            score = class_probabilities.get(prediction, 0.0) * 100

            if prediction == 1:
                status = "OVERLOADED"
            else:
                status = "NORMAL"

            print(f"\nPrediction: {status}")
            print(f"Model probability for predicted class: {score:.2f}%")
            print("-" * 45)

        except ValueError:
            print("Error: Enter valid numbers between 0 and 100.\n")

    print("Predictor stopped.")


if __name__ == "__main__":
    main()
