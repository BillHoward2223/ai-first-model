
import argparse
import csv
import os
import shutil
from datetime import datetime, timezone

import joblib
from sklearn.linear_model import SGDClassifier

MODEL_FILE = "incremental_model.joblib"
ARCHIVE_DIR = "processed_batches"
FEATURES = ["cpu_pct", "ram_pct"]
TARGET = "overloaded"


def load_csv(path):
    X, y = [], []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = set(FEATURES + [TARGET])
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must contain these columns: {FEATURES + [TARGET]}"
            )

        for line, row in enumerate(reader, start=2):
            try:
                cpu = float(row["cpu_pct"])
                ram = float(row["ram_pct"])
                label = int(row[TARGET])

                if not (0 <= cpu <= 100 and 0 <= ram <= 100):
                    raise ValueError("CPU/RAM must be between 0 and 100")

                if label not in (0, 1):
                    raise ValueError("Label must be 0 or 1")

                X.append([cpu, ram])
                y.append(label)

            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid CSV row {line}: {exc}") from exc

    if not X:
        raise ValueError(f"No training examples found in {path}")

    return X, y


def save_model(model):
    temporary = MODEL_FILE + ".tmp"
    joblib.dump(model, temporary)
    os.replace(temporary, MODEL_FILE)


def main():
    parser = argparse.ArgumentParser(
        description="Train or incrementally update an AI model"
    )
    parser.add_argument("mode", choices=["init", "update"])
    parser.add_argument("csv_file")
    args = parser.parse_args()

    X, y = load_csv(args.csv_file)

    if args.mode == "init":
        if os.path.exists(MODEL_FILE):
            raise FileExistsError(
                f"{MODEL_FILE} already exists. Back it up before reinitializing."
            )

        model = SGDClassifier(
            loss="log_loss",
            random_state=42,
        )

        # First call declares all possible classes.
        model.partial_fit(X, y, classes=[0, 1])
        save_model(model)

        print(f"Initial model trained on {len(X)} examples.")
        print(f"Saved: {MODEL_FILE}")
        return

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            "No model found. Run init first."
        )

    model = joblib.load(MODEL_FILE)

    # Update the existing model using only the new batch.
    model.partial_fit(X, y)

    # Save the updated model before archiving the input batch.
    save_model(model)

    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archived_name = f"batch_{timestamp}.csv"
    archive_path = os.path.join(ARCHIVE_DIR, archived_name)

    shutil.move(args.csv_file, archive_path)

    print(f"Updated model using {len(X)} new examples.")
    print(f"Saved: {MODEL_FILE}")
    print(f"Archived batch: {archive_path}")


if __name__ == "__main__":
    main()
