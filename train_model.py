import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_dataset():
    """
    Create a reproducible Loan Approval dataset.
    """

    rng = np.random.default_rng(42)
    number_of_applicants = 300

    data = pd.DataFrame({
        "credit_score": rng.integers(500, 851, number_of_applicants),
        "loan_percent_income": rng.uniform(0.05, 0.60, number_of_applicants),
        "person_income": rng.integers(20000, 120001, number_of_applicants),
        "loan_amnt": rng.integers(1000, 40001, number_of_applicants)
    })

    # Loan approval rule used to generate the target
    data["approval_score"] = (
        0.50 * (data["credit_score"] / 850)
        + 0.20 * (1 - data["loan_percent_income"])
        + 0.15 * (data["person_income"] / 120000)
        + 0.15 * (1 - data["loan_amnt"] / 40000)
    )

    # 1 = APPROVED, 0 = REJECTED
    data["loan_status"] = (
        data["approval_score"] >= 0.55
    ).astype(int)

    return data


def train_model():
    print("Creating Loan Approval dataset...")

    data = create_dataset()

    data.to_csv("loan_approval_ci_data.csv", index=False)

    print("Dataset created successfully.")
    print("Number of records:", len(data))

    features = [
        "credit_score",
        "loan_percent_income",
        "person_income",
        "loan_amnt"
    ]

    X = data[features]
    y = data["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    print("Training Loan Approval model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    joblib.dump(
        model,
        "loan_approval_model.pkl"
    )

    print("\nModel saved as loan_approval_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")

    return accuracy


if __name__ == "__main__":
    train_model()
