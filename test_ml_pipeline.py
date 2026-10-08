import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_created(self):
        self.assertTrue(
            os.path.exists("loan_approval_ci_data.csv")
        )

    def test_model_created(self):
        self.assertTrue(
            os.path.exists("loan_approval_model.pkl")
        )

    def test_metrics_created(self):
        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load(
            "loan_approval_model.pkl"
        )

        sample = pd.DataFrame([{
            "credit_score": 750,
            "loan_percent_income": 0.20,
            "person_income": 80000,
            "loan_amnt": 15000
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [0, 1]
        )

    def test_high_quality_loan(self):
        model = joblib.load(
            "loan_approval_model.pkl"
        )

        sample = pd.DataFrame([{
            "credit_score": 800,
            "loan_percent_income": 0.15,
            "person_income": 100000,
            "loan_amnt": 10000
        }])

        prediction = model.predict(sample)[0]

        self.assertEqual(
            int(prediction),
            0
        )

    def test_low_quality_loan(self):
        model = joblib.load(
            "loan_approval_model.pkl"
        )

        sample = pd.DataFrame([{
            "credit_score": 520,
            "loan_percent_income": 0.55,
            "person_income": 25000,
            "loan_amnt": 35000
        }])

        prediction = model.predict(sample)[0]

        self.assertEqual(
            int(prediction),
            0
        )


if __name__ == "__main__":
    unittest.main()
