import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_quality_loan_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "credit_score": 750,
                "loan_percent_income": 0.20,
                "person_income": 80000,
                "loan_amnt": 10000
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "APPROVED"
        )

    def test_low_quality_loan_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "credit_score": 500,
                "loan_percent_income": 0.80,
                "person_income": 30000,
                "loan_amnt": 25000
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "REJECTED"
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "credit_score": 750,
                "loan_percent_income": 0.20
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
