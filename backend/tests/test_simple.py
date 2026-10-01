import math
import unittest

from fastapi.testclient import TestClient

from backend.main import app


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        self.assertIn("model", data)

    def test_model_info_endpoint_is_json_safe(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200, response.text)

        data = response.json()
        self.assertIn("model_params", data)

        params = data["model_params"]
        self.assertNotIn("missing", params)

        flat_values = [
            value
            for value in params.values()
            if isinstance(value, (int, float))
        ]
        self.assertFalse(
            any(math.isnan(value) or math.isinf(value) for value in flat_values if isinstance(value, float)),
            msg="NaN or infinite values should be sanitized before JSON serialization",
        )


if __name__ == "__main__":
    unittest.main()