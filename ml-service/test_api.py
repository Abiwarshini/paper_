import os
import sys
import unittest
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app

class MLServiceAPITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(len(data['conditions']), 7)

    def test_transformer_predict_valid(self):
        payload = {
            "age_months": 24,
            "height_cm": 85.0,
            "weight_kg": 11.5,
            "gender": "Female",
            "muac_cm": 14.0,
            "dietary_diversity": 4,
            "meal_frequency": 3,
            "breastfeeding_status": "Partial",
            "water_sanitation_index": 4
        }
        response = self.app.post('/predict/transformer', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['model'], 'FT-Transformer')
        self.assertEqual(len(data['predictions']), 7)
        self.assertIn('top_prediction', data)
        self.assertIn('top_factors', data)

    def test_transformer_predict_underweight_case(self):
        payload = {
            "age_months": 36,
            "height_cm": 90.0,
            "weight_kg": 8.0,  # Severely underweight for 36m
            "gender": "Male",
            "muac_cm": 11.0,   # Low MUAC
            "dietary_diversity": 2,
            "meal_frequency": 2
        }
        response = self.app.post('/predict/transformer', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['overall_risk'], 'HIGH')

    def test_xgboost_predict_valid(self):
        payload = {
            "age_months": 18,
            "height_cm": 80.0,
            "weight_kg": 10.0,
            "gender": "Male",
            "muac_cm": 13.5
        }
        response = self.app.post('/predict/xgboost', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['model'], 'XGBoost')
        self.assertEqual(len(data['predictions']), 7)

    def test_invalid_input_validation(self):
        payload = {
            "age_months": -5,  # Invalid negative age
            "height_cm": 80.0,
            "weight_kg": 10.0
        }
        response = self.app.post('/predict/transformer', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 422)

    def test_model_comparison_endpoint(self):
        response = self.app.get('/compare')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('summary', data)

if __name__ == '__main__':
    unittest.main()
