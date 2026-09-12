import os
import sys
import unittest
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import app
from preprocessing.preprocessing import TARGET_CONDITIONS, NUTRITION_CONDITIONS, DISEASE_CONDITIONS


class MLServiceAPITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['num_conditions'], 10)
        self.assertEqual(len(data['nutrition_conditions']), 5)
        self.assertEqual(len(data['disease_conditions']), 5)
        self.assertIn('medical_disclaimer', data)

    def test_xgboost_predict_valid(self):
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
        response = self.app.post('/predict/xgboost', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['model'], 'XGBoost')
        self.assertEqual(len(data['predictions']), 10)
        self.assertEqual(len(data['nutrition_assessment']), 5)
        self.assertEqual(len(data['disease_screening']), 5)
        self.assertIn('top_prediction', data)
        self.assertIn('top_factors', data)
        self.assertIn('medical_disclaimer', data)

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
        self.assertEqual(len(data['predictions']), 10)
        self.assertEqual(len(data['nutrition_assessment']), 5)
        self.assertEqual(len(data['disease_screening']), 5)
        self.assertIn('top_prediction', data)
        self.assertIn('top_factors', data)

    def test_predict_dual_side_by_side(self):
        payload = {
            "age_months": 36,
            "height_cm": 88.0,
            "weight_kg": 9.5,
            "gender": "Male",
            "muac_cm": 12.0,
            "dietary_diversity": 2,
            "meal_frequency": 2,
            "breastfeeding_status": "Weaned",
            "water_sanitation_index": 2
        }
        response = self.app.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('best_model', data)
        self.assertIn('xgboost', data)
        self.assertIn('transformer', data)
        self.assertEqual(len(data['comparison']), 10)
        self.assertIn('delta', data['comparison'][0])

    def test_invalid_input_validation(self):
        # Missing weight
        payload = {
            "age_months": 24,
            "height_cm": 85.0
        }
        response = self.app.post('/predict/xgboost', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

        # Negative age
        payload_invalid_age = {
            "age_months": -5,
            "height_cm": 85.0,
            "weight_kg": 11.0
        }
        response = self.app.post('/predict/transformer', data=json.dumps(payload_invalid_age), content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_model_comparison_endpoint(self):
        response = self.app.get('/compare')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('summary', data)
        self.assertEqual(len(data['conditions']), 10)
        self.assertIn('xgboost_details', data)
        self.assertIn('transformer_details', data)


if __name__ == '__main__':
    unittest.main()
