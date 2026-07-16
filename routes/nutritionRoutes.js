const express = require('express');
const router = express.Router();
const { saveNutritionPlan, getNutritionPlan } = require('../controllers/nutritionController');
const { protect, authorize } = require('../middleware/authMiddleware');

router.use(protect); // Secure nutrition endpoints

router.post('/', authorize('Admin', 'Doctor', 'Health Worker'), saveNutritionPlan);
router.get('/:childId', getNutritionPlan);

module.exports = router;
