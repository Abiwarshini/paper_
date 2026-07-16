const express = require('express');
const router = express.Router();
const { predictChild, getPredictionsByChild } = require('../controllers/predictionController');
const { protect } = require('../middleware/authMiddleware');

router.use(protect); // Secure prediction endpoints

router.post('/', predictChild);
router.get('/:childId', getPredictionsByChild);

module.exports = router;
