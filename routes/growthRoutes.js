const express = require('express');
const router = express.Router();
const { recordGrowth, getGrowthHistory } = require('../controllers/growthController');
const { protect } = require('../middleware/authMiddleware');

router.use(protect); // Secure growth records

router.post('/', recordGrowth);
router.get('/:childId', getGrowthHistory);

module.exports = router;
