const express = require('express');
const router = express.Router();
const { getSystemReports } = require('../controllers/reportController');
const { protect, authorize } = require('../middleware/authMiddleware');

router.use(protect);
router.get('/', authorize('Admin', 'Doctor', 'Health Worker'), getSystemReports);

module.exports = router;
