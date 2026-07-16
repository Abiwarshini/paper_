const express = require('express');
const router = express.Router();
const { register, login, logout, forgotPassword, resetPassword, updateProfile } = require('../controllers/authController');
const { protect } = require('../middleware/authMiddleware');
const upload = require('../middleware/uploadMiddleware');

router.post('/register', upload.single('profileImage'), register);
router.post('/login', login);
router.post('/logout', protect, logout);
router.post('/forgot-password', forgotPassword);
router.post('/reset-password', resetPassword);
router.put('/profile', protect, upload.single('profileImage'), updateProfile);

module.exports = router;
