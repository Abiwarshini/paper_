const crypto = require('crypto');
const User = require('../models/User');
const sendEmail = require('../services/mailerService');
const jwt = require('jsonwebtoken');

// Helper to generate token and send response
const sendTokenResponse = (user, statusCode, res, message = 'Success') => {
  const token = jwt.sign(
    { id: user._id },
    process.env.JWT_SECRET || 'supersecretjwtkey_ai_malnutrition_2026_growth_system',
    { expiresIn: process.env.JWT_EXPIRE || '30d' }
  );

  res.status(statusCode).json({
    success: true,
    message,
    data: {
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        phone: user.phone,
        profileImage: user.profileImage
      }
    }
  });
};

// @desc    Register a user
// @route   POST /api/auth/register
// @access  Public
exports.register = async (req, res, next) => {
  try {
    const { name, email, password, role, phone } = req.body;

    // Check if user already exists
    const userExists = await User.findOne({ email });
    if (userExists) {
      return res.status(400).json({
        success: false,
        message: 'User already exists',
        errors: ['User already exists with this email address']
      });
    }

    // Capture profile image if uploaded
    let profileImage = '';
    if (req.file) {
      profileImage = `/uploads/${req.file.filename}`;
    }

    // Create user
    const user = await User.create({
      name,
      email,
      password,
      role: role || 'Parent',
      phone,
      profileImage
    });

    sendTokenResponse(user, 201, res, 'User registered successfully');
  } catch (error) {
    next(error);
  }
};

// @desc    Login user
// @route   POST /api/auth/login
// @access  Public
exports.login = async (req, res, next) => {
  try {
    const { email, password } = req.body;

    // Validate email & password
    if (!email || !password) {
      return res.status(400).json({
        success: false,
        message: 'Please provide email and password',
        errors: ['Email and password are required']
      });
    }

    // Check for user
    const user = await User.findOne({ email }).select('+password');
    if (!user) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials',
        errors: ['Invalid email or password']
      });
    }

    // Check if password matches
    const isMatch = await user.matchPassword(password);
    if (!isMatch) {
      return res.status(401).json({
        success: false,
        message: 'Invalid credentials',
        errors: ['Invalid email or password']
      });
    }

    sendTokenResponse(user, 200, res, 'User logged in successfully');
  } catch (error) {
    next(error);
  }
};

// @desc    Logout user
// @route   POST /api/auth/logout
// @access  Private
exports.logout = async (req, res, next) => {
  try {
    res.status(200).json({
      success: true,
      message: 'User logged out successfully',
      data: {}
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Forgot password
// @route   POST /api/auth/forgot-password
// @access  Public
exports.forgotPassword = async (req, res, next) => {
  try {
    const { email } = req.body;
    const user = await User.findOne({ email });

    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found',
        errors: ['No account registered with this email address']
      });
    }

    // Generate reset token
    const resetToken = crypto.randomBytes(20).toString('hex');

    // Hash token and set to resetPasswordToken field
    user.resetPasswordToken = crypto
      .createHash('sha256')
      .update(resetToken)
      .digest('hex');

    // Set expire: 10 minutes
    user.resetPasswordExpire = Date.now() + 10 * 60 * 1000;
    await user.save({ validateBeforeSave: false });

    // Create reset URL
    // e.g. frontend reset page: http://localhost:5173/reset-password/:token
    const resetUrl = `${req.protocol}://${req.get('host').replace('5000', '5173')}/reset-password/${resetToken}`;

    const message = `You are receiving this email because you (or someone else) has requested the reset of a password. Please make a PUT request to: \n\n ${resetUrl}`;
    const html = `
      <h3>Growth & Malnutrition Monitoring System</h3>
      <p>You requested a password reset. Please click the link below to set a new password:</p>
      <a href="${resetUrl}" target="_blank" style="padding: 10px 20px; background-color: #0d6efd; color: white; text-decoration: none; border-radius: 5px; display: inline-block;">Reset Password</a>
      <p>If you did not request this, please ignore this email.</p>
      <p>This link is valid for 10 minutes.</p>
    `;

    try {
      const previewUrl = await sendEmail({
        email: user.email,
        subject: 'Password reset token',
        message,
        html
      });

      res.status(200).json({
        success: true,
        message: 'Email sent successfully',
        data: {
          previewUrl: previewUrl || undefined
        }
      });
    } catch (err) {
      console.error(err);
      user.resetPasswordToken = undefined;
      user.resetPasswordExpire = undefined;
      await user.save({ validateBeforeSave: false });

      return res.status(500).json({
        success: false,
        message: 'Email could not be sent',
        errors: ['Failed to trigger mailer, try again later']
      });
    }
  } catch (error) {
    next(error);
  }
};

// @desc    Reset password
// @route   POST /api/auth/reset-password
// @access  Public
exports.resetPassword = async (req, res, next) => {
  try {
    const { token, password } = req.body;

    if (!token || !password) {
      return res.status(400).json({
        success: false,
        message: 'Token and new password are required',
        errors: ['Missing token or password']
      });
    }

    // Get hashed token
    const resetPasswordToken = crypto
      .createHash('sha256')
      .update(token)
      .digest('hex');

    const user = await User.findOne({
      resetPasswordToken,
      resetPasswordExpire: { $gt: Date.now() }
    });

    if (!user) {
      return res.status(400).json({
        success: false,
        message: 'Invalid or expired token',
        errors: ['Reset token is invalid or has expired']
      });
    }

    // Set new password
    user.password = password;
    user.resetPasswordToken = undefined;
    user.resetPasswordExpire = undefined;
    await user.save();

    sendTokenResponse(user, 200, res, 'Password reset successfully');
  } catch (error) {
    next(error);
  }
};

// @desc    Update current user profile
// @route   PUT /api/auth/profile
// @access  Private
exports.updateProfile = async (req, res, next) => {
  try {
    const user = await User.findById(req.user.id);
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found',
        errors: ['User not found']
      });
    }

    user.name = req.body.name || user.name;
    user.phone = req.body.phone || user.phone;

    if (req.body.password) {
      user.password = req.body.password;
    }

    if (req.file) {
      user.profileImage = `/uploads/${req.file.filename}`;
    }

    await user.save();

    res.status(200).json({
      success: true,
      message: 'Profile updated successfully',
      data: {
        user: {
          id: user._id,
          name: user.name,
          email: user.email,
          role: user.role,
          phone: user.phone,
          profileImage: user.profileImage
        }
      }
    });
  } catch (error) {
    next(error);
  }
};

