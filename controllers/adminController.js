const User = require('../models/User');

// @desc    Get all users (searchable, filter by role)
// @route   GET /api/admin/users
// @access  Private (Admin)
exports.getUsers = async (req, res, next) => {
  try {
    let query = {};

    if (req.query.role) {
      query.role = req.query.role;
    }

    if (req.query.search) {
      const searchRegex = new RegExp(req.query.search, 'i');
      query.$or = [
        { name: searchRegex },
        { email: searchRegex },
        { phone: searchRegex }
      ];
    }

    const users = await User.find(query).sort({ createdAt: -1 });

    res.status(200).json({
      success: true,
      message: 'Users retrieved successfully',
      data: users
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Create a doctor, health worker, or parent account directly
// @route   POST /api/admin/users
// @access  Private (Admin)
exports.createUser = async (req, res, next) => {
  try {
    const { name, email, password, role, phone } = req.body;

    const userExists = await User.findOne({ email });
    if (userExists) {
      return res.status(400).json({
        success: false,
        message: 'Email already registered',
        errors: ['User already exists with this email']
      });
    }

    const user = await User.create({
      name,
      email,
      password,
      role,
      phone
    });

    res.status(201).json({
      success: true,
      message: `${role} account created successfully`,
      data: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        phone: user.phone
      }
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Update any user account
// @route   PUT /api/admin/users/:id
// @access  Private (Admin)
exports.updateUser = async (req, res, next) => {
  try {
    let user = await User.findById(req.params.id);
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found',
        errors: [`No account found with ID ${req.params.id}`]
      });
    }

    // Hash password if updating password
    if (req.body.password) {
      user.password = req.body.password;
    }

    // Merge other fields
    user.name = req.body.name || user.name;
    user.email = req.body.email || user.email;
    user.role = req.body.role || user.role;
    user.phone = req.body.phone || user.phone;

    await user.save();

    res.status(200).json({
      success: true,
      message: 'User account updated successfully',
      data: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        phone: user.phone
      }
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Delete a user account
// @route   DELETE /api/admin/users/:id
// @access  Private (Admin)
exports.deleteUser = async (req, res, next) => {
  try {
    const user = await User.findById(req.params.id);
    if (!user) {
      return res.status(404).json({
        success: false,
        message: 'User not found',
        errors: [`No account found with ID ${req.params.id}`]
      });
    }

    // Prevent Admin deleting themselves
    if (user._id.toString() === req.user.id) {
      return res.status(400).json({
        success: false,
        message: 'Action denied',
        errors: ['Admins cannot delete their own active accounts']
      });
    }

    await User.findByIdAndDelete(req.params.id);

    res.status(200).json({
      success: true,
      message: 'User account deleted successfully',
      data: {}
    });
  } catch (error) {
    next(error);
  }
};
