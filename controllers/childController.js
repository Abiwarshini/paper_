const Child = require('../models/Child');
const GrowthRecord = require('../models/GrowthRecord');
const Prediction = require('../models/Prediction');

// @desc    Create a new child profile
// @route   POST /api/children
// @access  Private (Admin, Doctor, Health Worker, Parent)
exports.createChild = async (req, res, next) => {
  try {
    const { childName, dob, gender, motherName, fatherName, address, district, state, phone } = req.body;

    const child = await Child.create({
      childName,
      dob,
      gender,
      motherName,
      fatherName,
      address,
      district,
      state,
      phone,
      createdBy: req.user.id
    });

    res.status(201).json({
      success: true,
      message: 'Child profile created successfully',
      data: child
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Get all children (with Search, Filter, Pagination)
// @route   GET /api/children
// @access  Private
exports.getChildren = async (req, res, next) => {
  try {
    // 1. Build query object based on user role
    let query = {};
    if (req.user.role === 'Parent') {
      query.createdBy = req.user.id;
    }

    // 2. Search by Child Name or Parent Names
    if (req.query.search) {
      const searchRegex = new RegExp(req.query.search, 'i');
      query.$or = [
        { childName: searchRegex },
        { motherName: searchRegex },
        { fatherName: searchRegex }
      ];
    }

    // 3. Filter options
    if (req.query.gender) {
      query.gender = req.query.gender;
    }
    if (req.query.district) {
      query.district = { $regex: new RegExp(req.query.district, 'i') };
    }
    if (req.query.state) {
      query.state = { $regex: new RegExp(req.query.state, 'i') };
    }

    // 4. Pagination Setup
    const page = parseInt(req.query.page, 10) || 1;
    const limit = parseInt(req.query.limit, 10) || 10;
    const startIndex = (page - 1) * limit;
    const endIndex = page * limit;
    const total = await Child.countDocuments(query);

    // 5. Query DB
    const children = await Child.find(query)
      .sort({ createdAt: -1 })
      .skip(startIndex)
      .limit(limit);

    // Pagination details
    const pagination = {
      currentPage: page,
      totalPages: Math.ceil(total / limit),
      totalRecords: total,
      limit
    };

    if (endIndex < total) {
      pagination.next = {
        page: page + 1,
        limit
      };
    }
    if (startIndex > 0) {
      pagination.prev = {
        page: page - 1,
        limit
      };
    }

    res.status(200).json({
      success: true,
      message: 'Children retrieved successfully',
      data: {
        children,
        pagination
      }
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Get single child details
// @route   GET /api/children/:id
// @access  Private
exports.getChild = async (req, res, next) => {
  try {
    const child = await Child.findById(req.params.id);

    if (!child) {
      return res.status(404).json({
        success: false,
        message: 'Child profile not found',
        errors: [`Child with ID ${req.params.id} does not exist`]
      });
    }

    // Authorization check
    if (req.user.role === 'Parent' && child.createdBy.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to view this profile',
        errors: ['You do not have permission to view other parents\' children details']
      });
    }

    // Retrieve growth history and latest predictions
    const growthHistory = await GrowthRecord.find({ childId: child._id }).sort({ measurementDate: -1 });
    const latestPrediction = await Prediction.findOne({ childId: child._id }).sort({ predictedDate: -1 });

    res.status(200).json({
      success: true,
      message: 'Child details retrieved successfully',
      data: {
        child,
        growthHistory,
        latestPrediction
      }
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Update child profile
// @route   PUT /api/children/:id
// @access  Private
exports.updateChild = async (req, res, next) => {
  try {
    let child = await Child.findById(req.params.id);

    if (!child) {
      return res.status(404).json({
        success: false,
        message: 'Child profile not found',
        errors: [`Child with ID ${req.params.id} does not exist`]
      });
    }

    // Authorization check
    if (req.user.role === 'Parent' && child.createdBy.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to edit this profile',
        errors: ['You do not have permission to modify this child profile']
      });
    }

    child = await Child.findByIdAndUpdate(req.params.id, req.body, {
      new: true,
      runValidators: true
    });

    res.status(200).json({
      success: true,
      message: 'Child profile updated successfully',
      data: child
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Delete child profile
// @route   DELETE /api/children/:id
// @access  Private
exports.deleteChild = async (req, res, next) => {
  try {
    const child = await Child.findById(req.params.id);

    if (!child) {
      return res.status(404).json({
        success: false,
        message: 'Child profile not found',
        errors: [`Child with ID ${req.params.id} does not exist`]
      });
    }

    // Authorization check
    if (req.user.role === 'Parent' && child.createdBy.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized to delete this profile',
        errors: ['You do not have permission to delete this child profile']
      });
    }

    // Delete child profile
    await Child.findByIdAndDelete(req.params.id);

    // Cascade delete child's growth records and predictions
    await GrowthRecord.deleteMany({ childId: child._id });
    await Prediction.deleteMany({ childId: child._id });

    res.status(200).json({
      success: true,
      message: 'Child profile and related health logs deleted successfully',
      data: {}
    });
  } catch (error) {
    next(error);
  }
};
