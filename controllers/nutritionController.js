const NutritionPlan = require('../models/NutritionPlan');
const Child = require('../models/Child');

// @desc    Create or update nutrition plan for a child
// @route   POST /api/nutrition
// @access  Private (Admin, Doctor, Health Worker)
exports.saveNutritionPlan = async (req, res, next) => {
  try {
    const { childId, mealPlan, calories, protein, vitamins } = req.body;

    // Validate child
    const child = await Child.findById(childId);
    if (!child) {
      return res.status(404).json({
        success: false,
        message: 'Child not found',
        errors: [`Child with ID ${childId} does not exist`]
      });
    }

    const plan = await NutritionPlan.findOneAndUpdate(
      { childId },
      { mealPlan, calories, protein, vitamins },
      { new: true, upsert: true, runValidators: true }
    );

    res.status(200).json({
      success: true,
      message: 'Nutrition plan saved successfully',
      data: plan
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Get nutrition plan for a child
// @route   GET /api/nutrition/:childId
// @access  Private
exports.getNutritionPlan = async (req, res, next) => {
  try {
    const { childId } = req.params;

    // Validate child
    const child = await Child.findById(childId);
    if (!child) {
      return res.status(404).json({
        success: false,
        message: 'Child not found',
        errors: [`Child with ID ${childId} does not exist`]
      });
    }

    if (req.user.role === 'Parent' && child.createdBy.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized',
        errors: ['You do not have permission to view nutrition plans for this child']
      });
    }

    const plan = await NutritionPlan.findOne({ childId });

    if (!plan) {
      return res.status(404).json({
        success: false,
        message: 'Nutrition plan not found for this child',
        errors: ['No nutrition record found']
      });
    }

    res.status(200).json({
      success: true,
      message: 'Nutrition plan retrieved successfully',
      data: plan
    });
  } catch (error) {
    next(error);
  }
};
