const GrowthRecord = require('../models/GrowthRecord');
const Child = require('../models/Child');
const Prediction = require('../models/Prediction');
const NutritionPlan = require('../models/NutritionPlan');
const Notification = require('../models/Notification');
const { predictMalnutrition } = require('../services/predictionService');

// @desc    Record growth parameters for a child & trigger prediction
// @route   POST /api/growth
// @access  Private (Admin, Doctor, Health Worker, Parent)
exports.recordGrowth = async (req, res, next) => {
  try {
    const { childId, height, weight, MUAC, headCircumference, measurementDate } = req.body;

    // Validate if child exists
    const child = await Child.findById(childId);
    if (!child) {
      return res.status(404).json({
        success: false,
        message: 'Child not found',
        errors: [`Child with ID ${childId} does not exist`]
      });
    }

    // Calculate age in months based on Child's dob and measurement date
    const mDate = measurementDate ? new Date(measurementDate) : new Date();
    const dobDate = new Date(child.dob);
    let ageMonths = (mDate.getFullYear() - dobDate.getFullYear()) * 12 + (mDate.getMonth() - dobDate.getMonth());
    if (mDate.getDate() < dobDate.getDate()) {
      ageMonths--; // partial month adjustment
    }
    ageMonths = Math.max(0, ageMonths); // ensure non-negative

    // Create growth record
    const growthRecord = new GrowthRecord({
      childId,
      height,
      weight,
      MUAC,
      headCircumference,
      ageMonths,
      measurementDate: mDate
    });

    await growthRecord.save();

    // Trigger AI prediction logic based on WHO percentiles
    const predictionResult = predictMalnutrition({
      ageMonths,
      gender: child.gender,
      height,
      weight,
      MUAC,
      headCircumference
    });

    // Save Prediction in MongoDB
    const prediction = await Prediction.create({
      childId,
      riskLevel: predictionResult.riskLevel,
      confidence: predictionResult.confidence,
      recommendation: predictionResult.recommendation,
      growthStatus: predictionResult.growthStatus,
      followUpSuggestion: predictionResult.followUpSuggestion,
      predictedDate: mDate
    });

    // Sync Nutrition Plan in MongoDB
    await NutritionPlan.findOneAndUpdate(
      { childId },
      {
        mealPlan: predictionResult.nutrition.mealPlan,
        calories: predictionResult.nutrition.calories,
        protein: predictionResult.nutrition.protein,
        vitamins: predictionResult.nutrition.vitamins
      },
      { upsert: true, new: true }
    );

    // Create Notification if Risk is Moderate or High
    if (predictionResult.riskLevel !== 'Low') {
      await Notification.create({
        user: child.createdBy, // Alert the parent
        message: `Growth Monitoring Alert: ${child.childName} was predicted with ${predictionResult.growthStatus} (${predictionResult.riskLevel} Risk).`
      });

      // Also alert the recording user if they are a doctor or health worker
      if (req.user.id !== child.createdBy.toString()) {
        await Notification.create({
          user: req.user.id,
          message: `Health Alert: ${child.childName} shows signs of ${predictionResult.growthStatus}. Recommended follow-up logged.`
        });
      }
    }

    res.status(201).json({
      success: true,
      message: 'Growth parameters recorded and AI diagnosis synchronized successfully',
      data: {
        growthRecord,
        prediction,
        nutrition: predictionResult.nutrition
      }
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Get growth history for a specific child
// @route   GET /api/growth/:childId
// @access  Private
exports.getGrowthHistory = async (req, res, next) => {
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

    // Role verification
    if (req.user.role === 'Parent' && child.createdBy.toString() !== req.user.id) {
      return res.status(403).json({
        success: false,
        message: 'Not authorized',
        errors: ['You do not have permission to view this child\'s health parameters']
      });
    }

    // Fetch growth logs
    const growthRecords = await GrowthRecord.find({ childId }).sort({ measurementDate: 1 });
    const predictions = await Prediction.find({ childId }).sort({ predictedDate: 1 });
    const nutritionPlan = await NutritionPlan.findOne({ childId });

    res.status(200).json({
      success: true,
      message: 'Growth history logs fetched successfully',
      data: {
        child,
        growthRecords,
        predictions,
        nutritionPlan
      }
    });
  } catch (error) {
    next(error);
  }
};
