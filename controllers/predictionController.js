const Prediction = require('../models/Prediction');
const Child = require('../models/Child');
const { predictMalnutrition } = require('../services/predictionService');

// @desc    Run and log AI malnutrition prediction
// @route   POST /api/predict
// @access  Private
exports.predictChild = async (req, res, next) => {
  try {
    const { childId, childName, age, gender, height, weight, MUAC, headCircumference } = req.body;

    const ageMonths = parseInt(age || 0, 10);
    const hVal = parseFloat(height);
    const wVal = parseFloat(weight);
    const mVal = parseFloat(MUAC);
    const hcVal = parseFloat(headCircumference);

    // Call prediction service logic
    const predictionResult = predictMalnutrition({
      ageMonths,
      gender,
      height: hVal,
      weight: wVal,
      MUAC: mVal,
      headCircumference: hcVal
    });

    let dbPrediction = null;

    // If childId is provided, persist prediction log in Database
    if (childId) {
      const child = await Child.findById(childId);
      if (child) {
        dbPrediction = await Prediction.create({
          childId,
          riskLevel: predictionResult.riskLevel,
          confidence: predictionResult.confidence,
          recommendation: predictionResult.recommendation,
          growthStatus: predictionResult.growthStatus,
          followUpSuggestion: predictionResult.followUpSuggestion
        });
      }
    }

    res.status(200).json({
      success: true,
      message: 'AI Prediction generated successfully',
      data: {
        predictionResult: predictionResult.growthStatus,
        confidence: predictionResult.confidence,
        recommendation: predictionResult.recommendation,
        followUpSuggestion: predictionResult.followUpSuggestion,
        bmi: predictionResult.bmi,
        zScores: predictionResult.zScores,
        savedRecord: dbPrediction
      }
    });
  } catch (error) {
    next(error);
  }
};

// @desc    Get prediction history for a child
// @route   GET /api/predict/:childId
// @access  Private
exports.getPredictionsByChild = async (req, res, next) => {
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
        errors: ['You do not have permission to view predictions for this child']
      });
    }

    const predictions = await Prediction.find({ childId }).sort({ predictedDate: -1 });

    res.status(200).json({
      success: true,
      message: 'Predictions retrieved successfully',
      data: predictions
    });
  } catch (error) {
    next(error);
  }
};
