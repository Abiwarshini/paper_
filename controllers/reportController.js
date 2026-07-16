const Child = require('../models/Child');
const Prediction = require('../models/Prediction');
const User = require('../models/User');

// @desc    Get aggregate growth, role distributions and status breakdowns for reporting
// @route   GET /api/reports
// @access  Private (Admin, Doctor, Health Worker)
exports.getSystemReports = async (req, res, next) => {
  try {
    // 1. Counts of registrations and user roles
    const totalChildren = await Child.countDocuments();
    const totalUsers = await User.countDocuments();
    const doctorCount = await User.countDocuments({ role: 'Doctor' });
    const healthWorkerCount = await User.countDocuments({ role: 'Health Worker' });
    const parentCount = await User.countDocuments({ role: 'Parent' });

    // 2. Aggregate child growth classifications by grabbing the latest diagnostic predictions
    const childrenIds = await Child.find({}, '_id');
    
    let healthy = 0;
    let moderateMalnutrition = 0;
    let severeMalnutrition = 0;
    let stunted = 0;
    let wasted = 0;
    let underweight = 0;

    for (const child of childrenIds) {
      const latestPred = await Prediction.findOne({ childId: child._id }).sort({ predictedDate: -1 });
      if (latestPred) {
        const status = latestPred.growthStatus;
        if (status === 'Healthy') healthy++;
        else if (status === 'Moderate Malnutrition') moderateMalnutrition++;
        else if (status === 'Severe Malnutrition') severeMalnutrition++;
        else if (status === 'Stunted') stunted++;
        else if (status === 'Wasted') wasted++;
        else if (status === 'Underweight') underweight++;
      } else {
        healthy++; // default fallback if no predictions logged yet
      }
    }

    // 3. Recent predictions list (populated with child data)
    const recentPredictions = await Prediction.find()
      .populate('childId', 'childName gender dob')
      .sort({ predictedDate: -1 })
      .limit(10);

    // 4. Geographic hot-spot distribution (grouped by district)
    const regionalBreakdown = await Child.aggregate([
      {
        $group: {
          _id: '$district',
          count: { $sum: 1 }
        }
      },
      { $sort: { count: -1 } },
      { $limit: 10 }
    ]);

    res.status(200).json({
      success: true,
      message: 'System reports compiled successfully',
      data: {
        counts: {
          totalChildren,
          totalUsers,
          doctors: doctorCount,
          healthWorkers: healthWorkerCount,
          parents: parentCount
        },
        statusBreakdown: {
          healthy,
          moderateMalnutrition,
          severeMalnutrition,
          stunted,
          wasted,
          underweight
        },
        regionalBreakdown,
        recentPredictions
      }
    });
  } catch (error) {
    next(error);
  }
};
