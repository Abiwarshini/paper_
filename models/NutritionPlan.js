const mongoose = require('mongoose');

const NutritionPlanSchema = new mongoose.Schema({
  childId: {
    type: mongoose.Schema.ObjectId,
    ref: 'Child',
    required: true,
    unique: true
  },
  mealPlan: {
    type: [String],
    default: []
  },
  calories: {
    type: Number,
    required: [true, 'Please add recommended daily calories (kcal)']
  },
  protein: {
    type: Number,
    required: [true, 'Please add recommended daily protein (g)']
  },
  vitamins: {
    type: [String],
    default: []
  }
}, {
  timestamps: true
});

module.exports = mongoose.model('NutritionPlan', NutritionPlanSchema);
