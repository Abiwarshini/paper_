const mongoose = require('mongoose');

const PredictionSchema = new mongoose.Schema({
  childId: {
    type: mongoose.Schema.ObjectId,
    ref: 'Child',
    required: true
  },
  riskLevel: {
    type: String,
    enum: ['Low', 'Moderate', 'High'],
    required: true
  },
  confidence: {
    type: Number,
    required: true // Percentage value (e.g. 92)
  },
  recommendation: {
    type: String,
    required: true
  },
  growthStatus: {
    type: String,
    required: true // E.g., 'Healthy', 'Moderate Malnutrition', 'Severe Malnutrition', 'Stunted', 'Wasted', 'Underweight'
  },
  followUpSuggestion: {
    type: String,
    default: 'Schedule next physical checkup in 30 days.'
  },
  predictedDate: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

module.exports = mongoose.model('Prediction', PredictionSchema);
