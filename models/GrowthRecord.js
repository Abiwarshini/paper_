const mongoose = require('mongoose');

const GrowthRecordSchema = new mongoose.Schema({
  childId: {
    type: mongoose.Schema.ObjectId,
    ref: 'Child',
    required: true
  },
  height: {
    type: Number,
    required: [true, 'Please add height in cm']
  },
  weight: {
    type: Number,
    required: [true, 'Please add weight in kg']
  },
  MUAC: {
    type: Number,
    required: [true, 'Please add Mid-Upper Arm Circumference (MUAC) in cm']
  },
  headCircumference: {
    type: Number,
    required: [true, 'Please add head circumference in cm']
  },
  BMI: {
    type: Number
  },
  ageMonths: {
    type: Number,
    required: [true, 'Please add age in months']
  },
  measurementDate: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

// Calculate BMI before saving
GrowthRecordSchema.pre('save', function(next) {
  if (this.height && this.weight) {
    const heightInMeters = this.height / 100;
    this.BMI = parseFloat((this.weight / (heightInMeters * heightInMeters)).toFixed(2));
  }
  next();
});

module.exports = mongoose.model('GrowthRecord', GrowthRecordSchema);
