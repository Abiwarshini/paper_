const mongoose = require('mongoose');

const ChildSchema = new mongoose.Schema({
  childName: {
    type: String,
    required: [true, 'Please add the child\'s name'],
    trim: true
  },
  dob: {
    type: Date,
    required: [true, 'Please add date of birth']
  },
  gender: {
    type: String,
    required: [true, 'Please select gender'],
    enum: ['Male', 'Female', 'Other']
  },
  motherName: {
    type: String,
    trim: true
  },
  fatherName: {
    type: String,
    trim: true
  },
  address: {
    type: String,
    trim: true
  },
  district: {
    type: String,
    trim: true
  },
  state: {
    type: String,
    trim: true
  },
  phone: {
    type: String,
    trim: true
  },
  createdBy: {
    type: mongoose.Schema.ObjectId,
    ref: 'User',
    required: true
  }
}, {
  timestamps: true
});

module.exports = mongoose.model('Child', ChildSchema);
