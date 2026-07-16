// WHO Growth Reference Interpolation and Prediction Logic
// Reference ranges for Age 0 - 60 months (5 years)

const growthData = {
  Male: {
    // ageMonths: [median_height, sd_height, median_weight, sd_weight, median_bmi, sd_bmi]
    0: [49.9, 1.9, 3.3, 0.4, 13.3, 0.8],
    6: [67.6, 2.3, 7.9, 0.8, 17.3, 1.0],
    12: [75.7, 2.6, 9.6, 0.9, 16.7, 0.9],
    18: [82.3, 2.9, 10.9, 1.0, 16.1, 0.9],
    24: [87.8, 3.2, 12.2, 1.2, 15.8, 0.9],
    36: [96.1, 3.7, 14.3, 1.5, 15.5, 0.9],
    48: [103.3, 4.2, 16.3, 1.8, 15.3, 0.9],
    60: [110.0, 4.6, 18.3, 2.1, 15.1, 0.9]
  },
  Female: {
    0: [49.1, 1.9, 3.2, 0.4, 13.3, 0.8],
    6: [65.7, 2.3, 7.3, 0.8, 16.9, 1.0],
    12: [74.0, 2.6, 8.9, 0.9, 16.3, 0.9],
    18: [80.7, 2.9, 10.2, 1.0, 15.7, 0.9],
    24: [86.4, 3.2, 11.5, 1.2, 15.4, 0.9],
    36: [95.1, 3.7, 13.9, 1.5, 15.2, 0.9],
    48: [102.7, 4.2, 15.5, 1.8, 15.0, 0.9],
    60: [109.4, 4.6, 17.4, 2.1, 14.8, 0.9]
  }
};

// Simple linear interpolation helper
const interpolate = (age, gender, metricIndex) => {
  const genderData = growthData[gender] || growthData.Male;
  const ages = Object.keys(genderData).map(Number).sort((a, b) => a - b);
  
  if (age <= ages[0]) {
    return {
      median: genderData[ages[0]][metricIndex],
      sd: genderData[ages[0]][metricIndex + 1]
    };
  }
  
  if (age >= ages[ages.length - 1]) {
    return {
      median: genderData[ages[ages.length - 1]][metricIndex],
      sd: genderData[ages[ages.length - 1]][metricIndex + 1]
    };
  }

  // Find surrounding age brackets
  let lowerAge = ages[0];
  let upperAge = ages[ages.length - 1];
  for (let i = 0; i < ages.length - 1; i++) {
    if (age >= ages[i] && age <= ages[i + 1]) {
      lowerAge = ages[i];
      upperAge = ages[i + 1];
      break;
    }
  }

  const fraction = (age - lowerAge) / (upperAge - lowerAge);
  
  const lowerMedian = genderData[lowerAge][metricIndex];
  const upperMedian = genderData[upperAge][metricIndex];
  const lowerSD = genderData[lowerAge][metricIndex + 1];
  const upperSD = genderData[upperAge][metricIndex + 1];

  return {
    median: lowerMedian + fraction * (upperMedian - lowerMedian),
    sd: lowerSD + fraction * (upperSD - lowerSD)
  };
};

/**
 * Predict Malnutrition Risk & Calculate Z-Scores
 * @param {Object} input
 * @param {number} input.ageMonths
 * @param {string} input.gender (Male/Female/Other)
 * @param {number} input.height (cm)
 * @param {number} input.weight (kg)
 * @param {number} input.MUAC (cm)
 * @param {number} input.headCircumference (cm)
 */
const predictMalnutrition = (input) => {
  const { ageMonths, height, weight, MUAC, headCircumference } = input;
  const gender = (input.gender === 'Female' || input.gender === 'Male') ? input.gender : 'Male';

  // Calculate BMI
  const heightInMeters = height / 100;
  const bmi = parseFloat((weight / (heightInMeters * heightInMeters)).toFixed(2));

  // Fetch WHO Reference values
  // Metrics: height=index 0, weight=index 2, bmi=index 4
  const heightRef = interpolate(ageMonths, gender, 0);
  const weightRef = interpolate(ageMonths, gender, 2);
  const bmiRef = interpolate(ageMonths, gender, 4);

  // Compute Z-Scores
  const HAZ = parseFloat(((height - heightRef.median) / heightRef.sd).toFixed(2));
  const WAZ = parseFloat(((weight - weightRef.median) / weightRef.sd).toFixed(2));
  const BAZ = parseFloat(((bmi - bmiRef.median) / bmiRef.sd).toFixed(2));

  // Determine Growth Status and Risk Levels
  let riskLevel = 'Low';
  let growthStatus = 'Healthy';
  let confidence = 95; // Default base confidence

  // MUAC indicator is crucial for acute malnutrition in children 6-59 months
  const hasMuacRisk = ageMonths >= 6 && ageMonths <= 60;

  if (MUAC < 11.5 || BAZ < -3 || WAZ < -3) {
    growthStatus = 'Severe Malnutrition';
    riskLevel = 'High';
  } else if ((MUAC >= 11.5 && MUAC < 12.5) || BAZ < -2 || WAZ < -2) {
    growthStatus = 'Moderate Malnutrition';
    riskLevel = 'Moderate';
  } else if (HAZ < -2) {
    growthStatus = 'Stunted';
    riskLevel = 'Moderate';
  } else if (BAZ < -2) {
    growthStatus = 'Wasted';
    riskLevel = 'Moderate';
  } else if (WAZ < -2) {
    growthStatus = 'Underweight';
    riskLevel = 'Moderate';
  }

  // Adjust confidence depending on parameters deviance
  const worstZ = Math.min(HAZ, WAZ, BAZ);
  if (Math.abs(worstZ) > 3) {
    confidence = Math.min(98, 90 + Math.round(Math.abs(worstZ) * 2));
  } else {
    confidence = Math.max(75, 95 - Math.round(Math.abs(worstZ) * 5));
  }

  // Generate nutritional recommendations and calories
  let recommendation = '';
  let followUpSuggestion = '';
  let calories = 1200;
  let protein = 20;
  let vitamins = ['Vitamin A', 'Vitamin D'];
  let mealPlan = [];

  if (growthStatus === 'Severe Malnutrition') {
    calories = 1800;
    protein = 45;
    vitamins = ['Vitamin A', 'Zinc', 'Iron', 'Multivitamins', 'Folic Acid'];
    recommendation = 'Urgent clinical evaluation required. Initiate Ready-to-Use Therapeutic Food (RUTF) (e.g. Plumpy\'Nut), 3 sachets daily. Closely monitor for infections, hypothermia, or dehydration.';
    followUpSuggestion = 'Immediate referral to the nearest stabilization center. Daily follow-up by community health worker.';
    mealPlan = [
      'Breakfast: RUTF (1 sachet) + Clean drinking water',
      'Mid-Morning: Fortified milk (F-75/F-100) or high energy gruel',
      'Lunch: RUTF (1 sachet) + mineral-supplemented water',
      'Afternoon: RUTF (1 sachet)',
      'Dinner: High-density vegetable purée with added oil, milk powder'
    ];
  } else if (growthStatus === 'Moderate Malnutrition' || growthStatus === 'Wasted' || growthStatus === 'Underweight') {
    calories = 1500;
    protein = 32;
    vitamins = ['Vitamin A', 'Zinc', 'Vitamin C', 'Calcium'];
    recommendation = 'Increase dietary intake with energy-dense nutrient-rich family foods. Provide 3 main meals and 2 snacks daily. Incorporate peanut butter, eggs, milk, lentils, and bananas.';
    followUpSuggestion = 'Re-check height, weight, and MUAC in 14 days. Review feeding logs.';
    mealPlan = [
      'Breakfast: Banana porridge with whole milk, 1 boiled egg',
      'Snack: Peanut butter sandwich or high-protein biscuits',
      'Lunch: Rice with thick split-pea soup (dal), mashed potato, vegetables cooked with vegetable oil',
      'Snack: Mixed seasonal fruit salad or yogurt',
      'Dinner: Minced meat/chicken or tofu stew with rice, green leafy vegetables'
    ];
  } else if (growthStatus === 'Stunted') {
    calories = 1400;
    protein = 35;
    vitamins = ['Calcium', 'Vitamin D', 'Zinc', 'Iron'];
    recommendation = 'Focus on high protein and calcium content to support linear skeletal growth. Provide dairy products (milk, cheese, yogurt), eggs, lean poultry, fish, beans, and dark leafy greens.';
    followUpSuggestion = 'Monthly growth monitoring and pediatrician review. Check for chronic clinical factors.';
    mealPlan = [
      'Breakfast: Ragi/Finger millet porridge with milk and nuts, 1 scrambled egg',
      'Snack: A glass of whole milk, almonds',
      'Lunch: Fish or chicken curry with brown rice, sautéed spinach',
      'Snack: Cottage cheese (paneer) cubes or yoghurt cup',
      'Dinner: Lentil soup (dal) with chapati, boiled beans, carrot salad'
    ];
  } else {
    // Healthy
    calories = 1200;
    protein = 22;
    vitamins = ['Vitamin D', 'Calcium (dietary)'];
    recommendation = 'Maintain a well-balanced, age-appropriate diet consisting of grains, vegetables, fruits, proteins, and dairy. Encourage active play and proper hydration.';
    followUpSuggestion = 'Routine checkup in 2-3 months as per standard growth monitoring schedules.';
    mealPlan = [
      'Breakfast: Oatmeal with chopped apples and honey, cup of warm milk',
      'Snack: Fresh orange slices or apple wedges',
      'Lunch: Rice/roti with seasonal vegetable stir-fry, bowl of yellow dal',
      'Snack: Handful of roasted chickpeas or plain yogurt',
      'Dinner: Vegetable soup, baked fish or tofu with stir-fried greens'
    ];
  }

  return {
    growthStatus,
    riskLevel,
    confidence,
    bmi,
    zScores: { HAZ, WAZ, BAZ },
    nutrition: {
      calories,
      protein,
      vitamins,
      mealPlan
    },
    recommendation,
    followUpSuggestion
  };
};

module.exports = { predictMalnutrition };
