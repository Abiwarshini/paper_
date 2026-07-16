import React, { useState, useEffect } from 'react';
import API from '../services/api';
import { 
  Activity, 
  HelpCircle, 
  TrendingUp, 
  FileText, 
  Utensils, 
  Heart,
  ChevronRight,
  ShieldCheck,
  AlertOctagon,
  Sparkles
} from 'lucide-react';
import Card from '../components/Card';
import Loader from '../components/Loader';
import Alert from '../components/Alert';

const Predictor = () => {
  const [children, setChildren] = useState([]);
  const [selectedChildId, setSelectedChildId] = useState('');
  const [loading, setLoading] = useState(true);

  // Form input states
  const [formData, setFormData] = useState({
    age: '',
    gender: 'Male',
    height: '',
    weight: '',
    MUAC: '',
    headCircumference: ''
  });

  // Result states
  const [result, setResult] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');

  // Fetch children list for easy loading
  useEffect(() => {
    const fetchKids = async () => {
      try {
        setLoading(true);
        const res = await API.get('/children?limit=100');
        if (res.data.success) {
          setChildren(res.data.data.children || []);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchKids();
  }, []);

  // Pre-fill form when child is selected
  const handleChildChange = async (e) => {
    const id = e.target.value;
    setSelectedChildId(id);
    setResult(null);
    setError('');

    if (!id) {
      setFormData({
        age: '',
        gender: 'Male',
        height: '',
        weight: '',
        MUAC: '',
        headCircumference: ''
      });
      return;
    }

    try {
      setSubmitting(true);
      const res = await API.get(`/children/${id}`);
      if (res.data.success) {
        const { child, growthHistory } = res.data.data;
        const latest = growthHistory?.[0]; // latest record
        
        // Calculate age in months
        const dobDate = new Date(child.dob);
        const refDate = latest ? new Date(latest.measurementDate) : new Date();
        let ageMonths = (refDate.getFullYear() - dobDate.getFullYear()) * 12 + (refDate.getMonth() - dobDate.getMonth());
        if (refDate.getDate() < dobDate.getDate()) ageMonths--;
        ageMonths = Math.max(0, ageMonths);

        setFormData({
          age: ageMonths.toString(),
          gender: child.gender,
          height: latest ? latest.height.toString() : '',
          weight: latest ? latest.weight.toString() : '',
          MUAC: latest ? (latest.MUAC || '').toString() : '',
          headCircumference: latest ? (latest.headCircumference || '').toString() : ''
        });
      }
    } catch (err) {
      console.error(err);
      setError('Could not pre-fill child details.');
    } finally {
      setSubmitting(false);
    }
  };

  const handleInputChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setResult(null);
    setError('');

    try {
      const payload = {
        age: parseInt(formData.age, 10),
        gender: formData.gender,
        height: parseFloat(formData.height),
        weight: parseFloat(formData.weight),
        MUAC: parseFloat(formData.MUAC),
        headCircumference: parseFloat(formData.headCircumference)
      };

      if (selectedChildId) {
        payload.childId = selectedChildId;
      }

      const res = await API.post('/predict', payload);
      if (res.data.success) {
        // Fetch detailed nutrition guidelines matching growth status
        const pred = res.data.data;
        setResult(pred);
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to generate prediction analysis.');
    } finally {
      setSubmitting(false);
    }
  };

  const getStatusIcon = (status) => {
    if (status === 'Healthy') {
      return <ShieldCheck size={48} style={{ color: 'var(--success)' }} />;
    }
    return <AlertOctagon size={48} style={{ color: 'var(--danger)' }} />;
  };

  const getZscoreClass = (zscore) => {
    if (zscore < -3) return 'z-score-severe';
    if (zscore < -2) return 'z-score-moderate';
    return 'z-score-normal';
  };

  const getZScoreInfo = (key, value) => {
    const label = key === 'HAZ' ? 'Height-for-Age (Stunting)' : key === 'WAZ' ? 'Weight-for-Age (Underweight)' : 'BMI-for-Age (Wasting)';
    return (
      <div key={key} className={`z-score-item ${getZscoreClass(value)}`} style={{
        padding: '16px',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--gray-200)',
        display: 'flex',
        flexDirection: 'column',
        gap: '6px',
        flex: 1
      }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--gray-500)', fontWeight: 'bold' }}>{label}</span>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <strong style={{ fontSize: '1.4rem' }}>{value} SD</strong>
          <span style={{ 
            fontSize: '0.75rem', 
            padding: '2px 8px', 
            borderRadius: '4px',
            fontWeight: '700',
            backgroundColor: value < -3 ? 'var(--danger-light)' : value < -2 ? 'var(--warning-light)' : 'var(--success-light)',
            color: value < -3 ? 'var(--danger)' : value < -2 ? 'var(--warning)' : 'var(--success)'
          }}>
            {value < -3 ? 'Severe Deviation' : value < -2 ? 'Moderate Deviation' : 'Standard Norm'}
          </span>
        </div>
      </div>
    );
  };

  if (loading) {
    return <Loader fullScreen />;
  }

  return (
    <div className="predictor-wrapper animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Title block */}
      <div style={{ 
        backgroundColor: 'var(--white)', 
        borderRadius: 'var(--radius-lg)', 
        padding: '24px', 
        boxShadow: 'var(--shadow-sm)'
      }}>
        <h1 style={{ fontSize: '1.6rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Sparkles className="text-primary" /> AI Malnutrition Predictor Calculator
        </h1>
        <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem', margin: '4px 0 0 0' }}>
          Run an instant, interactive health assessment matching heights and weights against global WHO standards.
        </p>
      </div>

      {error && <Alert type="danger" message={error} />}

      <div className="grid-2" style={{ gridTemplateColumns: '1fr 2fr', alignItems: 'flex-start' }}>
        
        {/* Left: Input Form */}
        <Card title="Assessment Parameters">
          <form onSubmit={handleSubmit} className="auth-form" style={{ padding: 0 }}>
            
            <div className="form-group">
              <label className="form-label">Pre-select Child (Optional)</label>
              <select
                value={selectedChildId}
                onChange={handleChildChange}
                className="form-input"
                disabled={submitting}
              >
                <option value="">-- Diagnostic Calculator Mode --</option>
                {children.map((c) => (
                  <option key={c._id} value={c._id}>{c.childName} ({c.gender}, {c.motherName})</option>
                ))}
              </select>
            </div>

            <hr style={{ border: 'none', borderTop: '1px solid var(--gray-100)', margin: '10px 0 20px 0' }} />

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Age (Months)</label>
                <input
                  type="number"
                  name="age"
                  className="form-input"
                  required
                  placeholder="e.g. 18"
                  value={formData.age}
                  onChange={handleInputChange}
                  disabled={submitting}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Gender</label>
                <select
                  name="gender"
                  className="form-input"
                  value={formData.gender}
                  onChange={handleInputChange}
                  disabled={submitting || !!selectedChildId}
                >
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                </select>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Height (cm)</label>
                <input
                  type="number"
                  step="0.1"
                  name="height"
                  className="form-input"
                  required
                  placeholder="e.g. 82.5"
                  value={formData.height}
                  onChange={handleInputChange}
                  disabled={submitting}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Weight (kg)</label>
                <input
                  type="number"
                  step="0.01"
                  name="weight"
                  className="form-input"
                  required
                  placeholder="e.g. 10.5"
                  value={formData.weight}
                  onChange={handleInputChange}
                  disabled={submitting}
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">MUAC (cm)</label>
                <input
                  type="number"
                  step="0.1"
                  name="MUAC"
                  className="form-input"
                  required
                  placeholder="e.g. 12.5"
                  value={formData.MUAC}
                  onChange={handleInputChange}
                  disabled={submitting}
                />
              </div>
              <div className="form-group">
                <label className="form-label">Head Circ (cm)</label>
                <input
                  type="number"
                  step="0.1"
                  name="headCircumference"
                  className="form-input"
                  required
                  placeholder="e.g. 44.0"
                  value={formData.headCircumference}
                  onChange={handleInputChange}
                  disabled={submitting}
                />
              </div>
            </div>

            <button type="submit" className="btn btn-primary" style={{ width: '100%', marginTop: '10px' }} disabled={submitting}>
              {submitting ? 'Generating AI Diagnosis...' : 'Diagnose Health Metrics'}
            </button>
          </form>
        </Card>

        {/* Right: Results Panel */}
        <div>
          {submitting ? (
            <Card>
              <div style={{ textAlign: 'center', padding: '60px 0' }}>
                <Loader size="md" />
                <p style={{ marginTop: '16px', color: 'var(--gray-500)' }}>Recalculating WHO growth curve margins...</p>
              </div>
            </Card>
          ) : result ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {/* Core result card */}
              <Card>
                <div style={{ 
                  display: 'flex', 
                  gap: '20px', 
                  alignItems: 'center',
                  padding: '10px 0'
                }}>
                  {getStatusIcon(result.predictionResult)}
                  <div>
                    <h3 style={{ margin: 0, fontSize: '0.85rem', color: 'var(--gray-500)', textTransform: 'uppercase', fontWeight: 'bold' }}>
                      Diagnostic Result
                    </h3>
                    <h2 style={{ fontSize: '1.6rem', color: result.predictionResult === 'Healthy' ? 'var(--success)' : 'var(--danger)', margin: '2px 0' }}>
                      {result.predictionResult}
                    </h2>
                    <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--gray-500)' }}>
                      System Confidence Index: <strong>{result.confidence}%</strong>
                    </p>
                  </div>
                </div>
              </Card>

              {/* Z-Scores Grid */}
              <Card title="Calculated WHO SD Z-Scores">
                <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
                  {Object.keys(result.zScores).map((key) => getZScoreInfo(key, result.zScores[key]))}
                </div>
                <div style={{ display: 'flex', gap: '6px', alignItems: 'center', marginTop: '16px', color: 'var(--gray-500)', fontSize: '0.78rem' }}>
                  <HelpCircle size={14} />
                  <span>Standard WHO bounds: Standard Norm is between -2 and +2. Lower than -2 signals malnutrition risk.</span>
                </div>
              </Card>

              {/* Dietary plan & recommendations */}
              <div className="grid-2">
                <Card title="Dietary & Micronutrients guide">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '0.85rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span className="text-gray-500">Target Calories:</span>
                      <strong>{result.predictionResult === 'Severe Malnutrition' ? '1800 kcal/day' : result.predictionResult === 'Moderate Malnutrition' ? '1500 kcal/day' : '1200 kcal/day'}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span className="text-gray-500">Target Protein:</span>
                      <strong>{result.predictionResult === 'Severe Malnutrition' ? '45 grams/day' : result.predictionResult === 'Moderate Malnutrition' ? '32 grams/day' : '22 grams/day'}</strong>
                    </div>
                    <hr style={{ border: 'none', borderTop: '1px solid var(--gray-100)' }} />
                    <div>
                      <span className="text-gray-500" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold', marginBottom: '6px' }}>Vitamins Supplements:</span>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {(result.predictionResult === 'Severe Malnutrition' ? ['Vitamin A', 'Zinc', 'Iron', 'Multivitamins', 'Folic Acid'] : result.predictionResult === 'Moderate Malnutrition' ? ['Vitamin A', 'Zinc', 'Vitamin C', 'Calcium'] : ['Vitamin D', 'Calcium (dietary)']).map((vit, idx) => (
                          <span key={idx} style={{ fontSize: '0.72rem', padding: '2px 8px', borderRadius: '4px', backgroundColor: 'var(--gray-100)', fontWeight: '700' }}>
                            {vit}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </Card>

                <Card title="Clinical Guidelines">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '0.85rem' }}>
                    <div>
                      <span className="text-gray-500" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold' }}>Therapeutic Strategy:</span>
                      <p style={{ margin: '4px 0 0 0', lineHeight: '1.4', color: 'var(--dark-light)' }}>
                        {result.recommendation}
                      </p>
                    </div>
                    <div>
                      <span className="text-gray-500" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold' }}>Follow-up Advice:</span>
                      <p style={{ margin: '4px 0 0 0', lineHeight: '1.4', color: 'var(--dark-light)', fontStyle: 'italic' }}>
                        {result.followUpSuggestion}
                      </p>
                    </div>
                  </div>
                </Card>
              </div>

              {/* Meal Plan detailed block */}
              <Card title="Custom Daily Feeding Menu Recommendations">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  {(result.predictionResult === 'Severe Malnutrition' ? [
                    'Breakfast: RUTF (1 sachet) + Clean drinking water',
                    'Mid-Morning: Fortified milk (F-75/F-100) or high energy gruel',
                    'Lunch: RUTF (1 sachet) + mineral-supplemented water',
                    'Afternoon: RUTF (1 sachet)',
                    'Dinner: High-density vegetable purée with added oil, milk powder'
                  ] : result.predictionResult === 'Moderate Malnutrition' ? [
                    'Breakfast: Banana porridge with whole milk, 1 boiled egg',
                    'Snack: Peanut butter sandwich or high-protein biscuits',
                    'Lunch: Rice with thick split-pea soup (dal), mashed potato, vegetables cooked with vegetable oil',
                    'Snack: Mixed seasonal fruit salad or yogurt',
                    'Dinner: Minced meat/chicken or tofu stew with rice, green leafy vegetables'
                  ] : [
                    'Breakfast: Oatmeal with chopped apples and honey, cup of warm milk',
                    'Snack: Fresh orange slices or apple wedges',
                    'Lunch: Rice/roti with seasonal vegetable stir-fry, bowl of yellow dal',
                    'Snack: Handful of roasted chickpeas or plain yogurt',
                    'Dinner: Vegetable soup, baked fish or tofu with stir-fried greens'
                  ]).map((meal, idx) => {
                    const [title, desc] = meal.split(': ');
                    return (
                      <div key={idx} style={{ display: 'flex', gap: '10px', fontSize: '0.85rem' }}>
                        <span style={{ color: 'var(--secondary)', fontWeight: 'bold', width: '90px', flexShrink: 0 }}>{title}</span>
                        <span style={{ color: 'var(--dark-light)' }}>{desc}</span>
                      </div>
                    );
                  })}
                </div>
              </Card>
            </div>
          ) : (
            <div style={{ 
              backgroundColor: 'var(--white)', 
              borderRadius: 'var(--radius-lg)', 
              padding: '80px 20px', 
              textAlign: 'center',
              boxShadow: 'var(--shadow-sm)',
              border: '1px dashed var(--gray-200)'
            }}>
              <Activity size={48} className="text-gray-300" style={{ marginBottom: '16px' }} />
              <h3>Awaiting Assessment Parameters</h3>
              <p style={{ color: 'var(--gray-500)', marginTop: '8px' }}>
                Fill in the height, weight, and age details on the left panel, and submit parameters to evaluate.
              </p>
            </div>
          )}
        </div>

      </div>

    </div>
  );
};

export default Predictor;
