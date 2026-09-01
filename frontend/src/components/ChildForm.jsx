import React, { useState, useEffect } from 'react';
import { Activity, Cpu, Sparkles, AlertCircle, Calculator, ChevronRight } from 'lucide-react';

export default function ChildForm({ onSubmit, loading, selectedModel, setSelectedModel }) {
  const [formData, setFormData] = useState({
    childName: 'Child #1042',
    age_months: '24',
    gender: 'Female',
    height_cm: '85.0',
    weight_kg: '11.5',
    muac_cm: '14.0',
    dietary_diversity: '4',
    meal_frequency: '3',
    breastfeeding_status: 'Partial',
    water_sanitation_index: '4'
  });

  const [derived, setDerived] = useState({
    bmi: 0,
    haz: 0,
    waz: 0,
    whz: 0
  });

  const [errors, setErrors] = useState({});

  // Auto-calculate derived indicators (BMI, Z-scores) dynamically
  useEffect(() => {
    const age = parseFloat(formData.age_months) || 0;
    const h = parseFloat(formData.height_cm) || 0;
    const w = parseFloat(formData.weight_kg) || 0;

    if (h > 0 && w > 0) {
      const h_m = h / 100;
      const bmiVal = (w / (h_m * h_m)).toFixed(2);

      const expH = 50.0 + 0.75 * age;
      const expW = 3.3 + 0.25 * age + 0.002 * (age * age);
      const expWH = 0.15 * h - 4.5;

      const hazVal = ((h - expH) / (3.5 + 0.02 * age)).toFixed(2);
      const wazVal = ((w - expW) / (0.8 + 0.05 * age)).toFixed(2);
      const whzVal = ((w - expWH) / (1.0 + 0.02 * (h / 10))).toFixed(2);

      setDerived({
        bmi: bmiVal,
        haz: hazVal,
        waz: wazVal,
        whz: whzVal
      });
    }
  }, [formData.age_months, formData.height_cm, formData.weight_kg]);

  const handleChange = (field, val) => {
    setFormData(prev => ({ ...prev, [field]: val }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: null }));
    }
  };

  const validate = () => {
    const errs = {};
    const age = parseFloat(formData.age_months);
    const h = parseFloat(formData.height_cm);
    const w = parseFloat(formData.weight_kg);
    const muac = parseFloat(formData.muac_cm);

    if (isNaN(age) || age < 0 || age > 120) errs.age_months = 'Age must be between 0 and 120 months.';
    if (isNaN(h) || h < 30 || h > 200) errs.height_cm = 'Height must be between 30 and 200 cm.';
    if (isNaN(w) || w < 1 || w > 100) errs.weight_kg = 'Weight must be between 1 and 100 kg.';
    if (isNaN(muac) || muac < 5 || muac > 25) errs.muac_cm = 'MUAC must be between 5 and 25 cm.';

    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!validate()) return;

    onSubmit({
      ...formData,
      age_months: parseFloat(formData.age_months),
      height_cm: parseFloat(formData.height_cm),
      weight_kg: parseFloat(formData.weight_kg),
      muac_cm: parseFloat(formData.muac_cm),
      dietary_diversity: parseInt(formData.dietary_diversity, 10),
      meal_frequency: parseInt(formData.meal_frequency, 10),
      water_sanitation_index: parseInt(formData.water_sanitation_index, 10),
      bmi: parseFloat(derived.bmi),
      haz: parseFloat(derived.haz),
      waz: parseFloat(derived.waz),
      whz: parseFloat(derived.whz)
    });
  };

  return (
    <form onSubmit={handleSubmit} className="glass-panel" style={{ padding: '2rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.4rem', fontWeight: '700', color: 'var(--text-primary)' }}>
            Child Health Assessment Details
          </h2>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            Enter anthropometric and nutritional metrics for multi-disease AI screening.
          </p>
        </div>

        {/* Model Selection Toggle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(255,255,255,0.05)', padding: '0.35rem 0.5rem', borderRadius: '999px', border: '1px solid var(--border-color)' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', paddingLeft: '0.5rem', fontWeight: 600 }}>
            Model Engine:
          </span>
          <button
            type="button"
            className={`nav-tab-btn ${selectedModel === 'FT-Transformer' ? 'active' : ''}`}
            onClick={() => setSelectedModel('FT-Transformer')}
            style={{ fontSize: '0.8rem', padding: '0.35rem 0.85rem' }}
          >
            <Cpu size={14} style={{ display: 'inline', marginRight: '0.35rem' }} />
            FT-Transformer
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${selectedModel === 'XGBoost' ? 'active' : ''}`}
            onClick={() => setSelectedModel('XGBoost')}
            style={{ fontSize: '0.8rem', padding: '0.35rem 0.85rem' }}
          >
            <Activity size={14} style={{ display: 'inline', marginRight: '0.35rem' }} />
            XGBoost Baseline
          </button>
        </div>
      </div>

      {/* Section 1: Child Information */}
      <div style={{ marginBottom: '1.75rem' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--accent-teal)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          Section 1 — Child Identification
        </h3>
        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Child ID / Name</label>
            <div className="input-wrapper">
              <input
                type="text"
                className="form-input"
                value={formData.childName}
                onChange={e => handleChange('childName', e.target.value)}
                placeholder="e.g. Child #1042"
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">
              Age (Months) <span style={{ color: 'var(--risk-high)' }}>*</span>
            </label>
            <div className="input-wrapper">
              <input
                type="number"
                step="0.1"
                className={`form-input ${errors.age_months ? 'error' : ''}`}
                value={formData.age_months}
                onChange={e => handleChange('age_months', e.target.value)}
                placeholder="0 - 120"
              />
              <span className="input-unit">months</span>
            </div>
            {errors.age_months && <span className="error-text">{errors.age_months}</span>}
          </div>

          <div className="form-group">
            <label className="form-label">Gender / Sex</label>
            <select
              className="form-select"
              value={formData.gender}
              onChange={e => handleChange('gender', e.target.value)}
            >
              <option value="Female">Female</option>
              <option value="Male">Male</option>
            </select>
          </div>
        </div>
      </div>

      {/* Section 2: Growth Measurements */}
      <div style={{ marginBottom: '1.75rem' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--accent-teal)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          Section 2 — Growth & Anthropometric Measurements
        </h3>

        {/* Derived Z-Scores & BMI Ribbon */}
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', background: 'rgba(99, 102, 241, 0.08)', padding: '0.85rem 1.25rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(99, 102, 241, 0.2)', marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Calculator size={16} color="var(--accent-primary)" />
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#a5b4fc' }}>Calculated Indicators:</span>
          </div>
          <span className="calc-badge">BMI: {derived.bmi} kg/m²</span>
          <span className="calc-badge">HAZ (Height-for-Age): {derived.haz}</span>
          <span className="calc-badge">WAZ (Weight-for-Age): {derived.waz}</span>
          <span className="calc-badge">WHZ (Weight-for-Height): {derived.whz}</span>
        </div>

        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">
              Height <span style={{ color: 'var(--risk-high)' }}>*</span>
            </label>
            <div className="input-wrapper">
              <input
                type="number"
                step="0.1"
                className={`form-input ${errors.height_cm ? 'error' : ''}`}
                value={formData.height_cm}
                onChange={e => handleChange('height_cm', e.target.value)}
                placeholder="30 - 200"
              />
              <span className="input-unit">cm</span>
            </div>
            {errors.height_cm && <span className="error-text">{errors.height_cm}</span>}
          </div>

          <div className="form-group">
            <label className="form-label">
              Weight <span style={{ color: 'var(--risk-high)' }}>*</span>
            </label>
            <div className="input-wrapper">
              <input
                type="number"
                step="0.1"
                className={`form-input ${errors.weight_kg ? 'error' : ''}`}
                value={formData.weight_kg}
                onChange={e => handleChange('weight_kg', e.target.value)}
                placeholder="1 - 100"
              />
              <span className="input-unit">kg</span>
            </div>
            {errors.weight_kg && <span className="error-text">{errors.weight_kg}</span>}
          </div>

          <div className="form-group">
            <label className="form-label">
              MUAC (Arm Circumference) <span style={{ color: 'var(--risk-high)' }}>*</span>
            </label>
            <div className="input-wrapper">
              <input
                type="number"
                step="0.1"
                className={`form-input ${errors.muac_cm ? 'error' : ''}`}
                value={formData.muac_cm}
                onChange={e => handleChange('muac_cm', e.target.value)}
                placeholder="8.0 - 20.0"
              />
              <span className="input-unit">cm</span>
            </div>
            {errors.muac_cm && <span className="error-text">{errors.muac_cm}</span>}
          </div>
        </div>
      </div>

      {/* Section 3: Nutrition & Health */}
      <div style={{ marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--accent-teal)', marginBottom: '1rem' }}>
          Section 3 — Nutrition & Health Indicators
        </h3>
        <div className="form-grid">
          <div className="form-group">
            <label className="form-label">Dietary Diversity Score (1 - 7)</label>
            <select
              className="form-select"
              value={formData.dietary_diversity}
              onChange={e => handleChange('dietary_diversity', e.target.value)}
            >
              <option value="1">1 - Very Low (1 food group)</option>
              <option value="2">2 - Low (2 food groups)</option>
              <option value="3">3 - Moderate (3 food groups)</option>
              <option value="4">4 - Adequate (4 food groups)</option>
              <option value="5">5 - Good (5 food groups)</option>
              <option value="6">6 - High (6 food groups)</option>
              <option value="7">7 - Optimal (7 food groups)</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Daily Meal Frequency</label>
            <select
              className="form-select"
              value={formData.meal_frequency}
              onChange={e => handleChange('meal_frequency', e.target.value)}
            >
              <option value="1">1 meal per day</option>
              <option value="2">2 meals per day</option>
              <option value="3">3 meals per day</option>
              <option value="4">4 meals per day</option>
              <option value="5">5+ meals per day</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Breastfeeding Status</label>
            <select
              className="form-select"
              value={formData.breastfeeding_status}
              onChange={e => handleChange('breastfeeding_status', e.target.value)}
            >
              <option value="Exclusive">Exclusive Breastfeeding</option>
              <option value="Partial">Partial Breastfeeding</option>
              <option value="Weaned">Weaned / Solid Foods</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Water & Sanitation Index (1 - 5)</label>
            <select
              className="form-select"
              value={formData.water_sanitation_index}
              onChange={e => handleChange('water_sanitation_index', e.target.value)}
            >
              <option value="1">1 - Poor Sanitation / Unimproved</option>
              <option value="2">2 - Limited Sanitation</option>
              <option value="3">3 - Basic Access</option>
              <option value="4">4 - Good Sanitation</option>
              <option value="5">5 - Safely Managed Water & Sanitation</option>
            </select>
          </div>
        </div>
      </div>

      {/* Submit Button */}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? (
            <>
              <Sparkles size={18} className="spin" />
              Running AI Multi-Disease Screening...
            </>
          ) : (
            <>
              <Sparkles size={18} />
              Run AI Assessment ({selectedModel})
              <ChevronRight size={18} />
            </>
          )}
        </button>
      </div>
    </form>
  );
}
