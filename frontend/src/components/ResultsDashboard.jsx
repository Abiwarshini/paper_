import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert, Cpu, ChevronDown, ChevronUp, FileText, Info } from 'lucide-react';

export default function ResultsDashboard({ result, childInfo, onReset }) {
  const [showTechDetails, setShowTechDetails] = useState(false);

  if (!result) return null;

  const {
    model,
    overall_risk,
    top_prediction,
    top_probability,
    predictions,
    top_factors,
    medical_disclaimer
  } = result;

  const formattedDate = new Date().toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  });

  const getRiskColor = (risk) => {
    if (risk === 'HIGH' || risk === 'High Risk') return 'var(--risk-high)';
    if (risk === 'MODERATE' || risk === 'Moderate Risk') return 'var(--risk-moderate)';
    return 'var(--risk-low)';
  };

  const getRiskBadgeClass = (risk) => {
    if (risk === 'High Risk' || risk === 'HIGH') return 'high';
    if (risk === 'Moderate Risk' || risk === 'MODERATE') return 'moderate';
    return 'low';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', animation: 'fadeIn 0.4s ease' }}>
      
      {/* Top Banner & Assessment Summary Header */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.35rem' }}>
            <Cpu size={22} color="var(--accent-primary)" />
            <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              AI Child Nutrition & Health Assessment
            </h2>
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            Child: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.childName || 'Child #1042'}</strong> | 
            Age: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.age_months} months</strong> | 
            Date: {formattedDate} | 
            Engine: <strong style={{ color: 'var(--accent-teal)' }}>{model}</strong>
          </p>
        </div>

        {/* Overall Risk Score Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
          <div>
            <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>
              Overall Risk Level
            </div>
            <div style={{ fontSize: '1.5rem', fontWeight: 900, color: getRiskColor(overall_risk) }}>
              {overall_risk} RISK
            </div>
          </div>
          {overall_risk === 'HIGH' ? (
            <ShieldAlert size={36} color="var(--risk-high)" />
          ) : overall_risk === 'MODERATE' ? (
            <AlertTriangle size={36} color="var(--risk-moderate)" />
          ) : (
            <CheckCircle size={36} color="var(--risk-low)" />
          )}
        </div>
      </div>

      {/* Primary Finding Card */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem', background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.05))', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
        <h3 style={{ fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--accent-teal)', fontWeight: 800, marginBottom: '0.5rem' }}>
          Primary Finding
        </h3>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <div style={{ fontSize: '1.8rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              {top_prediction}
            </div>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.25rem', maxWidth: '650px' }}>
              The AI model identified <strong>{top_prediction}</strong> as the primary nutritional condition requiring field-worker attention based on the child's anthropometric growth z-scores and dietary indicators.
            </p>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div style={{ fontSize: '2.2rem', fontWeight: 900, color: getRiskColor(top_probability >= 0.6 ? 'High Risk' : top_probability >= 0.3 ? 'Moderate Risk' : 'Low Risk') }}>
              {(top_probability * 100).toFixed(1)}%
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>Risk Probability</div>
          </div>
        </div>
      </div>

      {/* 7 Disease / Condition Prediction Cards */}
      <div>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '1rem' }}>
          Multi-Condition Risk Breakdown (7 Health Indicators)
        </h3>
        <div className="cards-grid">
          {predictions?.map((pred, idx) => {
            const badgeClass = getRiskBadgeClass(pred.risk_level);
            return (
              <div key={idx} className={`glass-panel disease-card risk-${badgeClass}`}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                  <span style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                    {pred.condition}
                  </span>
                  <span className={`risk-badge ${badgeClass}`}>
                    {pred.risk_level}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', margin: '0.5rem 0' }}>
                  <span style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                    {pred.percentage}%
                  </span>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>probability</span>
                </div>

                {/* Animated Progress Bar */}
                <div className="progress-bar-bg">
                  <div
                    className={`progress-bar-fill ${badgeClass}`}
                    style={{ width: `${Math.max(pred.percentage, 5)}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Important Contributing Factors (Explainability) */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Info size={18} color="var(--accent-teal)" />
          Important Contributing Factors (Model Feature Attributions)
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
          The following features had the strongest influence on the AI assessment output:
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>
          {top_factors?.map((factor, idx) => (
            <div key={idx} style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: '0.9rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                  {idx + 1}. {factor.feature}
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
                  {factor.influence}
                </div>
              </div>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, padding: '0.2rem 0.6rem', borderRadius: 'var(--radius-full)', background: 'rgba(99, 102, 241, 0.15)', color: '#a5b4fc', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                {factor.impact}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Expandable Technical Details Section */}
      <div className="glass-panel" style={{ padding: '1rem 1.5rem' }}>
        <button
          type="button"
          onClick={() => setShowTechDetails(!showTechDetails)}
          style={{ width: '100%', background: 'transparent', border: 'none', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', fontSize: '0.9rem', fontWeight: 700 }}
        >
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <FileText size={16} color="var(--accent-primary)" />
            View Model Technical Details & Architecture Specifications
          </span>
          {showTechDetails ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>

        {showTechDetails && (
          <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Architecture:</strong> FT-Transformer (Feature Tokenizer Transformer)
              </div>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Token Embedding Size:</strong> 64 dimensions
              </div>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Transformer Layers:</strong> 3 Self-Attention Layers
              </div>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Classification Type:</strong> Multi-Label Sigmoid Output
              </div>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Training Samples:</strong> 69,999 child records
              </div>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Test Evaluation Set:</strong> 15,000 child records
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Medical Disclaimer */}
      <div className="disclaimer-box">
        <AlertTriangle size={24} style={{ flexShrink: 0 }} />
        <div>
          <strong style={{ color: '#fbbf24', display: 'block', marginBottom: '0.2rem' }}>Medical & Clinical Screening Disclaimer</strong>
          {medical_disclaimer}
        </div>
      </div>

      {/* Back / New Assessment Button */}
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '1rem' }}>
        <button type="button" onClick={onReset} className="btn-primary" style={{ background: 'rgba(255,255,255,0.08)', border: '1px solid var(--border-color)', boxShadow: 'none' }}>
          ← Start New Child Health Screening
        </button>
      </div>

    </div>
  );
}
