import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert, Cpu, ChevronDown, ChevronUp, Sparkles, Activity, HeartPulse, ArrowRight, Stethoscope, BarChart3 } from 'lucide-react';

export default function ResultsDashboard({ result, childInfo, onReset }) {
  const [selectedModelTab, setSelectedModelTab] = useState('best'); // 'best' | 'xgboost' | 'transformer' | 'dnn' | 'tabnet'

  if (!result) return null;

  const isFourModel = Boolean(result.xgboost && result.transformer && result.dnn && result.tabnet);

  // Active displayed model
  let displayedResult = result;
  if (isFourModel) {
    if (selectedModelTab === 'best') {
      const bestName = (result.best_model || 'XGBoost').toLowerCase();
      displayedResult = result[bestName] || result.xgboost || result.dnn || result.tabnet;
    } else {
      displayedResult = result[selectedModelTab] || result.xgboost;
    }
  }

  const {
    model,
    architecture,
    overall_risk,
    prediction,
    probability,
    percentage,
    conditions = [],
    top_factors = [],
    explainability_method,
    medical_disclaimer
  } = displayedResult;

  const formattedDate = new Date().toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  });

  const getRiskColor = (risk) => {
    if (!risk) return '#22c55e';
    const r = String(risk).toUpperCase();
    if (r.includes('HIGH')) return '#ef4444';
    if (r.includes('MODERATE')) return '#eab308';
    return '#22c55e';
  };

  const getRiskBadgeClass = (risk) => {
    if (!risk) return 'low';
    const r = String(risk).toUpperCase();
    if (r.includes('HIGH')) return 'high';
    if (r.includes('MODERATE')) return 'moderate';
    return 'low';
  };

  const stuntingCondition = conditions.find(c => c.condition === 'Stunting');
  const wastingCondition = conditions.find(c => c.condition === 'Wasting');
  const malnutritionCondition = conditions.find(c => c.condition === 'Malnutrition') || {
    percentage: percentage || (probability ? (probability * 100).toFixed(1) : 50.0),
    severity: overall_risk || 'LOW'
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', animation: 'fadeIn 0.4s ease' }}>
      
      {/* Top Banner & Assessment Summary Header */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.35rem' }}>
            <HeartPulse size={24} color="#6366f1" />
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              AI Malnutrition Risk Assessment
            </h2>
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            Child: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.childName || 'Child #1042'}</strong> | 
            Age: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.child_age_months || childInfo?.age_months} mo</strong> | 
            Sex: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.child_sex || childInfo?.gender || 'Female'}</strong> | 
            Residence: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.residence || 'Rural'}</strong> | 
            Date: {formattedDate}
          </p>
        </div>

        {/* Overall Risk Score Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1.25rem', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)' }}>
          <div>
            <div style={{ fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--text-secondary)', fontWeight: 700 }}>
              Malnutrition Risk Tier
            </div>
            <div style={{ fontSize: '1.4rem', fontWeight: 900, color: getRiskColor(overall_risk) }}>
              {overall_risk || 'LOW'} RISK
            </div>
          </div>
          {(overall_risk || '').toUpperCase() === 'HIGH' ? (
            <ShieldAlert size={36} color="#ef4444" />
          ) : (overall_risk || '').toUpperCase() === 'MODERATE' ? (
            <AlertTriangle size={36} color="#eab308" />
          ) : (
            <CheckCircle size={36} color="#22c55e" />
          )}
        </div>
      </div>

      {/* Model Selection Switcher (when 4-model result is returned) */}
      {isFourModel && (
        <div className="glass-panel" style={{ padding: '1rem 1.5rem', background: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.25)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Sparkles size={20} color="#818cf8" />
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase', color: '#818cf8', letterSpacing: '0.05em' }}>
                4-Model Consensus Suite
              </span>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Viewing predictions from: <strong style={{ color: '#fff' }}>{model}</strong>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
            <button
              type="button"
              className={`btn ${selectedModelTab === 'best' ? 'btn-primary' : 'btn-outline'}`}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}
              onClick={() => setSelectedModelTab('best')}
            >
              ★ Leading Model ({result.best_model || 'XGBoost'})
            </button>
            <button
              type="button"
              className={`btn ${selectedModelTab === 'xgboost' ? 'btn-primary' : 'btn-outline'}`}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}
              onClick={() => setSelectedModelTab('xgboost')}
            >
              XGBoost
            </button>
            <button
              type="button"
              className={`btn ${selectedModelTab === 'transformer' ? 'btn-primary' : 'btn-outline'}`}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}
              onClick={() => setSelectedModelTab('transformer')}
            >
              FT-Transformer
            </button>
            <button
              type="button"
              className={`btn ${selectedModelTab === 'dnn' ? 'btn-primary' : 'btn-outline'}`}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}
              onClick={() => setSelectedModelTab('dnn')}
            >
              DNN
            </button>
            <button
              type="button"
              className={`btn ${selectedModelTab === 'tabnet' ? 'btn-primary' : 'btn-outline'}`}
              style={{ fontSize: '0.8rem', padding: '0.35rem 0.8rem' }}
              onClick={() => setSelectedModelTab('tabnet')}
            >
              TabNet
            </button>
          </div>
        </div>
      )}

      {/* Primary Malnutrition Prediction Card & Details */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}>
        
        {/* Main Result Card */}
        <div className="glass-panel" style={{ padding: '1.75rem', position: 'relative', overflow: 'hidden' }}>
          <div style={{ position: 'absolute', top: '-10px', right: '-10px', width: '90px', height: '90px', background: `radial-gradient(circle, ${getRiskColor(overall_risk)}22 0%, transparent 70%)` }} />
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '1rem' }}>
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Primary Malnutrition Screening
              </span>
              <h3 style={{ fontSize: '1.3rem', fontWeight: 800, marginTop: '0.2rem', color: 'var(--text-primary)' }}>
                {prediction || 'Normal (Well-Nourished)'}
              </h3>
            </div>
            <span className={`badge ${getRiskBadgeClass(overall_risk)}`}>
              {overall_risk || 'LOW'}
            </span>
          </div>

          {/* Probability Gauge Bar */}
          <div style={{ marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.4rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Predicted Probability:</span>
              <strong style={{ color: getRiskColor(overall_risk), fontSize: '1.1rem' }}>
                {malnutritionCondition.percentage}%
              </strong>
            </div>
            <div style={{ width: '100%', height: '10px', background: 'rgba(255,255,255,0.08)', borderRadius: '999px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${Math.min(100, Math.max(5, malnutritionCondition.percentage))}%`,
                  height: '100%',
                  background: `linear-gradient(90deg, #6366f1, ${getRiskColor(overall_risk)})`,
                  borderRadius: '999px',
                  transition: 'width 0.8s ease'
                }}
              />
            </div>
          </div>

          <div style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.06)', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <strong>Model Architecture:</strong> {model} ({architecture || 'Tabular Machine Learning'})
          </div>
        </div>

        {/* Stunting & Wasting Breakdown Card */}
        <div className="glass-panel" style={{ padding: '1.75rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Activity size={18} color="#06b6d4" />
            WHO Anthropometric Condition Breakdown
          </h3>

          {/* Stunting Card */}
          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)', marginBottom: '0.75rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <div>
                <strong style={{ fontSize: '0.95rem' }}>Stunting (Height-for-Age Deficit)</strong>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Chronic nutritional deprivation (HAZ &le; -2.0 SD)</div>
              </div>
              <span className={`badge ${getRiskBadgeClass(stuntingCondition?.severity || 'low')}`}>
                {stuntingCondition?.severity || 'LOW'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Risk Probability:</span>
              <strong style={{ color: getRiskColor(stuntingCondition?.severity) }}>{stuntingCondition?.percentage || 0}%</strong>
            </div>
          </div>

          {/* Wasting Card */}
          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '10px', border: '1px solid rgba(255,255,255,0.06)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.4rem' }}>
              <div>
                <strong style={{ fontSize: '0.95rem' }}>Wasting (Weight-for-Height Deficit)</strong>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Acute nutritional deficiency (WHZ &le; -2.0 SD)</div>
              </div>
              <span className={`badge ${getRiskBadgeClass(wastingCondition?.severity || 'low')}`}>
                {wastingCondition?.severity || 'LOW'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem' }}>
              <span style={{ color: 'var(--text-secondary)' }}>Risk Probability:</span>
              <strong style={{ color: getRiskColor(wastingCondition?.severity) }}>{wastingCondition?.percentage || 0}%</strong>
            </div>
          </div>
        </div>

      </div>

      {/* Model Explainability & Feature Contribution */}
      <div className="glass-panel" style={{ padding: '1.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <h3 style={{ fontSize: '1.15rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <BarChart3 size={20} color="#6366f1" />
              Model Explainability & Key Driving Factors
            </h3>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Attribution method: <strong style={{ color: '#a5b4fc' }}>{explainability_method || 'Feature Importance Analysis'}</strong>
            </p>
          </div>
        </div>

        {top_factors && top_factors.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            {top_factors.map((factor, idx) => (
              <div
                key={idx}
                style={{
                  padding: '1rem',
                  background: 'rgba(255,255,255,0.02)',
                  borderRadius: '10px',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderLeft: factor.impact_level === 'High Impact' ? '4px solid #ef4444' : (factor.impact_level === 'Moderate Impact' ? '4px solid #eab308' : '4px solid #6366f1')
                }}
              >
                <div style={{ fontSize: '0.85rem', fontWeight: 700, marginBottom: '0.35rem' }}>
                  {factor.feature}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.8rem' }}>
                  <span style={{ color: 'var(--text-secondary)' }}>Relative Weight:</span>
                  <strong style={{ color: '#fff' }}>{factor.contribution_pct}%</strong>
                </div>
                <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: factor.impact_level === 'High Impact' ? '#ef4444' : (factor.impact_level === 'Moderate Impact' ? '#eab308' : '#818cf8'), fontWeight: 600 }}>
                  ● {factor.impact_level}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>No feature importance data returned for this case.</p>
        )}
      </div>

      {/* Mandatory Medical Safety Disclaimer */}
      <div className="glass-panel" style={{ padding: '1.25rem 1.75rem', background: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.25)', display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
        <Stethoscope size={24} color="#ef4444" style={{ flexShrink: 0, marginTop: '0.2rem' }} />
        <div>
          <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#ef4444', marginBottom: '0.25rem' }}>
            Clinical Safety & Ethical Disclaimer
          </h4>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            This system provides <strong>AI-based malnutrition risk screening for academic and research purposes</strong> and is <strong>not a medical diagnosis</strong>. 
            Any child flagged with Moderate or High risk should immediately be referred to a qualified pediatrician or public health worker (ANM/ICDS) for clinical measurement and intervention.
          </p>
        </div>
      </div>

      {/* Action Footer */}
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem' }}>
        <button
          type="button"
          className="btn btn-outline"
          onClick={onReset}
        >
          Screen Another Child
        </button>
      </div>

    </div>
  );
}
