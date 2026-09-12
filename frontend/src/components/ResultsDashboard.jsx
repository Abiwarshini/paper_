import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert, Cpu, ChevronDown, ChevronUp, FileText, Info, Sparkles, Activity, HeartPulse, ArrowRight } from 'lucide-react';

export default function ResultsDashboard({ result, childInfo, onReset }) {
  const [showTechDetails, setShowTechDetails] = useState(false);
  const [showComparisonTable, setShowComparisonTable] = useState(true);

  if (!result) return null;

  // Determine if this is a dual-model result or single-model result
  const isDual = Boolean(result.xgboost && result.transformer);
  const primaryResult = isDual
    ? (result.best_model === 'FT-Transformer' ? result.transformer : result.xgboost)
    : result;

  const {
    model,
    overall_risk,
    top_prediction,
    top_probability,
    top_disease_risk,
    top_disease_probability,
    nutrition_assessment,
    disease_screening,
    top_factors,
    medical_disclaimer
  } = primaryResult;

  const formattedDate = new Date().toLocaleDateString('en-GB', {
    day: '2-digit',
    month: 'short',
    year: 'numeric'
  });

  const getRiskColor = (risk) => {
    if (!risk) return 'var(--risk-low)';
    const r = risk.toUpperCase();
    if (r.includes('HIGH')) return 'var(--risk-high)';
    if (r.includes('MODERATE')) return 'var(--risk-moderate)';
    return 'var(--risk-low)';
  };

  const getRiskBadgeClass = (risk) => {
    if (!risk) return 'low';
    const r = risk.toUpperCase();
    if (r.includes('HIGH')) return 'high';
    if (r.includes('MODERATE')) return 'moderate';
    return 'low';
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem', animation: 'fadeIn 0.4s ease' }}>
      
      {/* Top Banner & Assessment Summary Header */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '0.35rem' }}>
            <HeartPulse size={24} color="var(--accent-teal)" />
            <h2 style={{ fontSize: '1.5rem', fontWeight: 800, color: 'var(--text-primary)' }}>
              Pediatric Nutrition & Health Screening Assessment
            </h2>
          </div>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
            Child: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.childName || 'Child #1042'}</strong> | 
            Age: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.age_months} months</strong> | 
            Gender: <strong style={{ color: 'var(--text-primary)' }}>{childInfo?.gender || 'Female'}</strong> | 
            Date: {formattedDate}
          </p>
        </div>

        {/* Overall Risk Score Badge */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(255,255,255,0.03)', padding: '0.75rem 1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
          <div>
            <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: 700 }}>
              Overall Screening Risk
            </div>
            <div style={{ fontSize: '1.5rem', fontWeight: 900, color: getRiskColor(overall_risk) }}>
              {overall_risk || 'LOW'} RISK
            </div>
          </div>
          {(overall_risk || '').toUpperCase() === 'HIGH' ? (
            <ShieldAlert size={36} color="var(--risk-high)" />
          ) : (overall_risk || '').toUpperCase() === 'MODERATE' ? (
            <AlertTriangle size={36} color="var(--risk-moderate)" />
          ) : (
            <CheckCircle size={36} color="var(--risk-low)" />
          )}
        </div>
      </div>

      {/* Dual Model Winner Banner (if dual prediction was run) */}
      {isDual && (
        <div className="glass-panel" style={{ padding: '1.25rem 1.75rem', background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(20, 184, 166, 0.1))', border: '1px solid rgba(99, 102, 241, 0.3)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Sparkles size={24} color="#a5b4fc" />
              <div>
                <span style={{ fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--accent-teal)' }}>
                  Automated Dual-Model Consensus Evaluation
                </span>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)', marginTop: '0.15rem' }}>
                  Best Performing Model Selected: <span style={{ color: 'var(--accent-teal)' }}>{result.best_model}</span>
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  {result.best_model_reason}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: '1rem' }}>
              <div style={{ textAlign: 'right', padding: '0.5rem 1rem', background: 'rgba(255,255,255,0.04)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>XGBoost Baseline</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#a5b4fc' }}>
                  {result.xgboost?.top_prediction}: {(result.xgboost?.top_probability * 100).toFixed(1)}%
                </div>
              </div>
              <div style={{ textAlign: 'right', padding: '0.5rem 1rem', background: 'rgba(255,255,255,0.04)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>FT-Transformer</div>
                <div style={{ fontSize: '1.1rem', fontWeight: 800, color: '#2dd4bf' }}>
                  {result.transformer?.top_prediction}: {(result.transformer?.top_probability * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Side-by-Side Model Comparison Table (when Dual Model enabled) */}
      {isDual && result.comparison && (
        <div className="glass-panel" style={{ padding: '1.5rem 2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={18} color="var(--accent-teal)" />
              Side-by-Side Model Prediction Comparison (10 Conditions)
            </h3>
            <button
              type="button"
              onClick={() => setShowComparisonTable(!showComparisonTable)}
              style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
            >
              {showComparisonTable ? 'Collapse Table' : 'Expand Table'}
              {showComparisonTable ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
          </div>

          {showComparisonTable && (
            <div style={{ overflowX: 'auto' }}>
              <table className="comparison-table" style={{ width: '100%' }}>
                <thead>
                  <tr>
                    <th>Category</th>
                    <th>Condition Target</th>
                    <th>XGBoost Probability</th>
                    <th>FT-Transformer Probability</th>
                    <th>Risk Level</th>
                    <th>Delta (Diff)</th>
                  </tr>
                </thead>
                <tbody>
                  {result.comparison.map((row, idx) => (
                    <tr key={idx}>
                      <td style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{row.category}</td>
                      <td style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{row.condition}</td>
                      <td style={{ fontWeight: 700, color: '#a5b4fc' }}>{row.xgboost_percentage}%</td>
                      <td style={{ fontWeight: 700, color: '#2dd4bf' }}>{row.transformer_percentage}%</td>
                      <td>
                        <span className={`risk-badge ${getRiskBadgeClass(row.xgboost_risk)}`}>
                          {row.xgboost_risk}
                        </span>
                      </td>
                      <td style={{ color: Math.abs(row.delta) <= 3 ? 'var(--text-muted)' : (row.delta > 0 ? '#2dd4bf' : '#a5b4fc'), fontWeight: 600 }}>
                        {row.delta > 0 ? `+${row.delta}%` : `${row.delta}%`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* SECTION 1: Growth & Nutrition Assessment (5 Indicators) */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={20} color="var(--accent-primary)" />
              Section 1 — Growth & Nutrition Assessment (5 Indicators)
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Evaluated based on child anthropometrics and WHO Child Growth Standards (HAZ, WHZ, WAZ).
            </p>
          </div>
          {top_prediction && (
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Primary Growth Finding: <strong style={{ color: 'var(--text-primary)' }}>{top_prediction}</strong> ({((top_probability || 0) * 100).toFixed(1)}%)
            </span>
          )}
        </div>

        <div className="cards-grid">
          {(nutrition_assessment || primaryResult.predictions?.slice(0, 5))?.map((pred, idx) => {
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
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    threshold: {((pred.threshold || 0.5) * 100).toFixed(0)}%
                  </span>
                </div>

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

      {/* SECTION 2: Pediatric Disease & Health-Condition Risk Screening (5 Targets) */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '1rem' }}>
          <div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 800, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <HeartPulse size={20} color="var(--accent-teal)" />
              Section 2 — Pediatric Disease & Health-Condition Risk Screening (5 Targets)
            </h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Biomarker and clinical nutritional deficiency risk screening (Anemia, IDA, Vitamin A, PEM, Micronutrients).
            </p>
          </div>
          {top_disease_risk && (
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Primary Disease Risk: <strong style={{ color: 'var(--text-primary)' }}>{top_disease_risk}</strong> ({((top_disease_probability || 0) * 100).toFixed(1)}%)
            </span>
          )}
        </div>

        <div className="cards-grid">
          {(disease_screening || primaryResult.predictions?.slice(5, 10))?.map((pred, idx) => {
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
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    threshold: {((pred.threshold || 0.4) * 100).toFixed(0)}%
                  </span>
                </div>

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
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Info size={18} color="var(--accent-teal)" />
          Model Explainability — Top Contributing Biomarkers & Features
        </h3>
        <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '1.25rem' }}>
          Extracted directly via <strong>{isDual ? 'XGBoost SHAP Tree Attributions & Transformer Gradients' : (primaryResult.model === 'XGBoost' ? 'SHAP Feature Importances' : 'Input Gradient Attributions')}</strong>:
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

      {/* Supportive Clinical & Nutrition Recommendations */}
      <div className="glass-panel" style={{ padding: '1.5rem 2rem', background: 'linear-gradient(135deg, rgba(20, 184, 166, 0.08), rgba(99, 102, 241, 0.05))', border: '1px solid rgba(20, 184, 166, 0.3)' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-teal)', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sparkles size={18} color="var(--accent-teal)" />
          Supportive Guidance & Nutrition Decision Support
        </h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: 'var(--radius-sm)' }}>
            <strong style={{ color: 'var(--text-primary)', fontSize: '0.9rem', display: 'block', marginBottom: '0.35rem' }}>
              Dietary Intervention
            </strong>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              Increase dietary diversity to &ge;5 food groups per WHO IYCF guidelines. Introduce iron-rich complementary foods (pulses, greens, animal-source foods) and Vitamin A sources.
            </p>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: 'var(--radius-sm)' }}>
            <strong style={{ color: 'var(--text-primary)', fontSize: '0.9rem', display: 'block', marginBottom: '0.35rem' }}>
              Growth Monitoring
            </strong>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              Schedule fortnightly growth monitoring at the local Anganwadi/ICDS center to track height-for-age and weight-for-height trajectory.
            </p>
          </div>
          <div style={{ background: 'rgba(255,255,255,0.03)', padding: '1rem', borderRadius: 'var(--radius-sm)' }}>
            <strong style={{ color: 'var(--text-primary)', fontSize: '0.9rem', display: 'block', marginBottom: '0.35rem' }}>
              Healthcare Referral
            </strong>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              For moderate or high anemia/PEM risk, prompt clinical evaluation by an Auxiliary Nurse Midwife (ANM) or pediatrician for IFA supplementation and deworming.
            </p>
          </div>
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
            View Technical Architecture & Data Specifications
          </span>
          {showTechDetails ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>

        {showTechDetails && (
          <div style={{ marginTop: '1.25rem', paddingTop: '1rem', borderTop: '1px solid var(--border-color)', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Models Implemented:</strong> XGBoost & FT-Transformer
              </div>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Conditions Evaluated:</strong> 10 Targets (5 Nutrition + 5 Disease Risk)
              </div>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Leakage Protection:</strong> Direct target z-scores excluded from features
              </div>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Explainability:</strong> SHAP (XGBoost) + Input Gradients (Transformer)
              </div>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Training Samples:</strong> 69,999 records (70% train split)
              </div>
              <div>
                <strong style={{ color: 'var(--text-primary)' }}>Test Benchmark:</strong> 15,000 held-out evaluation samples
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Prominent Medical Disclaimer */}
      <div className="disclaimer-box">
        <AlertTriangle size={26} style={{ flexShrink: 0, color: '#fbbf24' }} />
        <div>
          <strong style={{ color: '#fbbf24', display: 'block', marginBottom: '0.25rem' }}>
            Medical & Clinical Research Disclaimer
          </strong>
          {medical_disclaimer || "AI-based risk screening only — this result is not a medical diagnosis. High-risk results should be reviewed by a qualified healthcare professional."}
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
