import React, { useState } from 'react';
import ChildForm from './components/ChildForm';
import ResultsDashboard from './components/ResultsDashboard';
import ModelComparison from './components/ModelComparison';
import { Activity, Shield, Cpu, RefreshCw, AlertCircle } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('screening'); // 'screening' | 'comparison'
  const [selectedModel, setSelectedModel] = useState('FT-Transformer');
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [lastSubmittedChild, setLastSubmittedChild] = useState(null);
  const [apiError, setApiError] = useState(null);

  const handleFormSubmit = async (formData) => {
    setLoading(true);
    setApiError(null);
    setPredictionResult(null);
    setLastSubmittedChild(formData);

    const endpoint = selectedModel === 'XGBoost'
      ? 'http://localhost:3000/api/prediction/xgboost'
      : 'http://localhost:3000/api/prediction/transformer';

    try {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || errData.message || 'The AI prediction service encountered an issue.');
      }

      const result = await response.json();
      setPredictionResult(result);
    } catch (err) {
      setApiError(err.message || 'Unable to connect to prediction backend server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      
      {/* Top Header Navbar */}
      <header className="navbar">
        <div className="brand-logo">
          <Shield size={26} color="#6366f1" />
          <span>NutriPredict AI — Child Health Screening</span>
        </div>

        <nav className="nav-tabs">
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'screening' ? 'active' : ''}`}
            onClick={() => setActiveTab('screening')}
          >
            <Activity size={16} style={{ display: 'inline', marginRight: '0.4rem' }} />
            Child Health Assessment
          </button>
          <button
            type="button"
            className={`nav-tab-btn ${activeTab === 'comparison' ? 'active' : ''}`}
            onClick={() => setActiveTab('comparison')}
          >
            <Cpu size={16} style={{ display: 'inline', marginRight: '0.4rem' }} />
            XGBoost vs Transformer Comparison
          </button>
        </nav>
      </header>

      {/* Main Content Area */}
      <main style={{ flex: 1, padding: '2rem 1.5rem', maxWidth: '1200px', margin: '0 auto', width: '100%' }}>
        {activeTab === 'screening' && (
          <div>
            {!predictionResult && (
              <ChildForm
                onSubmit={handleFormSubmit}
                loading={loading}
                selectedModel={selectedModel}
                setSelectedModel={setSelectedModel}
              />
            )}

            {apiError && (
              <div className="glass-panel" style={{ padding: '1.5rem', marginTop: '1.5rem', borderColor: 'var(--risk-high)', display: 'flex', alignItems: 'center', gap: '1rem' }}>
                <AlertCircle size={28} color="var(--risk-high)" />
                <div>
                  <h4 style={{ color: 'var(--risk-high)', fontWeight: 700 }}>Assessment Service Error</h4>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>{apiError}</p>
                </div>
              </div>
            )}

            {predictionResult && (
              <ResultsDashboard
                result={predictionResult}
                childInfo={lastSubmittedChild}
                onReset={() => setPredictionResult(null)}
              />
            )}
          </div>
        )}

        {activeTab === 'comparison' && <ModelComparison />}
      </main>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--border-color)', padding: '1.25rem 2rem', textAlign: 'center', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
        AI-Based Child Malnutrition & Multi-Disease Early Prediction System • Powered by FT-Transformer & XGBoost
      </footer>
    </div>
  );
}
