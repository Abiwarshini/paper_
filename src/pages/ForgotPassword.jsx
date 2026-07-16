import React, { useState, useContext } from 'react';
import { Link } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { HeartPulse, Mail, Loader2, CheckCircle2, ExternalLink } from 'lucide-react';
import Alert from '../components/Alert';
import '../css/Login.css';

const ForgotPassword = () => {
  const { forgotPassword } = useContext(AuthContext);
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [previewUrl, setPreviewUrl] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess(false);
    setPreviewUrl('');

    const result = await forgotPassword(email);

    if (result.success) {
      setSuccess(true);
      if (result.data?.data?.previewUrl) {
        setPreviewUrl(result.data.data.previewUrl);
      }
    } else {
      setError(result.message || 'Request failed. Make sure email is registered.');
    }
    setLoading(false);
  };

  return (
    <div className="auth-page-container">
      <div className="auth-card animate-fade-in">
        <div className="auth-header">
          <Link to="/" className="auth-brand">
            <HeartPulse className="auth-logo-icon" size={32} />
            <span>NutriGrowth <span className="logo-sub">AI</span></span>
          </Link>
          <h2>Recover Password</h2>
          <p>Enter your registered email to request a reset link</p>
        </div>

        {error && (
          <Alert type="danger" message={error} />
        )}

        {success ? (
          <div className="forgot-success-container" style={{ textAlign: 'center', padding: '10px 0' }}>
            <CheckCircle2 size={48} style={{ color: 'var(--success)', marginBottom: '16px' }} />
            <h3 style={{ marginBottom: '8px' }}>Request Successful</h3>
            <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem', marginBottom: '20px' }}>
              We've dispatched a password reset link to <strong>{email}</strong>.
            </p>

            {previewUrl && (
              <div className="dev-preview-box" style={{ 
                backgroundColor: 'var(--primary-light)', 
                border: '1px dashed var(--primary)', 
                borderRadius: 'var(--radius-md)', 
                padding: '16px', 
                marginBottom: '20px',
                textAlign: 'left'
              }}>
                <h5 style={{ color: 'var(--primary-dark)', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px' }}>
                  Mock Mailer Sandbox Triggered
                </h5>
                <p style={{ fontSize: '0.8rem', color: 'var(--dark-light)', marginBottom: '12px' }}>
                  Since the system is in development mode, the outbound email has been intercepted. You can view the full HTML email and complete the reset flow by clicking below:
                </p>
                <a 
                  href={previewUrl} 
                  target="_blank" 
                  rel="noopener noreferrer" 
                  className="btn btn-primary"
                  style={{ width: '100%', fontSize: '0.8rem', padding: '8px 16px', gap: '6px' }}
                >
                  View Reset Email Inbox <ExternalLink size={14} />
                </a>
              </div>
            )}

            <Link to="/login" className="btn btn-light" style={{ width: '100%' }}>
              Back to Login
            </Link>
          </div>
        ) : (
          <form className="auth-form" onSubmit={handleSubmit}>
            <div className="auth-form-group">
              <label className="auth-label">Email Address</label>
              <div className="auth-input-wrapper">
                <Mail className="auth-input-icon" size={18} />
                <input
                  type="email"
                  className="auth-input"
                  placeholder="doctor@clinic.org"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={loading}
                />
              </div>
            </div>

            <button type="submit" className="btn btn-primary auth-submit-btn" disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="spinner-icon" size={18} /> Sending Email...
                </>
              ) : (
                'Send Reset Link'
              )}
            </button>
          </form>
        )}

        {!success && (
          <div className="auth-footer">
            <p>Remember your password? <Link to="/login">Sign in here</Link></p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ForgotPassword;
