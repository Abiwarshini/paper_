import React, { useState, useContext } from 'react';
import { Link, useParams, useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { HeartPulse, Lock, Loader2, CheckCircle } from 'lucide-react';
import Alert from '../components/Alert';
import '../css/Login.css';

const ResetPassword = () => {
  const { resetPassword } = useContext(AuthContext);
  const { token } = useParams();
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (password.length < 6) {
      setError('Password must be at least 6 characters');
      return;
    }

    setLoading(true);

    const result = await resetPassword(token, password);

    if (result.success) {
      setSuccess(true);
      setTimeout(() => {
        navigate('/dashboard');
      }, 3000);
    } else {
      setError(result.message || 'Failed to reset password. The link may have expired.');
      setLoading(false);
    }
  };

  return (
    <div className="auth-page-container">
      <div className="auth-card animate-fade-in">
        <div className="auth-header">
          <Link to="/" className="auth-brand">
            <HeartPulse className="auth-logo-icon" size={32} />
            <span>NutriGrowth <span className="logo-sub">AI</span></span>
          </Link>
          <h2>Reset Password</h2>
          <p>Set a new, secure password for your account</p>
        </div>

        {error && (
          <Alert type="danger" message={error} />
        )}

        {success ? (
          <div className="forgot-success-container" style={{ textAlign: 'center', padding: '10px 0' }}>
            <CheckCircle size={48} style={{ color: 'var(--success)', marginBottom: '16px' }} />
            <h3 style={{ marginBottom: '8px' }}>Password Updated!</h3>
            <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem' }}>
              Your password has been changed successfully. Logging you in now...
            </p>
          </div>
        ) : (
          <form className="auth-form" onSubmit={handleSubmit}>
            <div className="auth-form-group">
              <label className="auth-label">New Password</label>
              <div className="auth-input-wrapper">
                <Lock className="auth-input-icon" size={18} />
                <input
                  type="password"
                  className="auth-input"
                  placeholder="••••••••"
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={loading}
                />
              </div>
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Confirm New Password</label>
              <div className="auth-input-wrapper">
                <Lock className="auth-input-icon" size={18} />
                <input
                  type="password"
                  className="auth-input"
                  placeholder="••••••••"
                  required
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  disabled={loading}
                />
              </div>
            </div>

            <button type="submit" className="btn btn-primary auth-submit-btn" disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="spinner-icon" size={18} /> Updating Password...
                </>
              ) : (
                'Save Password'
              )}
            </button>
          </form>
        )}
      </div>
    </div>
  );
};

export default ResetPassword;
