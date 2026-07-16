import React, { useState, useContext } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { HeartPulse, Mail, Lock, Loader2 } from 'lucide-react';
import Alert from '../components/Alert';
import '../css/Login.css';

const Login = () => {
  const { loginUser } = useContext(AuthContext);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState([]);
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setErrors([]);

    const result = await loginUser(email, password);

    if (result.success) {
      navigate('/dashboard');
    } else {
      setErrors(result.errors || ['Invalid credentials']);
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
          <h2>Welcome Back</h2>
          <p>Sign in to monitor pediatric growth statistics</p>
        </div>

        {errors.length > 0 && (
          <Alert type="danger" message="Login Failed">
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              {errors.map((err, i) => (
                <li key={i}>{err}</li>
              ))}
            </ul>
          </Alert>
        )}

        <form className="auth-form" onSubmit={handleLogin}>
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

          <div className="auth-form-group">
            <div className="label-row">
              <label className="auth-label">Password</label>
              <Link to="/forgot-password" className="forgot-link">Forgot password?</Link>
            </div>
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

          <button type="submit" className="btn btn-primary auth-submit-btn" disabled={loading}>
            {loading ? (
              <>
                <Loader2 className="spinner-icon" size={18} /> Logging In...
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <div className="auth-footer">
          <p>Don't have an account? <Link to="/register">Register here</Link></p>
        </div>
      </div>
    </div>
  );
};

export default Login;
