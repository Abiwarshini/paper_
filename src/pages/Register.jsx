import React, { useState, useContext } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { HeartPulse, User, Mail, Lock, Phone, ShieldAlert, FileImage, Loader2 } from 'lucide-react';
import Alert from '../components/Alert';
import '../css/Login.css';

const Register = () => {
  const { registerUser } = useContext(AuthContext);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    role: 'Parent',
    phone: ''
  });
  const [profileImage, setProfileImage] = useState(null);
  const [imagePreview, setImagePreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState([]);
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setProfileImage(file);
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrors([]);

    if (formData.password !== formData.confirmPassword) {
      setErrors(['Passwords do not match']);
      return;
    }

    setLoading(true);

    const submissionData = new FormData();
    submissionData.append('name', formData.name);
    submissionData.append('email', formData.email);
    submissionData.append('password', formData.password);
    submissionData.append('role', formData.role);
    submissionData.append('phone', formData.phone);
    if (profileImage) {
      submissionData.append('profileImage', profileImage);
    }

    const result = await registerUser(submissionData);

    if (result.success) {
      navigate('/dashboard');
    } else {
      setErrors(result.errors || ['Registration failed']);
      setLoading(false);
    }
  };

  return (
    <div className="auth-page-container">
      <div className="auth-card animate-fade-in" style={{ maxWidth: '550px' }}>
        <div className="auth-header">
          <Link to="/" className="auth-brand">
            <HeartPulse className="auth-logo-icon" size={32} />
            <span>NutriGrowth <span className="logo-sub">AI</span></span>
          </Link>
          <h2>Create Account</h2>
          <p>Register to log children measurements and monitor health status</p>
        </div>

        {errors.length > 0 && (
          <Alert type="danger" message="Registration Failed">
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              {errors.map((err, i) => (
                <li key={i}>{err}</li>
              ))}
            </ul>
          </Alert>
        )}

        <form className="auth-form" onSubmit={handleSubmit}>
          <div className="auth-form-group">
            <label className="auth-label">Full Name</label>
            <div className="auth-input-wrapper">
              <User className="auth-input-icon" size={18} />
              <input
                type="text"
                name="name"
                className="auth-input"
                placeholder="Dr. Sarah Connor"
                required
                value={formData.name}
                onChange={handleChange}
                disabled={loading}
              />
            </div>
          </div>

          <div className="auth-form-group">
            <label className="auth-label">Email Address</label>
            <div className="auth-input-wrapper">
              <Mail className="auth-input-icon" size={18} />
              <input
                type="email"
                name="email"
                className="auth-input"
                placeholder="s.connor@clinic.org"
                required
                value={formData.email}
                onChange={handleChange}
                disabled={loading}
              />
            </div>
          </div>

          <div className="form-row">
            <div className="auth-form-group">
              <label className="auth-label">Designated Role</label>
              <div className="auth-input-wrapper">
                <ShieldAlert className="auth-input-icon" size={18} />
                <select
                  name="role"
                  className="auth-input"
                  value={formData.role}
                  onChange={handleChange}
                  disabled={loading}
                  style={{ appearance: 'none', background: 'var(--white)' }}
                >
                  <option value="Parent">Parent</option>
                  <option value="Health Worker">Health Worker</option>
                  <option value="Doctor">Doctor</option>
                  <option value="Admin">Admin</option>
                </select>
              </div>
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Phone Number</label>
              <div className="auth-input-wrapper">
                <Phone className="auth-input-icon" size={18} />
                <input
                  type="tel"
                  name="phone"
                  className="auth-input"
                  placeholder="+91 9876543210"
                  required
                  value={formData.phone}
                  onChange={handleChange}
                  disabled={loading}
                />
              </div>
            </div>
          </div>

          <div className="form-row">
            <div className="auth-form-group">
              <label className="auth-label">Password</label>
              <div className="auth-input-wrapper">
                <Lock className="auth-input-icon" size={18} />
                <input
                  type="password"
                  name="password"
                  className="auth-input"
                  placeholder="••••••••"
                  required
                  value={formData.password}
                  onChange={handleChange}
                  disabled={loading}
                />
              </div>
            </div>

            <div className="auth-form-group">
              <label className="auth-label">Confirm Password</label>
              <div className="auth-input-wrapper">
                <Lock className="auth-input-icon" size={18} />
                <input
                  type="password"
                  name="confirmPassword"
                  className="auth-input"
                  placeholder="••••••••"
                  required
                  value={formData.confirmPassword}
                  onChange={handleChange}
                  disabled={loading}
                />
              </div>
            </div>
          </div>

          <div className="auth-form-group">
            <label className="auth-label">Profile Image (Optional)</label>
            <div className="profile-upload-container" style={{ display: 'flex', gap: '16px', alignItems: 'center', marginTop: '4px' }}>
              <div className="profile-preview-wrapper" style={{ 
                width: '60px', 
                height: '60px', 
                borderRadius: '50%', 
                border: '2px solid var(--gray-200)',
                backgroundColor: 'var(--gray-100)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden'
              }}>
                {imagePreview ? (
                  <img src={imagePreview} alt="Preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <FileImage size={24} className="text-gray-400" />
                )}
              </div>
              <div className="file-input-wrapper" style={{ position: 'relative', flex: 1 }}>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  disabled={loading}
                  style={{ display: 'none' }}
                  id="profile-img-upload"
                />
                <label htmlFor="profile-img-upload" className="btn btn-light" style={{ cursor: 'pointer', padding: '10px 16px', width: '100%', fontSize: '0.85rem' }}>
                  Choose Avatar Image File
                </label>
              </div>
            </div>
          </div>

          <button type="submit" className="btn btn-primary auth-submit-btn" disabled={loading} style={{ marginTop: '10px' }}>
            {loading ? (
              <>
                <Loader2 className="spinner-icon" size={18} /> Registering Account...
              </>
            ) : (
              'Create Account'
            )}
          </button>
        </form>

        <div className="auth-footer">
          <p>Already have an account? <Link to="/login">Sign in here</Link></p>
        </div>
      </div>
    </div>
  );
};

export default Register;
