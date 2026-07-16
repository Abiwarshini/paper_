import React, { useState, useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import API from '../services/api';
import { 
  User, 
  Mail, 
  Phone, 
  ShieldAlert, 
  Lock, 
  FileImage, 
  Loader2,
  CheckCircle,
  Camera
} from 'lucide-react';
import Card from '../components/Card';
import Alert from '../components/Alert';

const Profile = () => {
  const { user, updateLocalUser } = useContext(AuthContext);

  const [formData, setFormData] = useState({
    name: user?.name || '',
    phone: user?.phone || '',
    password: '',
    confirmPassword: ''
  });
  const [profileImage, setProfileImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(
    user?.profileImage ? `http://localhost:5000${user.profileImage}` : ''
  );
  
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState('');

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
    setSuccess(false);
    setError('');

    if (formData.password && formData.password !== formData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);

    try {
      const submissionData = new FormData();
      submissionData.append('name', formData.name);
      submissionData.append('phone', formData.phone);
      if (formData.password) {
        submissionData.append('password', formData.password);
      }
      if (profileImage) {
        submissionData.append('profileImage', profileImage);
      }

      const res = await API.put('/auth/profile', submissionData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      if (res.data.success) {
        setSuccess(true);
        // Sync local storage state
        const updatedUser = res.data.data.user;
        updateLocalUser(updatedUser);
        
        // Clear password fields
        setFormData({
          ...formData,
          password: '',
          confirmPassword: ''
        });
      }
    } catch (err) {
      setError(err.response?.data?.message || 'Failed to update profile settings.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="profile-wrapper animate-fade-in" style={{ maxWidth: '800px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Title */}
      <div>
        <h1 style={{ fontSize: '1.6rem' }}>My Profile Settings</h1>
        <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem', margin: '4px 0 0 0' }}>
          Update your profile contact information, password details, and profile avatar.
        </p>
      </div>

      {success && <Alert type="success" message="Profile settings updated successfully!" />}
      {error && <Alert type="danger" message={error} />}

      <div className="grid-2" style={{ gridTemplateColumns: '1fr 2fr', alignItems: 'flex-start' }}>
        
        {/* Left: Avatar Display Card */}
        <Card>
          <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center', 
            gap: '16px',
            textAlign: 'center',
            padding: '20px 0'
          }}>
            <div style={{ position: 'relative' }}>
              <div className="profile-avatar-large" style={{ 
                width: '120px', 
                height: '120px', 
                borderRadius: '50%', 
                backgroundColor: 'var(--primary-light)',
                border: '3px solid var(--primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
                boxShadow: 'var(--shadow-md)'
              }}>
                {imagePreview ? (
                  <img src={imagePreview} alt="Avatar" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  <User size={48} className="text-gray-400" />
                )}
              </div>
              
              {/* Photo upload camera button */}
              <label htmlFor="avatar-file-input" style={{ 
                position: 'absolute', 
                bottom: '0', 
                right: '0', 
                backgroundColor: 'var(--secondary)', 
                color: 'var(--white)',
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                boxShadow: 'var(--shadow-sm)',
                border: '2px solid var(--white)'
              }}>
                <Camera size={16} />
              </label>
              <input 
                type="file" 
                id="avatar-file-input" 
                accept="image/*" 
                onChange={handleFileChange}
                disabled={loading}
                style={{ display: 'none' }}
              />
            </div>

            <div>
              <h3 style={{ margin: 0, fontSize: '1.2rem' }}>{user?.name}</h3>
              <span className="badge" style={{ 
                fontSize: '0.75rem', 
                padding: '4px 10px', 
                borderRadius: '12px',
                backgroundColor: 'var(--gray-100)',
                color: 'var(--gray-500)',
                fontWeight: '700',
                marginTop: '6px',
                display: 'inline-block'
              }}>
                {user?.role} Role privilege
              </span>
            </div>
          </div>
        </Card>

        {/* Right: Update Form Panel */}
        <Card title="Update Account Details">
          <form onSubmit={handleSubmit} className="auth-form" style={{ padding: 0 }}>
            
            <div className="form-group">
              <label className="form-label">Full Name</label>
              <div className="auth-input-wrapper">
                <User className="auth-input-icon" size={18} />
                <input
                  type="text"
                  name="name"
                  className="auth-input"
                  required
                  value={formData.name}
                  onChange={handleChange}
                  disabled={loading}
                />
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Email Address (Read-only)</label>
              <div className="auth-input-wrapper" style={{ opacity: 0.7 }}>
                <Mail className="auth-input-icon" size={18} />
                <input
                  type="email"
                  className="auth-input"
                  value={user?.email || ''}
                  disabled={true}
                  style={{ backgroundColor: 'var(--gray-100)' }}
                />
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Designated Role Privilege</label>
                <div className="auth-input-wrapper" style={{ opacity: 0.7 }}>
                  <ShieldAlert className="auth-input-icon" size={18} />
                  <input
                    type="text"
                    className="auth-input"
                    value={user?.role || ''}
                    disabled={true}
                    style={{ backgroundColor: 'var(--gray-100)' }}
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Contact Phone Number</label>
                <div className="auth-input-wrapper">
                  <Phone className="auth-input-icon" size={18} />
                  <input
                    type="tel"
                    name="phone"
                    className="auth-input"
                    required
                    value={formData.phone}
                    onChange={handleChange}
                    disabled={loading}
                  />
                </div>
              </div>
            </div>

            <hr style={{ border: 'none', borderTop: '1px solid var(--gray-100)', margin: '15px 0' }} />

            <div className="form-row">
              <div className="form-group">
                <label className="form-label">Change Password (Optional)</label>
                <div className="auth-input-wrapper">
                  <Lock className="auth-input-icon" size={18} />
                  <input
                    type="password"
                    name="password"
                    className="auth-input"
                    placeholder="New secure password"
                    value={formData.password}
                    onChange={handleChange}
                    disabled={loading}
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Confirm New Password</label>
                <div className="auth-input-wrapper">
                  <Lock className="auth-input-icon" size={18} />
                  <input
                    type="password"
                    name="confirmPassword"
                    className="auth-input"
                    placeholder="Confirm new password"
                    value={formData.confirmPassword}
                    onChange={handleChange}
                    disabled={loading}
                  />
                </div>
              </div>
            </div>

            <button type="submit" className="btn btn-primary" style={{ width: '100%', marginTop: '10px' }} disabled={loading}>
              {loading ? (
                <>
                  <Loader2 className="spinner-icon" size={18} /> Updating profile settings...
                </>
              ) : (
                'Save Profile Changes'
              )}
            </button>
          </form>
        </Card>

      </div>

    </div>
  );
};

export default Profile;
