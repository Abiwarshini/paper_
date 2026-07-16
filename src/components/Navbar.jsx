import React, { useContext, useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { Bell, Menu, User, LogOut, Settings, Award } from 'lucide-react';
import API from '../services/api';
import '../css/Navbar.css';

const Navbar = ({ onToggleSidebar }) => {
  const { user, logoutUser } = useContext(AuthContext);
  const [showDropdown, setShowDropdown] = useState(false);
  const [notifications, setNotifications] = useState([]);
  const [showNotifications, setShowNotifications] = useState(false);
  const navigate = useNavigate();

  // Load simple notifications
  useEffect(() => {
    let active = true;
    const fetchNotifications = async () => {
      try {
        // Fetch from notification endpoint if logged in (simulate or fetch)
        // For simplicity and safety we fetch from /children or handle it
        // Let's create an endpoint in backend eventually, or fallback
        // We will query child predictions for warning levels as notifications
        const res = await API.get('/children');
        if (active && res.data.success) {
          // just mock alert if any children exist or pull from predictions
          const alerts = [];
          const childrenList = res.data.data.children || [];
          if (childrenList.length > 0) {
            alerts.push({
              id: 1,
              message: `Logged in as ${user?.role || 'User'}. System active.`
            });
          }
          setNotifications(alerts);
        }
      } catch (err) {
        console.warn('Could not fetch notifications logs', err);
      }
    };

    if (user) {
      fetchNotifications();
    }
    return () => {
      active = false;
    };
  }, [user]);

  const handleLogout = async () => {
    await logoutUser();
    navigate('/login');
  };

  const getRoleBadgeColor = (role) => {
    switch (role) {
      case 'Admin': return 'badge-admin';
      case 'Doctor': return 'badge-doctor';
      case 'Health Worker': return 'badge-hw';
      case 'Parent':
      default:
        return 'badge-parent';
    }
  };

  return (
    <nav className="navbar-container">
      <div className="navbar-left">
        <button className="sidebar-toggle-btn" onClick={onToggleSidebar}>
          <Menu size={22} />
        </button>
        <Link to="/dashboard" className="navbar-brand">
          <Award className="brand-logo-icon" size={26} />
          <span className="brand-title">NutriGrowth <span className="brand-title-sub">AI</span></span>
        </Link>
      </div>

      <div className="navbar-right">
        {/* Notifications Dropdown */}
        <div className="nav-icon-container">
          <button className="nav-icon-btn" onClick={() => setShowNotifications(!showNotifications)}>
            <Bell size={20} />
            {notifications.length > 0 && <span className="notification-badge">{notifications.length}</span>}
          </button>
          
          {showNotifications && (
            <div className="notifications-dropdown animate-fade-in">
              <div className="dropdown-header">Notifications</div>
              <div className="dropdown-body">
                {notifications.length === 0 ? (
                  <p className="no-notifications">No health alerts recorded.</p>
                ) : (
                  notifications.map((n) => (
                    <div key={n.id} className="notification-item">
                      <p className="notification-msg">{n.message}</p>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </div>

        {/* User Account Dropdown */}
        <div className="nav-profile-container">
          <button className="nav-profile-btn" onClick={() => setShowDropdown(!showDropdown)}>
            {user?.profileImage ? (
              <img src={`http://localhost:5000${user.profileImage}`} alt="Profile" className="nav-avatar" />
            ) : (
              <div className="nav-avatar-placeholder">
                <User size={18} />
              </div>
            )}
            <div className="nav-user-info">
              <span className="nav-username">{user?.name || 'Loading...'}</span>
              <span className={`nav-role-badge ${getRoleBadgeColor(user?.role)}`}>{user?.role}</span>
            </div>
          </button>

          {showDropdown && (
            <div className="profile-dropdown animate-fade-in" onMouseLeave={() => setShowDropdown(false)}>
              <Link to="/profile" className="dropdown-link" onClick={() => setShowDropdown(false)}>
                <Settings size={16} />
                <span>Profile Settings</span>
              </Link>
              <hr className="dropdown-divider" />
              <button className="dropdown-link logout-btn" onClick={handleLogout}>
                <LogOut size={16} />
                <span>Logout</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
