import React, { useContext } from 'react';
import { NavLink } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import { 
  LayoutDashboard, 
  Baby, 
  TrendingUp, 
  Activity, 
  FileText, 
  ShieldAlert, 
  UserCircle,
  HelpCircle
} from 'lucide-react';
import '../css/Sidebar.css';

const Sidebar = ({ isOpen, onClose }) => {
  const { user } = useContext(AuthContext);

  const getLinks = () => {
    const links = [
      { to: '/dashboard', label: 'Dashboard', icon: <LayoutDashboard size={20} />, roles: ['Admin', 'Doctor', 'Health Worker', 'Parent'] },
      { to: '/children', label: 'Child Registry', icon: <Baby size={20} />, roles: ['Admin', 'Doctor', 'Health Worker', 'Parent'] },
      { to: '/growth-monitoring', label: 'Growth Monitoring', icon: <TrendingUp size={20} />, roles: ['Admin', 'Doctor', 'Health Worker', 'Parent'] },
      { to: '/predict', label: 'AI Predictor', icon: <Activity size={20} />, roles: ['Admin', 'Doctor', 'Health Worker', 'Parent'] },
      { to: '/reports', label: 'System Reports', icon: <FileText size={20} />, roles: ['Admin', 'Doctor', 'Health Worker'] },
      { to: '/admin', label: 'Admin Panel', icon: <ShieldAlert size={20} />, roles: ['Admin'] },
      { to: '/profile', label: 'My Profile', icon: <UserCircle size={20} />, roles: ['Admin', 'Doctor', 'Health Worker', 'Parent'] }
    ];

    return links.filter(link => link.roles.includes(user?.role));
  };

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && <div className="sidebar-overlay-mobile" onClick={onClose}></div>}
      
      <aside className={`sidebar-container ${isOpen ? 'open' : ''}`}>
        <div className="sidebar-links-wrapper">
          {getLinks().map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}
              onClick={onClose} // Auto close drawer on mobile navigation
            >
              <span className="sidebar-link-icon">{link.icon}</span>
              <span className="sidebar-link-label">{link.label}</span>
            </NavLink>
          ))}
        </div>
        
        <div className="sidebar-footer">
          <div className="sidebar-help-card">
            <HelpCircle className="help-icon" size={24} />
            <h4>Need Assistance?</h4>
            <p>Consult the documentation or contact system support.</p>
          </div>
          <p className="footer-copyright">© 2026 NutriGrowth AI</p>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
