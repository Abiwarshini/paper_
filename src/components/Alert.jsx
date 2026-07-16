import React from 'react';
import { AlertCircle, CheckCircle2, Info, AlertTriangle } from 'lucide-react';
import '../css/Alert.css';

const Alert = ({ message, type = 'info', children }) => {
  const getIcon = () => {
    switch (type) {
      case 'success':
        return <CheckCircle2 className="alert-inline-icon" size={20} />;
      case 'warning':
        return <AlertTriangle className="alert-inline-icon" size={20} />;
      case 'danger':
        return <AlertCircle className="alert-inline-icon" size={20} />;
      case 'info':
      default:
        return <Info className="alert-inline-icon" size={20} />;
    }
  };

  return (
    <div className={`alert-inline ${type}`}>
      <div className="alert-inline-header">
        {getIcon()}
        <span className="alert-inline-text">{message}</span>
      </div>
      {children && <div className="alert-inline-content">{children}</div>}
    </div>
  );
};

export default Alert;
