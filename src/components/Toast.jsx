import React, { useEffect } from 'react';
import { X, CheckCircle, AlertTriangle, AlertCircle, Info } from 'lucide-react';
import '../css/Toast.css';

const Toast = ({ message, type = 'success', onClose, duration = 4000 }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, duration);

    return () => clearTimeout(timer);
  }, [onClose, duration]);

  const getIcon = () => {
    switch (type) {
      case 'success':
        return <CheckCircle className="toast-icon success" />;
      case 'warning':
        return <AlertTriangle className="toast-icon warning" />;
      case 'error':
        return <AlertCircle className="toast-icon error" />;
      case 'info':
      default:
        return <Info className="toast-icon info" />;
    }
  };

  return (
    <div className={`toast-message ${type} animate-slide-in`}>
      <div className="toast-content">
        {getIcon()}
        <span className="toast-text">{message}</span>
      </div>
      <button className="toast-close" onClick={onClose}>
        <X size={16} />
      </button>
    </div>
  );
};

export default Toast;
