import React from 'react';
import '../css/Loader.css';

const Loader = ({ fullScreen = false, size = 'md' }) => {
  return (
    <div className={`loader-container ${fullScreen ? 'fullscreen' : ''}`}>
      <div className={`spinner ${size}`}>
        <div className="double-bounce1"></div>
        <div className="double-bounce2"></div>
      </div>
      <p className="loader-text">Loading healthcare metrics...</p>
    </div>
  );
};

export default Loader;
