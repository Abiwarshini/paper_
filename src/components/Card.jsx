import React from 'react';
import '../css/Card.css';

const Card = ({ title, value, icon, description, trend, type = 'primary', children, headerActions }) => {
  // If children are provided, render as a Standard Content Panel Card
  if (children) {
    return (
      <div className="content-card">
        {(title || headerActions) && (
          <div className="content-card-header">
            {title && <h3 className="content-card-title">{title}</h3>}
            {headerActions && <div className="content-card-actions">{headerActions}</div>}
          </div>
        )}
        <div className="content-card-body">{children}</div>
      </div>
    );
  }

  // Otherwise, render as a Dashboard Stat Metric Card
  return (
    <div className={`metric-card ${type}`}>
      <div className="metric-card-inner">
        <div className="metric-card-info">
          <span className="metric-card-title">{title}</span>
          <h2 className="metric-card-value">{value}</h2>
          {(description || trend) && (
            <p className="metric-card-description">
              {trend && <span className="metric-trend">{trend}</span>}
              {description}
            </p>
          )}
        </div>
        {icon && <div className="metric-card-icon-wrapper">{icon}</div>}
      </div>
    </div>
  );
};

export default Card;
