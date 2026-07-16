import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { 
  HeartPulse, 
  Activity, 
  Users, 
  ShieldAlert, 
  Award, 
  CheckCircle, 
  MessageSquare,
  ArrowRight,
  TrendingUp,
  FileText
} from 'lucide-react';
import '../css/Landing.css';

const Landing = () => {
  const [contact, setContact] = useState({ name: '', email: '', message: '' });
  const [success, setSuccess] = useState(false);

  const handleContactSubmit = (e) => {
    e.preventDefault();
    setSuccess(true);
    setContact({ name: '', email: '', message: '' });
    setTimeout(() => setSuccess(false), 5000);
  };

  return (
    <div className="landing-page-container">
      {/* Header / Navbar */}
      <header className="landing-header">
        <div className="landing-logo">
          <HeartPulse className="landing-logo-icon" />
          <span>NutriGrowth <span className="sub-logo">AI</span></span>
        </div>
        <nav className="landing-nav">
          <a href="#about">About</a>
          <a href="#features">Features</a>
          <a href="#benefits">Benefits</a>
          <a href="#stats">Statistics</a>
          <a href="#contact">Contact</a>
          <Link to="/login" className="btn-nav-login">Sign In</Link>
        </nav>
      </header>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content animate-fade-in">
          <div className="hero-badge">
            <TrendingUp size={16} />
            <span>AI-Powered Pediatric Health</span>
          </div>
          <h1>Early Malnutrition Prediction & Growth Monitoring</h1>
          <p>
            Detect child health risks before they develop. Using WHO growth references 
            and intelligent parameters, NutriGrowth AI provides community doctors, health workers, 
            and parents with actionable nutritional interventions.
          </p>
          <div className="hero-actions">
            <Link to="/register" className="btn btn-primary btn-lg">
              Get Started <ArrowRight size={18} />
            </Link>
            <a href="#about" className="btn btn-light btn-lg">Learn More</a>
          </div>
        </div>
        <div className="hero-visual">
          <div className="visual-circle animate-pulse"></div>
          <div className="visual-card-wrapper">
            <div className="visual-card">
              <Activity className="visual-icon text-primary" />
              <div>
                <h5>AI Prediction</h5>
                <span className="badge badge-success">98% Accuracy</span>
              </div>
            </div>
            <div className="visual-card second">
              <HeartPulse className="visual-icon text-success" />
              <div>
                <h5>Growth Index</h5>
                <span>WHO Standards</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* About Project */}
      <section id="about" className="about-section">
        <div className="section-header">
          <h2>About The Project</h2>
          <p>Fighting early-stage child malnutrition through cloud diagnostics and community statistics.</p>
        </div>
        <div className="about-grid">
          <div className="about-text">
            <h3>Bridging the Gap in Pediatric Care</h3>
            <p>
              Pediatric malnutrition remains one of the largest preventable causes of stunting and wasting 
              globally. Standard diagnoses are often performed too late due to a lack of structured growth logs.
            </p>
            <p>
              NutriGrowth AI provides a simple, unified cloud repository for health workers to capture height, 
              weight, and MUAC, automatically comparing measurements against WHO growth percentiles. 
              Its AI predictive service highlights risk trajectories to initiate therapeutic feeding programs early.
            </p>
          </div>
          <div className="about-image-placeholder">
            <div className="about-glass-panel">
              <h4>Active Indicators Logged</h4>
              <ul>
                <li><CheckCircle size={16} className="text-primary" /> Height-for-Age (HAZ stunting)</li>
                <li><CheckCircle size={16} className="text-primary" /> Weight-for-Age (WAZ underweight)</li>
                <li><CheckCircle size={16} className="text-primary" /> Weight-for-Height (WHZ wasting)</li>
                <li><CheckCircle size={16} className="text-primary" /> Mid-Upper Arm Circumference (MUAC acute risk)</li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="features-section">
        <div className="section-header">
          <h2>Core Features</h2>
          <p>Engineered for community health volunteers and medical directors alike.</p>
        </div>
        <div className="features-grid grid-3">
          <div className="feature-card">
            <div className="feature-icon bg-primary-light text-primary">
              <Activity size={24} />
            </div>
            <h3>AI Malnutrition Predictor</h3>
            <p>Calculate growth metrics and automatically predict healthy, stunting, wasting, MAM, or SAM risk classifications.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon bg-secondary-light text-secondary">
              <TrendingUp size={24} />
            </div>
            <h3>Growth Trend Tracking</h3>
            <p>Interactive chart visualization for height-for-age, weight-for-age, and BMI curves against standard WHO references.</p>
          </div>
          <div className="feature-card">
            <div className="feature-icon bg-info-light text-info">
              <FileText size={24} />
            </div>
            <h3>Nutritional Planning</h3>
            <p>Automatically generate clinical food recommendations, daily calorie-protein guidelines, and vitamin supplements.</p>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section id="benefits" className="benefits-section">
        <div className="section-header">
          <h2>Why NutriGrowth AI?</h2>
          <p>Providing specialized benefits across all levels of childcare.</p>
        </div>
        <div className="benefits-grid grid-2">
          <div className="benefit-item">
            <Users className="benefit-icon" />
            <div>
              <h4>For Health Workers & Doctors</h4>
              <p>Quickly screen children during field campaigns, generate growth reports instantly, and track regional records without manual paper logs.</p>
            </div>
          </div>
          <div className="benefit-item">
            <ShieldAlert className="benefit-icon" />
            <div>
              <h4>For Health Directors & Admins</h4>
              <p>Monitor malnutrition rates grouped by district, track performance maps, and deploy resources to high-risk zones efficiently.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Statistics Section */}
      <section id="stats" className="stats-section">
        <div className="section-header">
          <h2>Project Impact</h2>
          <p>Real-time analytics logged across active community centers.</p>
        </div>
        <div className="stats-grid grid-4">
          <div className="stat-card">
            <h3>1,250+</h3>
            <p>Children Monitored</p>
          </div>
          <div className="stat-card">
            <h3>98.2%</h3>
            <p>Prediction Accuracy</p>
          </div>
          <div className="stat-card">
            <h3>45+</h3>
            <p>Active Clinics</p>
          </div>
          <div className="stat-card">
            <h3>85%</h3>
            <p>Recovery Rate</p>
          </div>
        </div>
      </section>

      {/* Contact Section */}
      <section id="contact" className="contact-section">
        <div className="section-header">
          <h2>Contact Project Team</h2>
          <p>Get in touch to deploy NutriGrowth AI in your local health district.</p>
        </div>
        <div className="contact-container">
          <form className="contact-form" onSubmit={handleContactSubmit}>
            {success && (
              <div className="alert-inline success" style={{ marginBottom: 20 }}>
                Thank you! Your message has been sent successfully. Our team will contact you.
              </div>
            )}
            <div className="form-group">
              <label className="form-label">Full Name</label>
              <input 
                type="text" 
                className="form-input" 
                required 
                value={contact.name}
                onChange={(e) => setContact({ ...contact, name: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Email Address</label>
              <input 
                type="email" 
                className="form-input" 
                required 
                value={contact.email}
                onChange={(e) => setContact({ ...contact, email: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Message</label>
              <textarea 
                className="form-input" 
                rows="4" 
                required
                value={contact.message}
                onChange={(e) => setContact({ ...contact, message: e.target.value })}
              ></textarea>
            </div>
            <button type="submit" className="btn btn-primary">
              <MessageSquare size={18} /> Send Message
            </button>
          </form>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="footer-top">
          <div className="footer-brand">
            <HeartPulse className="landing-logo-icon" />
            <span>NutriGrowth AI</span>
          </div>
          <p>Deploying intelligent technology in the service of child development.</p>
        </div>
        <hr className="footer-divider" />
        <div className="footer-bottom">
          <p>© 2026 NutriGrowth AI Project. All rights reserved.</p>
          <div className="footer-links">
            <a href="#about">Privacy Policy</a>
            <a href="#about">Terms of Service</a>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
