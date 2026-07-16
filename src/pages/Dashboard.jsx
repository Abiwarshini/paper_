import React, { useState, useEffect, useContext } from 'react';
import { Link } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import API from '../services/api';
import { 
  Baby, 
  ShieldAlert, 
  TrendingUp, 
  Activity, 
  Users, 
  Plus, 
  ArrowRight, 
  User, 
  Calendar,
  AlertTriangle,
  Heart
} from 'lucide-react';
import Card from '../components/Card';
import Loader from '../components/Loader';
import Alert from '../components/Alert';

const Dashboard = () => {
  const { user } = useContext(AuthContext);
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState({
    children: [],
    stats: null,
    recentPredictions: [],
    malnourishedCount: 0
  });
  const [error, setError] = useState('');

  const isClinician = ['Admin', 'Doctor', 'Health Worker'].includes(user?.role);

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        setError('');

        if (isClinician) {
          // Fetch reports to get statistics and counts
          const reportsRes = await API.get('/reports');
          const childrenRes = await API.get('/children?limit=5');

          if (reportsRes.data.success && childrenRes.data.success) {
            const counts = reportsRes.data.data.counts;
            const status = reportsRes.data.data.statusBreakdown;
            const malnourished = (status.moderateMalnutrition || 0) + (status.severeMalnutrition || 0) + (status.wasted || 0) + (status.stunted || 0) + (status.underweight || 0);

            setData({
              children: childrenRes.data.data.children || [],
              stats: counts,
              recentPredictions: reportsRes.data.data.recentPredictions || [],
              malnourishedCount: malnourished
            });
          }
        } else {
          // Fetch parent's children profiles
          const childrenRes = await API.get('/children');
          if (childrenRes.data.success) {
            const kids = childrenRes.data.data.children || [];
            
            // For each child, let's grab their details to show latest metrics
            const kidsWithHistory = await Promise.all(
              kids.map(async (kid) => {
                try {
                  const detailsRes = await API.get(`/children/${kid._id}`);
                  if (detailsRes.data.success) {
                    return {
                      ...kid,
                      latestPrediction: detailsRes.data.data.latestPrediction || null,
                      latestRecord: detailsRes.data.data.growthHistory?.[0] || null
                    };
                  }
                  return kid;
                } catch {
                  return kid;
                }
              })
            );

            setData({
              children: kidsWithHistory,
              stats: null,
              recentPredictions: [],
              malnourishedCount: 0
            });
          }
        }
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard metrics. Please reload.');
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [user, isClinician]);

  if (loading) {
    return <Loader fullScreen />;
  }

  const getRiskColor = (level) => {
    switch (level) {
      case 'High': return 'var(--danger)';
      case 'Moderate': return 'var(--warning)';
      case 'Low':
      default:
        return 'var(--success)';
    }
  };

  return (
    <div className="dashboard-wrapper animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '30px' }}>
      
      {/* Welcome Banner */}
      <div className="welcome-banner" style={{ 
        background: 'linear-gradient(135deg, var(--primary-dark), var(--primary))', 
        borderRadius: 'var(--radius-lg)', 
        padding: '30px', 
        color: 'var(--white)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        boxShadow: 'var(--shadow-md)',
        flexWrap: 'wrap',
        gap: '20px'
      }}>
        <div>
          <h1 style={{ color: 'var(--white)', fontSize: '1.8rem', marginBottom: '8px' }}>
            Good day, {user?.name || 'Healthcare Worker'}!
          </h1>
          <p style={{ opacity: 0.9, fontSize: '0.95rem' }}>
            Welcome to the AI-Based Malnutrition Prediction and Growth Monitoring Panel.
          </p>
        </div>
        <div style={{ display: 'flex', gap: '12px' }}>
          <Link to="/children" className="btn btn-secondary pulse-button" style={{ color: 'var(--white)', textShadow: 'none' }}>
            <Plus size={18} /> Add Child Profile
          </Link>
          <Link to="/predict" className="btn btn-light" style={{ backgroundColor: 'rgba(255,255,255,0.15)', color: 'var(--white)', border: 'none' }}>
            Run AI Diagnosis
          </Link>
        </div>
      </div>

      {error && <Alert type="danger" message={error} />}

      {/* Clinician Dashboard (Admin, Doctor, Health Worker) */}
      {isClinician ? (
        <>
          {/* Stat Metrics Grid */}
          <div className="grid-4">
            <Card 
              title="Total Registered Children" 
              value={data.stats?.totalChildren || 0} 
              icon={<Baby size={24} style={{ color: 'var(--primary)' }} />}
              description="Children currently enrolled"
              type="primary"
            />
            <Card 
              title="Malnourished Cases" 
              value={data.malnourishedCount} 
              icon={<ShieldAlert size={24} style={{ color: 'var(--danger)' }} />}
              description="Children at high/moderate risk"
              type="danger"
            />
            <Card 
              title="Active Staff & Parents" 
              value={data.stats?.totalUsers || 0} 
              icon={<Users size={24} style={{ color: 'var(--secondary)' }} />}
              description={`Doctors: ${data.stats?.doctors || 0} | Workers: ${data.stats?.healthWorkers || 0}`}
              type="success"
            />
            <Card 
              title="Malnutrition Percentage" 
              value={data.stats?.totalChildren > 0 ? `${((data.malnourishedCount / data.stats.totalChildren) * 100).toFixed(1)}%` : '0%'} 
              icon={<TrendingUp size={24} style={{ color: 'var(--warning)' }} />}
              description="System stunting/wasting rate"
              type="warning"
            />
          </div>

          <div className="grid-2" style={{ gridTemplateColumns: '3fr 2fr' }}>
            
            {/* Recent Checkups Table */}
            <Card title="Recently Enrolled Children" headerActions={
              <Link to="/children" style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.85rem', color: 'var(--primary)', fontWeight: '600' }}>
                View All <ArrowRight size={14} />
              </Link>
            }>
              {data.children.length === 0 ? (
                <p style={{ color: 'var(--gray-500)', padding: '20px 0', textAlign: 'center' }}>No child profiles found.</p>
              ) : (
                <div style={{ overflowX: 'auto' }}>
                  <table className="data-table" style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left', padding: '12px' }}>Name</th>
                        <th style={{ textAlign: 'left', padding: '12px' }}>Gender</th>
                        <th style={{ textAlign: 'left', padding: '12px' }}>Date of Birth</th>
                        <th style={{ textAlign: 'right', padding: '12px' }}>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.children.map((child) => (
                        <tr key={child._id} style={{ borderBottom: '1px solid var(--gray-100)' }}>
                          <td style={{ padding: '12px', fontWeight: '600' }}>{child.childName}</td>
                          <td style={{ padding: '12px' }}>{child.gender}</td>
                          <td style={{ padding: '12px' }}>
                            <span style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem' }}>
                              <Calendar size={14} className="text-gray-400" />
                              {new Date(child.dob).toLocaleDateString()}
                            </span>
                          </td>
                          <td style={{ padding: '12px', textAlign: 'right' }}>
                            <Link to={`/children?childId=${child._id}`} className="btn btn-light" style={{ padding: '6px 12px', fontSize: '0.8rem' }}>
                              View
                            </Link>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>

            {/* Recent Alerts List */}
            <Card title="Recent AI Diagnosis Alerts">
              {data.recentPredictions.filter(p => p.riskLevel !== 'Low').length === 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', padding: '30px 0' }}>
                  <Heart size={32} style={{ color: 'var(--success)' }} />
                  <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem' }}>No recent health risks detected.</p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  {data.recentPredictions.filter(p => p.riskLevel !== 'Low').slice(0, 5).map((pred) => (
                    <div key={pred._id} style={{ 
                      display: 'flex', 
                      gap: '12px', 
                      padding: '12px', 
                      borderRadius: 'var(--radius-md)', 
                      backgroundColor: pred.riskLevel === 'High' ? 'var(--danger-light)' : 'var(--warning-light)',
                      borderLeft: `4px solid ${getRiskColor(pred.riskLevel)}`
                    }}>
                      <AlertTriangle size={20} style={{ color: getRiskColor(pred.riskLevel), flexShrink: 0, marginTop: '2px' }} />
                      <div>
                        <h5 style={{ margin: 0, fontSize: '0.9rem', fontWeight: '700' }}>
                          {pred.childId?.childName || 'Child Profile'} ({pred.growthStatus})
                        </h5>
                        <p style={{ margin: '4px 0 0 0', fontSize: '0.8rem', color: 'var(--dark-light)' }}>
                          Risk: <strong>{pred.riskLevel}</strong> | Confidence: {pred.confidence}%
                        </p>
                        <p style={{ margin: '4px 0 0 0', fontSize: '0.75rem', color: 'var(--gray-500)', fontStyle: 'italic' }}>
                          WHO evaluation recorded on {new Date(pred.predictedDate).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </Card>
          </div>
        </>
      ) : (
        /* Parent Dashboard */
        <Card title="My Enrolled Children" headerActions={
          <Link to="/children" className="btn btn-primary" style={{ padding: '8px 16px', fontSize: '0.85rem' }}>
            <Plus size={16} /> Enroll New Child
          </Link>
        }>
          {data.children.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px 20px' }}>
              <Baby size={48} style={{ color: 'var(--gray-300)', marginBottom: '16px' }} />
              <h3>No Children Enrolled Yet</h3>
              <p style={{ color: 'var(--gray-500)', marginBottom: '20px' }}>
                You have not registered any children profiles to your parent account yet.
              </p>
              <Link to="/children" className="btn btn-primary">
                Add Child Profile Details
              </Link>
            </div>
          ) : (
            <div className="grid-3">
              {data.children.map((child) => (
                <div key={child._id} className="child-summary-card" style={{ 
                  border: '1px solid var(--gray-200)', 
                  borderRadius: 'var(--radius-md)', 
                  padding: '20px', 
                  backgroundColor: 'var(--white)',
                  boxShadow: 'var(--shadow-sm)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '12px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h3 style={{ margin: 0, fontSize: '1.15rem' }}>{child.childName}</h3>
                    <span className="badge" style={{ 
                      fontSize: '0.75rem', 
                      padding: '4px 8px', 
                      borderRadius: '12px',
                      backgroundColor: child.gender === 'Male' ? 'var(--primary-light)' : '#ffe8f4',
                      color: child.gender === 'Male' ? 'var(--primary)' : '#ff6eb4',
                      fontWeight: '700'
                    }}>
                      {child.gender}
                    </span>
                  </div>

                  <hr style={{ border: 'none', borderTop: '1px solid var(--gray-100)' }} />

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', fontSize: '0.85rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span className="text-gray-500">Date of Birth:</span>
                      <strong>{new Date(child.dob).toLocaleDateString()}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span className="text-gray-500">Age Bracket:</span>
                      <strong>{child.latestRecord ? `${child.latestRecord.ageMonths} months` : 'N/A'}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span className="text-gray-500">Last Weight:</span>
                      <strong>{child.latestRecord ? `${child.latestRecord.weight} kg` : 'Not Measured'}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span className="text-gray-500">Last Height:</span>
                      <strong>{child.latestRecord ? `${child.latestRecord.height} cm` : 'Not Measured'}</strong>
                    </div>
                  </div>

                  <hr style={{ border: 'none', borderTop: '1px solid var(--gray-100)' }} />

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className="text-gray-500" style={{ fontSize: '0.85rem' }}>Growth Status:</span>
                    <span style={{ 
                      fontSize: '0.85rem', 
                      fontWeight: '700',
                      color: getRiskColor(child.latestPrediction?.riskLevel)
                    }}>
                      {child.latestPrediction?.growthStatus || 'Unassessed'}
                    </span>
                  </div>

                  <div style={{ display: 'flex', gap: '8px', marginTop: '10px' }}>
                    <Link 
                      to={`/growth-monitoring?childId=${child._id}`} 
                      className="btn btn-light" 
                      style={{ flex: 1, padding: '8px', fontSize: '0.8rem' }}
                    >
                      Growth History
                    </Link>
                    <Link 
                      to={`/children?childId=${child._id}`} 
                      className="btn btn-primary" 
                      style={{ flex: 1, padding: '8px', fontSize: '0.8rem' }}
                    >
                      View Details
                    </Link>
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>
      )}

    </div>
  );
};

export default Dashboard;
