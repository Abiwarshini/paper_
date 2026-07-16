import React, { useState, useEffect } from 'react';
import API from '../services/api';
import { 
  FileText, 
  Baby, 
  Users, 
  ShieldAlert, 
  Activity, 
  MapPin, 
  AlertTriangle,
  Heart
} from 'lucide-react';
import Card from '../components/Card';
import Loader from '../components/Loader';
import Alert from '../components/Alert';

// ChartJS elements
import {
  Chart as ChartJS,
  ArcElement,
  BarElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';

ChartJS.register(
  ArcElement,
  BarElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend
);

const Reports = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchReports = async () => {
      try {
        setLoading(true);
        const res = await API.get('/reports');
        if (res.data.success) {
          setData(res.data.data);
        }
      } catch (err) {
        console.error(err);
        setError('Failed to fetch global medical metrics summaries.');
      } finally {
        setLoading(false);
      }
    };
    fetchReports();
  }, []);

  if (loading) {
    return <Loader fullScreen />;
  }

  if (error || !data) {
    return (
      <div style={{ padding: '20px' }}>
        <Alert type="danger" message={error || 'Could not load reporting dashboard.'} />
      </div>
    );
  }

  // Compile Doughnut Chart Data (Malnutrition categories)
  const breakdown = data.statusBreakdown || {};
  const doughnutData = {
    labels: ['Healthy', 'Stunted', 'Wasted', 'Underweight', 'Moderate Malnutrition', 'Severe Malnutrition'],
    datasets: [
      {
        data: [
          breakdown.healthy || 0,
          breakdown.stunted || 0,
          breakdown.wasted || 0,
          breakdown.underweight || 0,
          breakdown.moderateMalnutrition || 0,
          breakdown.severeMalnutrition || 0
        ],
        backgroundColor: [
          '#2ecc71', // Healthy
          '#00bcd4', // Stunted
          '#ff9800', // Wasted
          '#ffc107', // Underweight
          '#f59e0b', // Moderate
          '#ef4444'  // Severe
        ],
        borderColor: '#ffffff',
        borderWidth: 2
      }
    ]
  };

  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right',
        labels: {
          font: { family: 'Plus Jakarta Sans', weight: '600', size: 11 }
        }
      }
    }
  };

  // Compile Bar Chart Data (District hot-spots)
  const regionBreakdown = data.regionalBreakdown || [];
  const barData = {
    labels: regionBreakdown.map(r => r._id || 'Unknown'),
    datasets: [
      {
        label: 'Enrolled Children',
        data: regionBreakdown.map(r => r.count),
        backgroundColor: 'rgba(30, 144, 255, 0.75)',
        borderColor: 'var(--primary)',
        borderWidth: 1.5,
        borderRadius: 4
      }
    ]
  };

  const barOptions = {
    indexAxis: 'y', // horizontal bar chart
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false }
    },
    scales: {
      x: {
        title: { display: true, text: 'Enrolled Case Count', font: { family: 'Plus Jakarta Sans', weight: '700' } },
        grid: { color: 'var(--gray-100)' },
        ticks: { stepSize: 1 }
      },
      y: {
        grid: { display: false }
      }
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'High': return 'var(--danger)';
      case 'Moderate': return 'var(--warning)';
      case 'Low':
      default:
        return 'var(--success)';
    }
  };

  const totalMalnourished = 
    (breakdown.stunted || 0) + 
    (breakdown.wasted || 0) + 
    (breakdown.underweight || 0) + 
    (breakdown.moderateMalnutrition || 0) + 
    (breakdown.severeMalnutrition || 0);

  return (
    <div className="reports-wrapper animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Title block */}
      <div style={{ 
        backgroundColor: 'var(--white)', 
        borderRadius: 'var(--radius-lg)', 
        padding: '24px', 
        boxShadow: 'var(--shadow-sm)'
      }}>
        <h1 style={{ fontSize: '1.6rem' }}>Clinical System Analytics Reports</h1>
        <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem', margin: '4px 0 0 0' }}>
          Aggregate caseload trends, stunting/wasting ratios, and geographical regional hot-spot tracking.
        </p>
      </div>

      {/* Counters Grid */}
      <div className="grid-4">
        <Card 
          title="Total Children Tracked" 
          value={data.counts?.totalChildren || 0} 
          icon={<Baby size={24} style={{ color: 'var(--primary)' }} />}
          description="Total diagnostic enrollments"
          type="primary"
        />
        <Card 
          title="Active Doctors & Workers" 
          value={(data.counts?.doctors || 0) + (data.counts?.healthWorkers || 0)} 
          icon={<Users size={24} style={{ color: 'var(--secondary)' }} />}
          description={`Doctors: ${data.counts?.doctors || 0} | HWs: ${data.counts?.healthWorkers || 0}`}
          type="success"
        />
        <Card 
          title="Total Malnourished Cases" 
          value={totalMalnourished} 
          icon={<ShieldAlert size={24} style={{ color: 'var(--danger)' }} />}
          description="Requires clinical follow-up checks"
          type="danger"
        />
        <Card 
          title="Clinically Healthy Cases" 
          value={breakdown.healthy || 0} 
          icon={<Heart size={24} style={{ color: 'var(--success)' }} />}
          description="Within WHO normal standard ranges"
          type="warning"
        />
      </div>

      {/* Graphs Grid */}
      <div className="grid-2">
        {/* Malnutrition Doughnut */}
        <Card title="Malnutrition Categories Distribution">
          <div style={{ height: '280px', position: 'relative' }}>
            <Doughnut data={doughnutData} options={doughnutOptions} />
          </div>
        </Card>

        {/* Hot-spots bar chart */}
        <Card title="Patient Caseload Distribution by District">
          {regionBreakdown.length === 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '60px 0' }}>
              <MapPin size={32} className="text-gray-300" />
              <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem' }}>No district records found.</p>
            </div>
          ) : (
            <div style={{ height: '280px', position: 'relative' }}>
              <Bar data={barData} options={barOptions} />
            </div>
          )}
        </Card>
      </div>

      {/* Recent Predictions */}
      <Card title="Recent Assessment Logs (All Enrolled Patients)">
        {data.recentPredictions?.length === 0 ? (
          <p style={{ color: 'var(--gray-500)', padding: '20px 0', textAlign: 'center' }}>No clinical assessments have been run yet.</p>
        ) : (
          <div className="table-responsive">
            <table className="custom-table" style={{ fontSize: '0.85rem' }}>
              <thead>
                <tr>
                  <th>Child Name</th>
                  <th>Gender</th>
                  <th>Assessment Date</th>
                  <th>Malnutrition Status</th>
                  <th>Confidence Index</th>
                  <th>System Risk Level</th>
                </tr>
              </thead>
              <tbody>
                {data.recentPredictions.map((pred) => (
                  <tr key={pred._id}>
                    <td style={{ fontWeight: '600' }}>{pred.childId?.childName || 'Child Profile Deleted'}</td>
                    <td>{pred.childId?.gender || 'N/A'}</td>
                    <td>{new Date(pred.predictedDate).toLocaleDateString()}</td>
                    <td>{pred.growthStatus}</td>
                    <td>{pred.confidence}%</td>
                    <td>
                      <span style={{ 
                        color: getRiskColor(pred.riskLevel),
                        fontWeight: '700'
                      }}>
                        {pred.riskLevel} Risk
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

    </div>
  );
};

export default Reports;
