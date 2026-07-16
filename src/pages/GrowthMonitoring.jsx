import React, { useState, useEffect, useContext } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import API from '../services/api';
import { 
  Baby, 
  Plus, 
  TrendingUp, 
  Table, 
  Activity, 
  ChevronRight,
  TrendingDown,
  Info
} from 'lucide-react';
import Card from '../components/Card';
import Loader from '../components/Loader';
import Alert from '../components/Alert';
import Modal from '../components/Modal';

// ChartJS setup
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// WHO Standard Reference values (50th percentile / Median)
const whoStandards = {
  Male: {
    height: [
      { age: 0, val: 49.9 },
      { age: 6, val: 67.6 },
      { age: 12, val: 75.7 },
      { age: 18, val: 82.3 },
      { age: 24, val: 87.8 },
      { age: 36, val: 96.1 },
      { age: 48, val: 103.3 },
      { age: 60, val: 110.0 }
    ],
    weight: [
      { age: 0, val: 3.3 },
      { age: 6, val: 7.9 },
      { age: 12, val: 9.6 },
      { age: 18, val: 10.9 },
      { age: 24, val: 12.2 },
      { age: 36, val: 14.3 },
      { age: 48, val: 16.3 },
      { age: 60, val: 18.3 }
    ],
    bmi: [
      { age: 0, val: 13.3 },
      { age: 6, val: 17.3 },
      { age: 12, val: 16.7 },
      { age: 18, val: 16.1 },
      { age: 24, val: 15.8 },
      { age: 36, val: 15.5 },
      { age: 48, val: 15.3 },
      { age: 60, val: 15.1 }
    ]
  },
  Female: {
    height: [
      { age: 0, val: 49.1 },
      { age: 6, val: 65.7 },
      { age: 12, val: 74.0 },
      { age: 18, val: 80.7 },
      { age: 24, val: 86.4 },
      { age: 36, val: 95.1 },
      { age: 48, val: 102.7 },
      { age: 60, val: 109.4 }
    ],
    weight: [
      { age: 0, val: 3.2 },
      { age: 6, val: 7.3 },
      { age: 12, val: 8.9 },
      { age: 18, val: 10.2 },
      { age: 24, val: 11.5 },
      { age: 36, val: 13.9 },
      { age: 48, val: 15.5 },
      { age: 60, val: 17.4 }
    ],
    bmi: [
      { age: 0, val: 13.3 },
      { age: 6, val: 16.9 },
      { age: 12, val: 16.3 },
      { age: 18, val: 15.7 },
      { age: 24, val: 15.4 },
      { age: 36, val: 15.2 },
      { age: 48, val: 15.0 },
      { age: 60, val: 14.8 }
    ]
  }
};

const GrowthMonitoring = () => {
  const { user } = useContext(AuthContext);
  const [searchParams, setSearchParams] = useSearchParams();
  const childIdParam = searchParams.get('childId');

  const [childrenList, setChildrenList] = useState([]);
  const [selectedChildId, setSelectedChildId] = useState('');
  const [loading, setLoading] = useState(true);
  const [historyData, setHistoryData] = useState(null);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [activeTab, setActiveTab] = useState('height'); // height | weight | bmi

  // Log growth modal state
  const [logModalOpen, setLogModalOpen] = useState(false);
  const [formData, setFormData] = useState({
    height: '',
    weight: '',
    MUAC: '',
    headCircumference: '',
    measurementDate: new Date().toISOString().substring(0, 10)
  });
  const [logError, setLogError] = useState('');
  const [logLoading, setLogLoading] = useState(false);

  // Fetch list of all children for selector
  useEffect(() => {
    const fetchKids = async () => {
      try {
        setLoading(true);
        const res = await API.get('/children?limit=100');
        if (res.data.success) {
          const list = res.data.data.children || [];
          setChildrenList(list);

          // If query param is present and valid, pick it
          if (childIdParam) {
            setSelectedChildId(childIdParam);
          } else if (list.length > 0) {
            setSelectedChildId(list[0]._id);
            setSearchParams({ childId: list[0]._id });
          }
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetchKids();
  }, []);

  // Fetch growth history when child selection changes
  useEffect(() => {
    const fetchHistory = async () => {
      if (!selectedChildId) return;
      try {
        setLoadingHistory(true);
        const res = await API.get(`/growth/${selectedChildId}`);
        if (res.data.success) {
          setHistoryData(res.data.data);
        }
      } catch (err) {
        console.error(err);
      } finally {
        setLoadingHistory(false);
      }
    };

    fetchHistory();
  }, [selectedChildId]);

  const handleChildSelectChange = (e) => {
    const id = e.target.value;
    setSelectedChildId(id);
    setSearchParams(id ? { childId: id } : {});
  };

  const handleOpenLogModal = () => {
    setFormData({
      height: '',
      weight: '',
      MUAC: '',
      headCircumference: '',
      measurementDate: new Date().toISOString().substring(0, 10)
    });
    setLogError('');
    setLogModalOpen(true);
  };

  const handleLogSubmit = async (e) => {
    e.preventDefault();
    setLogError('');
    setLogLoading(true);

    try {
      const payload = {
        childId: selectedChildId,
        height: parseFloat(formData.height),
        weight: parseFloat(formData.weight),
        MUAC: parseFloat(formData.MUAC),
        headCircumference: parseFloat(formData.headCircumference),
        measurementDate: formData.measurementDate
      };

      const res = await API.post('/growth', payload);
      if (res.data.success) {
        // Refetch history
        const historyRes = await API.get(`/growth/${selectedChildId}`);
        if (historyRes.data.success) {
          setHistoryData(historyRes.data.data);
        }
        setLogModalOpen(false);
      }
    } catch (err) {
      setLogError(err.response?.data?.message || 'Logging growth record failed.');
    } finally {
      setLogLoading(false);
    }
  };

  // Compile Chart data configurations
  const buildChartData = () => {
    if (!historyData || !historyData.growthRecords || historyData.growthRecords.length === 0) {
      return null;
    }

    const records = [...historyData.growthRecords].sort((a, b) => a.ageMonths - b.ageMonths);
    const gender = historyData.child.gender === 'Female' ? 'Female' : 'Male';
    
    // Child's actual coordinates
    const childAges = records.map(r => r.ageMonths);
    
    // WHO standard matching coordinates
    const standardRefs = whoStandards[gender][activeTab];

    // Build common list of age coordinates to plot
    // Merge child ages and standard ages
    const uniqueAges = Array.from(new Set([...childAges, ...standardRefs.map(s => s.age)])).sort((a, b) => a - b);

    // Helper to linear interpolate WHO reference values on the fly
    const getWhoVal = (age) => {
      const refs = standardRefs;
      if (age <= refs[0].age) return refs[0].val;
      if (age >= refs[refs.length - 1].age) return refs[refs.length - 1].val;

      let lowerIdx = 0;
      for (let i = 0; i < refs.length - 1; i++) {
        if (age >= refs[i].age && age <= refs[i+1].age) {
          lowerIdx = i;
          break;
        }
      }
      const lower = refs[lowerIdx];
      const upper = refs[lowerIdx + 1];
      const fraction = (age - lower.age) / (upper.age - lower.age);
      return parseFloat((lower.val + fraction * (upper.val - lower.val)).toFixed(2));
    };

    const actualValues = uniqueAges.map(age => {
      const match = records.find(r => r.ageMonths === age);
      if (!match) return null; // Chart.js handles nulls by showing gaps or skipping

      if (activeTab === 'height') return match.height;
      if (activeTab === 'weight') return match.weight;
      if (activeTab === 'bmi') {
        const hMeter = match.height / 100;
        return parseFloat((match.weight / (hMeter * hMeter)).toFixed(2));
      }
      return null;
    });

    const whoValues = uniqueAges.map(age => getWhoVal(age));

    return {
      labels: uniqueAges.map(a => `${a}M`),
      datasets: [
        {
          label: `${historyData.child.childName}'s Growth`,
          data: actualValues,
          borderColor: 'var(--primary)',
          backgroundColor: 'rgba(30, 144, 255, 0.1)',
          borderWidth: 3,
          pointBackgroundColor: 'var(--primary)',
          pointHoverRadius: 8,
          spanGaps: true,
          tension: 0.2
        },
        {
          label: `WHO 50th Percentile (${gender})`,
          data: whoValues,
          borderColor: 'var(--secondary)',
          borderDash: [5, 5],
          backgroundColor: 'transparent',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: 'var(--secondary)',
          tension: 0.1
        }
      ]
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: { family: 'Plus Jakarta Sans', weight: '600' }
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false
      }
    },
    scales: {
      y: {
        title: {
          display: true,
          text: activeTab === 'height' ? 'Height (cm)' : activeTab === 'weight' ? 'Weight (kg)' : 'BMI Index (kg/m²)',
          font: { family: 'Plus Jakarta Sans', weight: '700' }
        },
        grid: { color: 'var(--gray-100)' }
      },
      x: {
        title: {
          display: true,
          text: 'Child Age (Months)',
          font: { family: 'Plus Jakarta Sans', weight: '700' }
        },
        grid: { display: false }
      }
    }
  };

  if (loading) {
    return <Loader fullScreen />;
  }

  const hasData = historyData?.growthRecords && historyData.growthRecords.length > 0;

  return (
    <div className="growth-monitoring-wrapper animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Title & Child Selector */}
      <div style={{ 
        backgroundColor: 'var(--white)', 
        borderRadius: 'var(--radius-lg)', 
        padding: '24px', 
        boxShadow: 'var(--shadow-sm)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '16px'
      }}>
        <div>
          <h1 style={{ fontSize: '1.6rem' }}>Clinical Growth Charts</h1>
          <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem', margin: '4px 0 0 0' }}>
            Plot patient diagnostics and overlay metrics against WHO growth percentiles.
          </p>
        </div>
        
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <span style={{ fontSize: '0.85rem', fontWeight: 'bold', color: 'var(--dark-light)' }}>Select Patient:</span>
          {childrenList.length === 0 ? (
            <Link to="/children" className="btn btn-secondary btn-sm">Register Child First</Link>
          ) : (
            <select
              value={selectedChildId}
              onChange={handleChildSelectChange}
              className="form-select"
              style={{ padding: '8px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--gray-200)', fontSize: '0.9rem' }}
            >
              {childrenList.map((c) => (
                <option key={c._id} value={c._id}>{c.childName} ({c.gender})</option>
              ))}
            </select>
          )}
        </div>
      </div>

      {selectedChildId && historyData ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          {/* Main Assessment Summary */}
          {historyData.latestPrediction && (
            <div style={{ 
              backgroundColor: historyData.latestPrediction.riskLevel === 'High' ? 'var(--danger-light)' : historyData.latestPrediction.riskLevel === 'Moderate' ? 'var(--warning-light)' : 'var(--success-light)',
              borderLeft: `5px solid ${historyData.latestPrediction.riskLevel === 'High' ? 'var(--danger)' : historyData.latestPrediction.riskLevel === 'Moderate' ? 'var(--warning)' : 'var(--success)'}`,
              borderRadius: 'var(--radius-md)',
              padding: '16px 20px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: '16px'
            }}>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <Info size={20} style={{ color: historyData.latestPrediction.riskLevel === 'High' ? 'var(--danger)' : historyData.latestPrediction.riskLevel === 'Moderate' ? 'var(--warning)' : 'var(--success)' }} />
                <div>
                  <h4 style={{ margin: 0, fontSize: '0.95rem', fontWeight: '700' }}>
                    Current Status: {historyData.latestPrediction.growthStatus} ({historyData.latestPrediction.riskLevel} Malnutrition Risk)
                  </h4>
                  <p style={{ margin: '2px 0 0 0', fontSize: '0.8rem', color: 'var(--dark-light)' }}>
                    Assessments are automatically recalculated using height, weight, and MUAC configurations.
                  </p>
                </div>
              </div>
              <button className="btn btn-secondary btn-sm" onClick={handleOpenLogModal}>
                <Plus size={14} /> Log Growth Parameter
              </button>
            </div>
          )}

          {/* Grid: Graph and Log Table */}
          <div className="grid-2" style={{ gridTemplateColumns: '5fr 3fr' }}>
            
            {/* Chart Area */}
            <Card title="Interactive WHO Growth Charts">
              {loadingHistory ? (
                <Loader size="sm" />
              ) : !hasData ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px', padding: '60px 0' }}>
                  <TrendingUp size={36} className="text-gray-300" />
                  <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem' }}>No logged checks to chart. Please enter measurements.</p>
                  <button className="btn btn-primary btn-sm" onClick={handleOpenLogModal}>Log First Dimension</button>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {/* Tab Selector */}
                  <div style={{ display: 'flex', borderBottom: '1px solid var(--gray-200)', gap: '16px' }}>
                    <button 
                      onClick={() => setActiveTab('height')}
                      style={{ 
                        padding: '10px 16px', 
                        borderBottom: activeTab === 'height' ? '2px solid var(--primary)' : '2px solid transparent',
                        color: activeTab === 'height' ? 'var(--primary)' : 'var(--gray-500)',
                        fontWeight: '700',
                        fontSize: '0.85rem'
                      }}
                    >
                      Height-for-Age
                    </button>
                    <button 
                      onClick={() => setActiveTab('weight')}
                      style={{ 
                        padding: '10px 16px', 
                        borderBottom: activeTab === 'weight' ? '2px solid var(--primary)' : '2px solid transparent',
                        color: activeTab === 'weight' ? 'var(--primary)' : 'var(--gray-500)',
                        fontWeight: '700',
                        fontSize: '0.85rem'
                      }}
                    >
                      Weight-for-Age
                    </button>
                    <button 
                      onClick={() => setActiveTab('bmi')}
                      style={{ 
                        padding: '10px 16px', 
                        borderBottom: activeTab === 'bmi' ? '2px solid var(--primary)' : '2px solid transparent',
                        color: activeTab === 'bmi' ? 'var(--primary)' : 'var(--gray-500)',
                        fontWeight: '700',
                        fontSize: '0.85rem'
                      }}
                    >
                      BMI-for-Age
                    </button>
                  </div>

                  {/* Chart Container */}
                  <div style={{ height: '350px', position: 'relative' }}>
                    <Line data={buildChartData()} options={chartOptions} />
                  </div>
                </div>
              )}
            </Card>

            {/* List Table */}
            <Card title="Measurement Records" headerActions={
              hasData && <button className="btn btn-light btn-sm" onClick={handleOpenLogModal}><Plus size={14} /> Add</button>
            }>
              {loadingHistory ? (
                <Loader size="sm" />
              ) : !hasData ? (
                <p style={{ color: 'var(--gray-500)', textAlign: 'center', padding: '40px 0' }}>No logs registered.</p>
              ) : (
                <div style={{ maxHeight: '420px', overflowY: 'auto' }}>
                  <table className="custom-table" style={{ fontSize: '0.8rem' }}>
                    <thead>
                      <tr>
                        <th style={{ padding: '10px' }}>Date</th>
                        <th style={{ padding: '10px' }}>Age</th>
                        <th style={{ padding: '10px' }}>Ht (cm)</th>
                        <th style={{ padding: '10px' }}>Wt (kg)</th>
                        <th style={{ padding: '10px' }}>MUAC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...historyData.growthRecords].reverse().map((rec) => (
                        <tr key={rec._id}>
                          <td style={{ padding: '10px' }}>{new Date(rec.measurementDate).toLocaleDateString()}</td>
                          <td style={{ padding: '10px' }}>{rec.ageMonths} mo</td>
                          <td style={{ padding: '10px' }}>{rec.height}</td>
                          <td style={{ padding: '10px' }}>{rec.weight}</td>
                          <td style={{ padding: '10px' }}>{rec.MUAC || 'N/A'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Card>

          </div>

        </div>
      ) : (
        <div style={{ 
          backgroundColor: 'var(--white)', 
          borderRadius: 'var(--radius-lg)', 
          padding: '60px 20px', 
          textAlign: 'center',
          boxShadow: 'var(--shadow-sm)'
        }}>
          <Baby size={48} className="text-gray-300" style={{ marginBottom: '16px' }} />
          <h3>No Patient Selected</h3>
          <p style={{ color: 'var(--gray-500)', marginTop: '8px' }}>
            Choose a child from the top right selector to review curves and log dimension logs.
          </p>
        </div>
      )}

      {/* Log growth modal */}
      <Modal isOpen={logModalOpen} onClose={() => setLogModalOpen(false)} title="Add Growth Log Entry">
        {logError && <Alert type="danger" message={logError} />}
        <form onSubmit={handleLogSubmit} className="auth-form" style={{ padding: 0 }}>
          
          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Height (cm)</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                required
                placeholder="e.g. 78.5"
                value={formData.height}
                onChange={(e) => setFormData({ ...formData, height: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Weight (kg)</label>
              <input
                type="number"
                step="0.01"
                className="form-input"
                required
                placeholder="e.g. 9.8"
                value={formData.weight}
                onChange={(e) => setFormData({ ...formData, weight: e.target.value })}
              />
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">MUAC (cm)</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                required
                placeholder="e.g. 12.4"
                value={formData.MUAC}
                onChange={(e) => setFormData({ ...formData, MUAC: e.target.value })}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Head Circumference (cm)</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                required
                placeholder="e.g. 44.0"
                value={formData.headCircumference}
                onChange={(e) => setFormData({ ...formData, headCircumference: e.target.value })}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Measurement Date</label>
            <input
              type="date"
              className="form-input"
              required
              value={formData.measurementDate}
              onChange={(e) => setFormData({ ...formData, measurementDate: e.target.value })}
            />
          </div>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '20px' }}>
            <button type="button" className="btn btn-light" onClick={() => setLogModalOpen(false)}>
              Cancel
            </button>
            <button type="submit" className="btn btn-secondary" disabled={logLoading}>
              {logLoading ? 'Saving...' : 'Add Record'}
            </button>
          </div>
        </form>
      </Modal>

    </div>
  );
};

export default GrowthMonitoring;
