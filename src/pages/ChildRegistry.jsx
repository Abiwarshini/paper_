import React, { useState, useEffect, useContext } from 'react';
import { useSearchParams } from 'react-router-dom';
import { AuthContext } from '../context/AuthContext';
import API from '../services/api';
import { 
  Baby, 
  Search, 
  Plus, 
  Edit, 
  Trash2, 
  Eye, 
  X, 
  Activity, 
  Calendar, 
  User, 
  MapPin, 
  Phone, 
  FileText,
  AlertTriangle,
  ChevronLeft
} from 'lucide-react';
import Card from '../components/Card';
import Loader from '../components/Loader';
import Alert from '../components/Alert';
import Modal from '../components/Modal';
import ConfirmationDialog from '../components/ConfirmationDialog';
import SearchBar from '../components/SearchBar';
import Pagination from '../components/Pagination';

const ChildRegistry = () => {
  const { user } = useContext(AuthContext);
  const [searchParams, setSearchParams] = useSearchParams();
  const childIdParam = searchParams.get('childId');

  // List view states
  const [children, setChildren] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [genderFilter, setGenderFilter] = useState('');
  const [page, setPage] = useState(1);
  const [pagination, setPagination] = useState({ currentPage: 1, totalPages: 1, totalRecords: 0 });

  // Detail view states
  const [selectedChildId, setSelectedChildId] = useState(null);
  const [childDetails, setChildDetails] = useState(null);
  const [loadingDetails, setLoadingDetails] = useState(false);

  // Form modal states
  const [modalOpen, setModalOpen] = useState(false);
  const [editingChild, setEditingChild] = useState(null);
  const [formData, setFormData] = useState({
    childName: '',
    dob: '',
    gender: 'Male',
    motherName: '',
    fatherName: '',
    address: '',
    district: '',
    state: '',
    phone: ''
  });
  const [formError, setFormError] = useState('');
  const [formLoading, setFormLoading] = useState(false);

  // Growth Record modal states
  const [growthModalOpen, setGrowthModalOpen] = useState(false);
  const [growthFormData, setGrowthFormData] = useState({
    height: '',
    weight: '',
    MUAC: '',
    headCircumference: '',
    measurementDate: new Date().toISOString().substring(0, 10)
  });
  const [growthLoading, setGrowthLoading] = useState(false);

  // Delete dialog states
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deletingChildId, setDeletingChildId] = useState(null);

  // Load children list
  const fetchChildren = async () => {
    try {
      setLoading(true);
      setError('');
      let endpoint = `/children?page=${page}&limit=8`;
      if (search) endpoint += `&search=${encodeURIComponent(search)}`;
      if (genderFilter) endpoint += `&gender=${genderFilter}`;

      const res = await API.get(endpoint);
      if (res.data.success) {
        setChildren(res.data.data.children || []);
        setPagination(res.data.data.pagination || { currentPage: 1, totalPages: 1, totalRecords: 0 });
      }
    } catch (err) {
      console.error(err);
      setError('Could not retrieve child registry entries.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchChildren();
  }, [page, search, genderFilter]);

  // Load child details if selected
  const fetchChildDetails = async (id) => {
    try {
      setLoadingDetails(true);
      const res = await API.get(`/children/${id}`);
      if (res.data.success) {
        setChildDetails(res.data.data);
      }
    } catch (err) {
      console.error(err);
      setError('Could not retrieve clinical child history.');
    } finally {
      setLoadingDetails(false);
    }
  };

  useEffect(() => {
    if (selectedChildId) {
      fetchChildDetails(selectedChildId);
    } else {
      setChildDetails(null);
    }
  }, [selectedChildId]);

  // Listen to URL changes for childId
  useEffect(() => {
    if (childIdParam) {
      setSelectedChildId(childIdParam);
    } else {
      setSelectedChildId(null);
    }
  }, [childIdParam]);

  const handleSelectChild = (id) => {
    setSearchParams({ childId: id });
  };

  const handleCloseDetails = () => {
    setSearchParams({});
  };

  // Form submit (Create or Edit Child)
  const handleFormChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleOpenAddModal = () => {
    setEditingChild(null);
    setFormData({
      childName: '',
      dob: '',
      gender: 'Male',
      motherName: '',
      fatherName: '',
      address: '',
      district: '',
      state: '',
      phone: ''
    });
    setFormError('');
    setModalOpen(true);
  };

  const handleOpenEditModal = (child) => {
    setEditingChild(child);
    setFormData({
      childName: child.childName,
      dob: new Date(child.dob).toISOString().substring(0, 10),
      gender: child.gender,
      motherName: child.motherName,
      fatherName: child.fatherName,
      address: child.address,
      district: child.district,
      state: child.state,
      phone: child.phone
    });
    setFormError('');
    setModalOpen(true);
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setFormError('');
    setFormLoading(true);

    try {
      if (editingChild) {
        // Update Child
        const res = await API.put(`/children/${editingChild._id}`, formData);
        if (res.data.success) {
          fetchChildren();
          if (selectedChildId === editingChild._id) {
            fetchChildDetails(editingChild._id);
          }
          setModalOpen(false);
        }
      } else {
        // Create Child
        const res = await API.post('/children', formData);
        if (res.data.success) {
          fetchChildren();
          setModalOpen(false);
        }
      }
    } catch (err) {
      setFormError(err.response?.data?.message || 'Form submission failed.');
    } finally {
      setFormLoading(false);
    }
  };

  // Delete Child logic
  const handleOpenDelete = (id, e) => {
    e.stopPropagation();
    setDeletingChildId(id);
    setDeleteOpen(true);
  };

  const handleDeleteConfirm = async () => {
    try {
      const res = await API.delete(`/children/${deletingChildId}`);
      if (res.data.success) {
        setDeleteOpen(false);
        if (selectedChildId === deletingChildId) {
          handleCloseDetails();
        }
        fetchChildren();
      }
    } catch (err) {
      console.error(err);
      setError('Failed to delete child profile.');
    }
  };

  // Growth logs submission
  const handleGrowthSubmit = async (e) => {
    e.preventDefault();
    setGrowthLoading(true);
    try {
      const payload = {
        childId: selectedChildId,
        height: parseFloat(growthFormData.height),
        weight: parseFloat(growthFormData.weight),
        MUAC: parseFloat(growthFormData.MUAC),
        headCircumference: parseFloat(growthFormData.headCircumference),
        measurementDate: growthFormData.measurementDate
      };

      const res = await API.post('/growth', payload);
      if (res.data.success) {
        setGrowthModalOpen(false);
        fetchChildDetails(selectedChildId); // Reload history
      }
    } catch (err) {
      alert(err.response?.data?.message || 'Failed to record growth parameters.');
    } finally {
      setGrowthLoading(false);
    }
  };

  const getRiskBadge = (level) => {
    switch (level) {
      case 'High':
        return <span style={{ backgroundColor: 'var(--danger-light)', color: 'var(--danger)', padding: '4px 8px', borderRadius: '4px', fontWeight: 'bold' }}>High Risk</span>;
      case 'Moderate':
        return <span style={{ backgroundColor: 'var(--warning-light)', color: 'var(--warning)', padding: '4px 8px', borderRadius: '4px', fontWeight: 'bold' }}>Moderate Risk</span>;
      case 'Low':
      default:
        return <span style={{ backgroundColor: 'var(--success-light)', color: 'var(--success)', padding: '4px 8px', borderRadius: '4px', fontWeight: 'bold' }}>Low Risk</span>;
    }
  };

  if (loading && children.length === 0) {
    return <Loader fullScreen />;
  }

  return (
    <div className="registry-container animate-fade-in">
      
      {/* 1. Detail View Panel */}
      {selectedChildId && childDetails ? (
        <div className="detail-panel" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <button onClick={handleCloseDetails} className="btn btn-light" style={{ alignSelf: 'flex-start', padding: '8px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <ChevronLeft size={16} /> Back to Registry List
          </button>

          {loadingDetails ? (
            <Loader />
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
              {/* Header profile details */}
              <div style={{ 
                backgroundColor: 'var(--white)', 
                borderRadius: 'var(--radius-lg)', 
                padding: '24px', 
                boxShadow: 'var(--shadow-sm)',
                display: 'flex',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '20px'
              }}>
                <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                  <div style={{ 
                    width: '60px', 
                    height: '60px', 
                    borderRadius: '50%', 
                    backgroundColor: 'var(--primary-light)', 
                    color: 'var(--primary)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                  }}>
                    <Baby size={32} />
                  </div>
                  <div>
                    <h2 style={{ margin: 0, fontSize: '1.4rem' }}>{childDetails.child.childName}</h2>
                    <p style={{ margin: '4px 0 0 0', color: 'var(--gray-500)', fontSize: '0.9rem' }}>
                      {childDetails.child.gender} | Registered on {new Date(childDetails.child.createdAt).toLocaleDateString()}
                    </p>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                  <button className="btn btn-light" onClick={() => handleOpenEditModal(childDetails.child)}>
                    <Edit size={16} /> Edit Profile
                  </button>
                  <button className="btn btn-secondary" onClick={() => setGrowthModalOpen(true)}>
                    <Plus size={16} /> Log Growth Metrics
                  </button>
                </div>
              </div>

              {/* Detail Grid */}
              <div className="grid-3" style={{ gridTemplateColumns: '1fr 1fr 1fr' }}>
                {/* Profile Card */}
                <Card title="Demographics">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '0.9rem' }}>
                    <div><span className="text-gray-500" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold' }}>Mother's Name:</span><strong>{childDetails.child.motherName}</strong></div>
                    <div><span className="text-gray-500" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold' }}>Father's Name:</span><strong>{childDetails.child.fatherName}</strong></div>
                    <div><span className="text-gray-500" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold' }}>Date of Birth:</span><strong>{new Date(childDetails.child.dob).toLocaleDateString()}</strong></div>
                    <div><span className="text-gray-500" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold' }}>Address:</span><strong>{childDetails.child.address}, {childDetails.child.district}, {childDetails.child.state}</strong></div>
                    <div><span className="text-gray-500" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold' }}>Contact Phone:</span><strong>{childDetails.child.phone}</strong></div>
                  </div>
                </Card>

                {/* Latest Diagnosis */}
                <Card title="Latest AI Health Diagnosis">
                  {childDetails.latestPrediction ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span className="text-gray-500" style={{ fontSize: '0.85rem' }}>Growth Status:</span>
                        <strong style={{ fontSize: '0.95rem' }}>{childDetails.latestPrediction.growthStatus}</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span className="text-gray-500" style={{ fontSize: '0.85rem' }}>System Risk:</span>
                        {getRiskBadge(childDetails.latestPrediction.riskLevel)}
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span className="text-gray-500" style={{ fontSize: '0.85rem' }}>Confidence Index:</span>
                        <strong>{childDetails.latestPrediction.confidence}%</strong>
                      </div>
                      <hr style={{ border: 'none', borderTop: '1px solid var(--gray-100)' }} />
                      <div>
                        <span className="text-gray-500" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold', marginBottom: '4px' }}>Clinical Guidelines:</span>
                        <p style={{ fontSize: '0.8rem', lineHeight: '1.4', margin: 0, color: 'var(--dark-light)' }}>
                          {childDetails.latestPrediction.recommendation}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div style={{ textAlign: 'center', padding: '20px 0' }}>
                      <Activity size={32} style={{ color: 'var(--gray-300)', marginBottom: '8px' }} />
                      <p style={{ color: 'var(--gray-500)', fontSize: '0.85rem' }}>No AI diagnostic metrics recorded yet.</p>
                      <button className="btn btn-light" style={{ fontSize: '0.8rem', padding: '6px 12px', marginTop: '10px' }} onClick={() => setGrowthModalOpen(true)}>
                        Compute First Assessment
                      </button>
                    </div>
                  )}
                </Card>

                {/* Nutrition Plan */}
                <Card title="Dietary & Nutrients Plan">
                  {childDetails.latestPrediction ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', fontSize: '0.85rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span className="text-gray-500">Target Calories:</span>
                        <strong>{childDetails.latestPrediction.growthStatus === 'Severe Malnutrition' ? 1800 : childDetails.latestPrediction.growthStatus === 'Moderate Malnutrition' ? 1500 : 1200} kcal/day</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span className="text-gray-500">Target Protein:</span>
                        <strong>{childDetails.latestPrediction.growthStatus === 'Severe Malnutrition' ? '45g' : childDetails.latestPrediction.growthStatus === 'Moderate Malnutrition' ? '32g' : '22g'}/day</strong>
                      </div>
                      <hr style={{ border: 'none', borderTop: '1px solid var(--gray-100)' }} />
                      <div>
                        <span className="text-gray-500" style={{ display: 'block', fontSize: '0.75rem', fontWeight: 'bold', marginBottom: '4px' }}>Supplements:</span>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '2px' }}>
                          {(childDetails.latestPrediction.growthStatus === 'Severe Malnutrition' ? ['Vitamin A', 'Zinc', 'Iron', 'Folic Acid'] : ['Vitamin A', 'Zinc', 'Vitamin C']).map((vit, idx) => (
                            <span key={idx} style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: '4px', backgroundColor: 'var(--gray-100)', fontWeight: '600' }}>
                              {vit}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p style={{ color: 'var(--gray-500)', fontSize: '0.85rem', textAlign: 'center', padding: '20px 0' }}>Plan is auto-generated upon assessment.</p>
                  )}
                </Card>
              </div>

              {/* History Table */}
              <Card title="Measurement History Log">
                {childDetails.growthHistory.length === 0 ? (
                  <p style={{ color: 'var(--gray-500)', textAlign: 'center', padding: '20px 0' }}>No growth metrics recorded yet.</p>
                ) : (
                  <div className="table-responsive">
                    <table className="custom-table">
                      <thead>
                        <tr>
                          <th>Measurement Date</th>
                          <th>Age (Months)</th>
                          <th>Height (cm)</th>
                          <th>Weight (kg)</th>
                          <th>MUAC (cm)</th>
                          <th>Head Circ (cm)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {childDetails.growthHistory.map((rec) => (
                          <tr key={rec._id}>
                            <td>{new Date(rec.measurementDate).toLocaleDateString()}</td>
                            <td>{rec.ageMonths} months</td>
                            <td>{rec.height} cm</td>
                            <td>{rec.weight} kg</td>
                            <td>{rec.MUAC || 'N/A'}</td>
                            <td>{rec.headCircumference || 'N/A'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </Card>
            </div>
          )}
        </div>
      ) : (
        /* 2. Main List View */
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <h1 style={{ fontSize: '1.6rem' }}>Children Registry Database</h1>
              <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem', margin: '4px 0 0 0' }}>
                Search, filter, and access clinical profiles of enrolled children.
              </p>
            </div>
            <button className="btn btn-secondary" onClick={handleOpenAddModal}>
              <Plus size={18} /> Register Child Profile
            </button>
          </div>

          {/* Filtering and search tools */}
          <div style={{ 
            backgroundColor: 'var(--white)', 
            borderRadius: 'var(--radius-md)', 
            padding: '16px', 
            boxShadow: 'var(--shadow-sm)',
            display: 'flex',
            gap: '16px',
            alignItems: 'center',
            flexWrap: 'wrap'
          }}>
            <div style={{ flex: 1, minWidth: '250px' }}>
              <SearchBar onSearch={(term) => { setSearch(term); setPage(1); }} />
            </div>
            <div>
              <select
                className="form-select"
                style={{ padding: '10px 16px', border: '1px solid var(--gray-200)', borderRadius: 'var(--radius-md)', fontSize: '0.9rem', color: 'var(--dark)' }}
                value={genderFilter}
                onChange={(e) => { setGenderFilter(e.target.value); setPage(1); }}
              >
                <option value="">All Genders</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
          </div>

          {/* Data list */}
          {children.length === 0 ? (
            <div style={{ 
              backgroundColor: 'var(--white)', 
              borderRadius: 'var(--radius-lg)', 
              padding: '60px 20px', 
              textAlign: 'center',
              boxShadow: 'var(--shadow-sm)'
            }}>
              <Baby size={48} className="text-gray-300" style={{ marginBottom: '16px' }} />
              <h3>No Children Records Found</h3>
              <p style={{ color: 'var(--gray-500)', margin: '8px 0 0 0' }}>
                No records match your search query or no profiles are registered yet.
              </p>
            </div>
          ) : (
            <div className="table-responsive" style={{ boxShadow: 'var(--shadow-sm)' }}>
              <table className="custom-table">
                <thead>
                  <tr>
                    <th>Child Name</th>
                    <th>Gender</th>
                    <th>DOB</th>
                    <th>Mother's Name</th>
                    <th>Father's Name</th>
                    <th>District</th>
                    <th>Phone</th>
                    <th style={{ textAlign: 'right' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {children.map((child) => (
                    <tr key={child._id} onClick={() => handleSelectChild(child._id)} style={{ cursor: 'pointer' }}>
                      <td style={{ fontWeight: '600', color: 'var(--primary)' }}>{child.childName}</td>
                      <td>
                        <span className="badge" style={{ 
                          fontSize: '0.75rem', 
                          padding: '3px 6px', 
                          borderRadius: '8px', 
                          backgroundColor: child.gender === 'Male' ? 'var(--primary-light)' : '#ffe8f4',
                          color: child.gender === 'Male' ? 'var(--primary)' : '#ff6eb4'
                        }}>{child.gender}</span>
                      </td>
                      <td>{new Date(child.dob).toLocaleDateString()}</td>
                      <td>{child.motherName}</td>
                      <td>{child.fatherName}</td>
                      <td>{child.district}</td>
                      <td>{child.phone}</td>
                      <td style={{ textAlign: 'right' }} onClick={(e) => e.stopPropagation()}>
                        <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                          <button className="btn btn-light" style={{ padding: '6px 10px' }} onClick={() => handleSelectChild(child._id)}>
                            <Eye size={14} />
                          </button>
                          <button className="btn btn-light" style={{ padding: '6px 10px' }} onClick={() => handleOpenEditModal(child)}>
                            <Edit size={14} />
                          </button>
                          <button className="btn btn-light text-danger" style={{ padding: '6px 10px' }} onClick={(e) => handleOpenDelete(child._id, e)}>
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination controls */}
          <Pagination
            currentPage={pagination.currentPage}
            totalPages={pagination.totalPages}
            totalRecords={pagination.totalRecords}
            limit={pagination.limit}
            onPageChange={(p) => setPage(p)}
          />
        </div>
      )}

      {/* 3. Add/Edit Child Modal */}
      <Modal isOpen={modalOpen} onClose={() => setModalOpen(false)} title={editingChild ? "Edit Child Profile Details" : "Register New Child Profile"}>
        {formError && <Alert type="danger" message={formError} />}
        <form onSubmit={handleFormSubmit} className="auth-form" style={{ padding: 0 }}>
          <div className="form-group">
            <label className="form-label">Child Full Name</label>
            <input
              type="text"
              name="childName"
              className="form-input"
              required
              value={formData.childName}
              onChange={handleFormChange}
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Date of Birth</label>
              <input
                type="date"
                name="dob"
                className="form-input"
                required
                value={formData.dob}
                onChange={handleFormChange}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Gender</label>
              <select
                name="gender"
                className="form-input"
                value={formData.gender}
                onChange={handleFormChange}
              >
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
            </div>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Mother's Name</label>
              <input
                type="text"
                name="motherName"
                className="form-input"
                required
                value={formData.motherName}
                onChange={handleFormChange}
              />
            </div>
            <div className="form-group">
              <label className="form-label">Father's Name</label>
              <input
                type="text"
                name="fatherName"
                className="form-input"
                required
                value={formData.fatherName}
                onChange={handleFormChange}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Address</label>
            <input
              type="text"
              name="address"
              className="form-input"
              required
              value={formData.address}
              onChange={handleFormChange}
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">District</label>
              <input
                type="text"
                name="district"
                className="form-input"
                required
                value={formData.district}
                onChange={handleFormChange}
              />
            </div>
            <div className="form-group">
              <label className="form-label">State</label>
              <input
                type="text"
                name="state"
                className="form-input"
                required
                value={formData.state}
                onChange={handleFormChange}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Contact Phone Number</label>
            <input
              type="tel"
              name="phone"
              className="form-input"
              required
              value={formData.phone}
              onChange={handleFormChange}
            />
          </div>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '20px' }}>
            <button type="button" className="btn btn-light" onClick={() => setModalOpen(false)}>
              Cancel
            </button>
            <button type="submit" className="btn btn-secondary" disabled={formLoading}>
              {formLoading ? 'Submitting...' : (editingChild ? 'Update Profile' : 'Register Child')}
            </button>
          </div>
        </form>
      </Modal>

      {/* 4. Log Growth Parameters Modal */}
      <Modal isOpen={growthModalOpen} onClose={() => setGrowthModalOpen(false)} title="Log Growth Parameters">
        <form onSubmit={handleGrowthSubmit} className="auth-form" style={{ padding: 0 }}>
          <p style={{ color: 'var(--gray-500)', fontSize: '0.85rem', marginBottom: '16px' }}>
            Input height, weight, and mid-upper arm circumference (MUAC). The AI service will evaluate Z-scores and malnutrition risks.
          </p>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Height (cm)</label>
              <input
                type="number"
                step="0.1"
                className="form-input"
                required
                placeholder="e.g. 78.5"
                value={growthFormData.height}
                onChange={(e) => setGrowthFormData({ ...growthFormData, height: e.target.value })}
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
                value={growthFormData.weight}
                onChange={(e) => setGrowthFormData({ ...growthFormData, weight: e.target.value })}
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
                value={growthFormData.MUAC}
                onChange={(e) => setGrowthFormData({ ...growthFormData, MUAC: e.target.value })}
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
                value={growthFormData.headCircumference}
                onChange={(e) => setGrowthFormData({ ...growthFormData, headCircumference: e.target.value })}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Measurement Date</label>
            <input
              type="date"
              className="form-input"
              required
              value={growthFormData.measurementDate}
              onChange={(e) => setGrowthFormData({ ...growthFormData, measurementDate: e.target.value })}
            />
          </div>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '20px' }}>
            <button type="button" className="btn btn-light" onClick={() => setGrowthModalOpen(false)}>
              Cancel
            </button>
            <button type="submit" className="btn btn-secondary" disabled={growthLoading}>
              {growthLoading ? 'Diagnosing...' : 'Log Parameters & Assessment'}
            </button>
          </div>
        </form>
      </Modal>

      {/* 5. Delete Confirmation Dialog */}
      <ConfirmationDialog
        isOpen={deleteOpen}
        title="Delete Child Profile?"
        message="Are you sure you want to remove this child's medical registry and delete all recorded predictions and growth history? This cannot be undone."
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteOpen(false)}
      />

    </div>
  );
};

export default ChildRegistry;
