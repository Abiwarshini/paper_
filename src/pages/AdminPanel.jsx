import React, { useState, useEffect, useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import API from '../services/api';
import { 
  Users, 
  Plus, 
  Edit, 
  Trash2, 
  Mail, 
  Phone, 
  ShieldAlert, 
  Search,
  Lock,
  UserCheck
} from 'lucide-react';
import Card from '../components/Card';
import Loader from '../components/Loader';
import Alert from '../components/Alert';
import Modal from '../components/Modal';
import ConfirmationDialog from '../components/ConfirmationDialog';
import SearchBar from '../components/SearchBar';

const AdminPanel = () => {
  const { user: currentUser } = useContext(AuthContext);

  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  
  // Search & Filters
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('');

  // Modal states
  const [modalOpen, setModalOpen] = useState(false);
  const [editingUser, setEditingUser] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    phone: '',
    role: 'Parent',
    password: ''
  });
  const [formError, setFormError] = useState('');
  const [formLoading, setFormLoading] = useState(false);

  // Delete confirmation state
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deletingUserId, setDeletingUserId] = useState(null);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      setError('');
      let endpoint = '/admin/users?';
      if (search) endpoint += `&search=${encodeURIComponent(search)}`;
      if (roleFilter) endpoint += `&role=${roleFilter}`;

      const res = await API.get(endpoint);
      if (res.data.success) {
        setUsers(res.data.data || []);
      }
    } catch (err) {
      console.error(err);
      setError('Could not retrieve user directory.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsers();
  }, [search, roleFilter]);

  const handleOpenAddModal = () => {
    setEditingUser(null);
    setFormData({
      name: '',
      email: '',
      phone: '',
      role: 'Parent',
      password: ''
    });
    setFormError('');
    setModalOpen(true);
  };

  const handleOpenEditModal = (usr) => {
    setEditingUser(usr);
    setFormData({
      name: usr.name,
      email: usr.email,
      phone: usr.phone,
      role: usr.role,
      password: '' // empty password fields mean no change
    });
    setFormError('');
    setModalOpen(true);
  };

  const handleFormChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setFormError('');
    setFormLoading(true);

    try {
      const payload = { ...formData };
      if (editingUser && !payload.password) {
        delete payload.password; // Do not send empty password during edit
      }

      if (editingUser) {
        // Edit User
        const res = await API.put(`/admin/users/${editingUser._id}`, payload);
        if (res.data.success) {
          fetchUsers();
          setModalOpen(false);
        }
      } else {
        // Create User
        if (!payload.password) {
          setFormError('Password is required for new accounts.');
          setFormLoading(false);
          return;
        }
        const res = await API.post('/admin/users', payload);
        if (res.data.success) {
          fetchUsers();
          setModalOpen(false);
        }
      }
    } catch (err) {
      setFormError(err.response?.data?.message || 'Failed to submit user details.');
    } finally {
      setFormLoading(false);
    }
  };

  const handleOpenDelete = (id) => {
    setDeletingUserId(id);
    setDeleteOpen(true);
  };

  const handleDeleteConfirm = async () => {
    try {
      const res = await API.delete(`/admin/users/${deletingUserId}`);
      if (res.data.success) {
        setDeleteOpen(false);
        fetchUsers();
      }
    } catch (err) {
      alert(err.response?.data?.message || 'Failed to delete user.');
    }
  };

  const getRoleBadgeStyle = (role) => {
    switch (role) {
      case 'Admin': return { backgroundColor: 'var(--danger-light)', color: 'var(--danger)' };
      case 'Doctor': return { backgroundColor: 'var(--primary-light)', color: 'var(--primary)' };
      case 'Health Worker': return { backgroundColor: 'var(--secondary-light)', color: 'var(--secondary)' };
      case 'Parent':
      default:
        return { backgroundColor: 'var(--gray-100)', color: 'var(--gray-500)' };
    }
  };

  return (
    <div className="admin-wrapper animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      {/* Title */}
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
          <h1 style={{ fontSize: '1.6rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <UserCheck className="text-secondary" /> Administrative Directory
          </h1>
          <p style={{ color: 'var(--gray-500)', fontSize: '0.9rem', margin: '4px 0 0 0' }}>
            Manage staff profiles, permissions, passwords, and registered parent accounts.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={handleOpenAddModal}>
          <Plus size={18} /> Add User Account
        </button>
      </div>

      {error && <Alert type="danger" message={error} />}

      {/* Filter and Search Bar */}
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
          <SearchBar onSearch={(term) => setSearch(term)} placeholder="Search users by name, email, or phone..." />
        </div>
        <div>
          <select
            className="form-select"
            style={{ padding: '10px 16px', border: '1px solid var(--gray-200)', borderRadius: 'var(--radius-md)', fontSize: '0.9rem', color: 'var(--dark)' }}
            value={roleFilter}
            onChange={(e) => setRoleFilter(e.target.value)}
          >
            <option value="">All Roles</option>
            <option value="Admin">Admin</option>
            <option value="Doctor">Doctor</option>
            <option value="Health Worker">Health Worker</option>
            <option value="Parent">Parent</option>
          </select>
        </div>
      </div>

      {/* Table grid list */}
      {loading && users.length === 0 ? (
        <Loader />
      ) : users.length === 0 ? (
        <div style={{ 
          backgroundColor: 'var(--white)', 
          borderRadius: 'var(--radius-lg)', 
          padding: '60px 20px', 
          textAlign: 'center',
          boxShadow: 'var(--shadow-sm)'
        }}>
          <Users size={48} className="text-gray-300" style={{ marginBottom: '16px' }} />
          <h3>No User Profiles Recorded</h3>
          <p style={{ color: 'var(--gray-500)', marginTop: '8px' }}>
            No accounts match your current lookup parameters.
          </p>
        </div>
      ) : (
        <div className="table-responsive" style={{ boxShadow: 'var(--shadow-sm)' }}>
          <table className="custom-table">
            <thead>
              <tr>
                <th>Profile Name</th>
                <th>Role Badge</th>
                <th>Email Address</th>
                <th>Contact Phone</th>
                <th>Registered Date</th>
                <th style={{ textAlign: 'right' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {users.map((usr) => (
                <tr key={usr._id}>
                  <td style={{ fontWeight: '600' }}>
                    {usr.name} {usr._id === currentUser?.id && <span style={{ fontSize: '0.75rem', color: 'var(--primary)', fontWeight: 'bold' }}>(You)</span>}
                  </td>
                  <td>
                    <span className="badge" style={{ 
                      fontSize: '0.72rem', 
                      padding: '3px 8px', 
                      borderRadius: '8px',
                      fontWeight: '700',
                      ...getRoleBadgeStyle(usr.role)
                    }}>
                      {usr.role}
                    </span>
                  </td>
                  <td>{usr.email}</td>
                  <td>{usr.phone}</td>
                  <td>{new Date(usr.createdAt).toLocaleDateString()}</td>
                  <td style={{ textAlign: 'right' }}>
                    <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end' }}>
                      <button className="btn btn-light" style={{ padding: '6px 10px' }} onClick={() => handleOpenEditModal(usr)}>
                        <Edit size={14} />
                      </button>
                      <button 
                        className="btn btn-light text-danger" 
                        style={{ padding: '6px 10px' }} 
                        onClick={() => handleOpenDelete(usr._id)}
                        disabled={usr._id === currentUser?.id}
                        title={usr._id === currentUser?.id ? "Admins cannot delete their own active accounts" : "Delete User"}
                      >
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

      {/* User Form Modal */}
      <Modal isOpen={modalOpen} onClose={() => setModalOpen(false)} title={editingUser ? "Edit User Account details" : "Create New User Account"}>
        {formError && <Alert type="danger" message={formError} />}
        <form onSubmit={handleFormSubmit} className="auth-form" style={{ padding: 0 }}>
          
          <div className="form-group">
            <label className="form-label">Full Name</label>
            <input
              type="text"
              name="name"
              className="form-input"
              required
              value={formData.name}
              onChange={handleFormChange}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Email Address</label>
            <input
              type="email"
              name="email"
              className="form-input"
              required
              value={formData.email}
              onChange={handleFormChange}
            />
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Role Privilege</label>
              <select
                name="role"
                className="form-input"
                value={formData.role}
                onChange={handleFormChange}
              >
                <option value="Parent">Parent</option>
                <option value="Health Worker">Health Worker</option>
                <option value="Doctor">Doctor</option>
                <option value="Admin">Admin</option>
              </select>
            </div>
            <div className="form-group">
              <label className="form-label">Phone Number</label>
              <input
                type="tel"
                name="phone"
                className="form-input"
                required
                value={formData.phone}
                onChange={handleFormChange}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">
              {editingUser ? "Password (Leave blank to keep unchanged)" : "Password"}
            </label>
            <input
              type="password"
              name="password"
              className="form-input"
              placeholder={editingUser ? "••••••••" : "Type a secure password"}
              required={!editingUser}
              value={formData.password}
              onChange={handleFormChange}
            />
          </div>

          <div style={{ display: 'flex', gap: '12px', justifyContent: 'flex-end', marginTop: '20px' }}>
            <button type="button" className="btn btn-light" onClick={() => setModalOpen(false)}>
              Cancel
            </button>
            <button type="submit" className="btn btn-secondary" disabled={formLoading}>
              {formLoading ? 'Saving...' : 'Save User Account'}
            </button>
          </div>
        </form>
      </Modal>

      {/* Delete Confirmation */}
      <ConfirmationDialog
        isOpen={deleteOpen}
        title="Delete User Account?"
        message="Are you sure you want to delete this user profile? This action will permanently remove their credentials and revoke access. It cannot be undone."
        onConfirm={handleDeleteConfirm}
        onCancel={() => setDeleteOpen(false)}
      />

    </div>
  );
};

export default AdminPanel;
