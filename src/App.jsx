import React, { useState, useContext } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Outlet } from 'react-router-dom';
import { AuthContext } from './context/AuthContext';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Loader from './components/Loader';

// Pages
import Landing from './pages/Landing';
import Login from './pages/Login';
import Register from './pages/Register';
import ForgotPassword from './pages/ForgotPassword';
import ResetPassword from './pages/ResetPassword';
import Dashboard from './pages/Dashboard';
import ChildRegistry from './pages/ChildRegistry';
import GrowthMonitoring from './pages/GrowthMonitoring';
import Predictor from './pages/Predictor';
import Reports from './pages/Reports';
import AdminPanel from './pages/AdminPanel';
import Profile from './pages/Profile';

// Route guard for authenticated users
const ProtectedLayout = () => {
  const { isAuthenticated, loading } = useContext(AuthContext);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  if (loading) {
    return <Loader fullScreen />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return (
    <div className="app-layout">
      <Navbar onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
      <div className="app-body-wrapper">
        <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        <main className="app-main-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

// Route guard for public auth pages (redirects to dashboard if already logged in)
const PublicRoute = ({ children }) => {
  const { isAuthenticated, loading } = useContext(AuthContext);

  if (loading) {
    return <Loader fullScreen />;
  }

  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
};

// Route guard for Admins only
const AdminRoute = () => {
  const { user, isAuthenticated, loading } = useContext(AuthContext);

  if (loading) return <Loader fullScreen />;
  if (!isAuthenticated) return <Navigate to="/login" replace />;
  if (user?.role !== 'Admin') return <Navigate to="/dashboard" replace />;

  return <Outlet />;
};

function App() {
  return (
    <Router>
      <Routes>
        {/* Public Landing Page */}
        <Route path="/" element={<Landing />} />

        {/* Public Authentication Shell */}
        <Route path="/login" element={
          <PublicRoute>
            <Login />
          </PublicRoute>
        } />
        <Route path="/register" element={
          <PublicRoute>
            <Register />
          </PublicRoute>
        } />
        <Route path="/forgot-password" element={
          <PublicRoute>
            <ForgotPassword />
          </PublicRoute>
        } />
        <Route path="/reset-password/:token" element={
          <PublicRoute>
            <ResetPassword />
          </PublicRoute>
        } />

        {/* Protected Dashboard Features */}
        <Route element={<ProtectedLayout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/children" element={<ChildRegistry />} />
          <Route path="/growth-monitoring" element={<GrowthMonitoring />} />
          <Route path="/predict" element={<Predictor />} />
          <Route path="/profile" element={<Profile />} />

          {/* Admin Managed Pages */}
          <Route element={<AdminRoute />}>
            <Route path="/admin" element={<AdminPanel />} />
          </Route>
          
          {/* Shared Admin/Doctor/Health Worker Pages */}
          <Route path="/reports" element={<Reports />} />
        </Route>

        {/* Catch-all Fallback Redirect */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
