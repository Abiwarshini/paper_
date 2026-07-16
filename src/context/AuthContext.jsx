import React, { createContext, useState, useEffect } from 'react';
import API from '../services/api';

export const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user credentials exist in storage on mount
    const storedToken = localStorage.getItem('token');
    const storedUser = localStorage.getItem('user');

    if (storedToken && storedUser) {
      setToken(storedToken);
      setUser(JSON.parse(storedUser));
    }
    setLoading(false);
  }, []);

  // Login handler
  const loginUser = async (email, password) => {
    try {
      const res = await API.post('/auth/login', { email, password });
      const { token, user: userData } = res.data.data;
      
      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(userData));
      setToken(token);
      setUser(userData);
      
      return { success: true, user: userData };
    } catch (err) {
      const msg = err.response?.data?.message || 'Login failed';
      const errors = err.response?.data?.errors || [msg];
      return { success: false, message: msg, errors };
    }
  };

  // Register handler (Multipart Form Data for Profile Picture)
  const registerUser = async (formData) => {
    try {
      const res = await API.post('/auth/register', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      const { token, user: userData } = res.data.data;

      localStorage.setItem('token', token);
      localStorage.setItem('user', JSON.stringify(userData));
      setToken(token);
      setUser(userData);

      return { success: true, user: userData };
    } catch (err) {
      const msg = err.response?.data?.message || 'Registration failed';
      const errors = err.response?.data?.errors || [msg];
      return { success: false, message: msg, errors };
    }
  };

  // Logout handler
  const logoutUser = async () => {
    try {
      await API.post('/auth/logout');
    } catch (e) {
      console.warn('Backend logout failed', e);
    }
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setToken(null);
    setUser(null);
  };

  // Forgot password
  const forgotPassword = async (email) => {
    try {
      const res = await API.post('/auth/forgot-password', { email });
      return { success: true, data: res.data };
    } catch (err) {
      const msg = err.response?.data?.message || 'Request failed';
      return { success: false, message: msg };
    }
  };

  // Reset password
  const resetPassword = async (token, password) => {
    try {
      const res = await API.post('/auth/reset-password', { token, password });
      const { token: newToken, user: userData } = res.data.data;

      localStorage.setItem('token', newToken);
      localStorage.setItem('user', JSON.stringify(userData));
      setToken(newToken);
      setUser(userData);

      return { success: true, user: userData };
    } catch (err) {
      const msg = err.response?.data?.message || 'Reset failed';
      return { success: false, message: msg };
    }
  };

  // Update profile in local state
  const updateLocalUser = (updatedData) => {
    const newUser = { ...user, ...updatedData };
    localStorage.setItem('user', JSON.stringify(newUser));
    setUser(newUser);
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        loading,
        loginUser,
        registerUser,
        logoutUser,
        forgotPassword,
        resetPassword,
        updateLocalUser,
        isAuthenticated: !!token
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};
