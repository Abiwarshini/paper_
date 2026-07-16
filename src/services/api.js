import axios from 'axios';

const API = axios.create({
  baseURL: 'http://localhost:5000/api',
  headers: {
    'Content-Type': 'application/json'
  }
});

// Interceptor to inject JWT Token in headers
API.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Interceptor to handle global errors (like token expiry)
API.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Auto-clear credentials on session invalidation
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      // If we are not on public auth pages, redirect to login
      const publicPaths = ['/login', '/register', '/forgot-password', '/reset-password', '/'];
      if (!publicPaths.includes(window.location.pathname) && !window.location.pathname.startsWith('/reset-password/')) {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export default API;
