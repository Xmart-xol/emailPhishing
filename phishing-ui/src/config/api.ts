export const API_CONFIG = {
  BASE_URL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000',
  ENDPOINTS: {
    CLASSIFY: '/api/classify',
    HISTORY: '/api/classify/history', 
    ANALYTICS: '/api/analytics',
    ADMIN: '/api/admin',
    DASHBOARD_STATS: '/api/analytics/dashboard/stats',
    TRENDS: '/api/analytics/trends'
  }
};

export const getApiUrl = (endpoint: string) => `${API_CONFIG.BASE_URL}${endpoint}`;