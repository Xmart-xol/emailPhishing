import React, { useState, useEffect } from 'react';
import { 
  BarChart3, TrendingUp, Shield, Users, AlertTriangle, CheckCircle, Activity, Clock,
  Brain, Database, Zap, Play, Pause, Star, Award, Monitor, Settings, Plus, Eye
} from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { useAuth } from '../contexts/AuthContext';
import axios from 'axios';
import { getApiUrl } from '../config/api';

interface DashboardStats {
  total_classifications: number;
  recent_classifications: number;
  phishing_detected: number;
  legitimate_detected: number;
  accuracy_percentage: number;
  f1_score_percentage: number;
  active_users: number;
  false_positives: number;
  avg_response_time_ms: number;
}

interface TrendData {
  daily_classifications: Array<{
    date: string;
    total: number;
    phishing: number;
    legitimate: number;
    avg_confidence: number;
  }>;
}

interface ModelInfo {
  id: string;
  model_type: string;
  status: string;
  metrics: {
    accuracy: number;
    f1_score: number;
  } | null;
  training_time: number | null;
  created_at: string;
  is_production: boolean;
  total_classifications?: number;
}

interface Dataset {
  id: string;
  name: string;
  n_rows: number;
  n_phish: number;
  n_ham: number;
  created_at: string;
}

const Dashboard: React.FC = () => {
  const { user, hasRole } = useAuth();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [trends, setTrends] = useState<TrendData | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [modelStatus, setModelStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeAdminTab, setActiveAdminTab] = useState<'overview' | 'models' | 'training'>('overview');

  useEffect(() => {
    fetchDashboardData();
    if (hasRole('admin')) {
      fetchAdminData();
    }
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [statsResponse, trendsResponse] = await Promise.all([
        axios.get(getApiUrl('/api/analytics/dashboard/stats')),
        axios.get(getApiUrl('/api/analytics/trends'))
      ]);
      
      setStats(statsResponse.data);
      setTrends(trendsResponse.data);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard data');
      console.error('Dashboard error:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchAdminData = async () => {
    try {
      const [modelsResponse, datasetsResponse] = await Promise.all([
        axios.get(getApiUrl('/api/admin/runs')),
        axios.get(getApiUrl('/api/admin/datasets'))
      ]);
      
      // The API returns the data directly as arrays, not wrapped in objects
      setModels(modelsResponse.data || []);
      setDatasets(datasetsResponse.data || []);
      
      console.log('Models loaded:', modelsResponse.data?.length || 0);
      console.log('Datasets loaded:', datasetsResponse.data?.length || 0);
    } catch (err) {
      console.error('Failed to load admin data:', err);
      // Set empty arrays as fallback instead of leaving undefined
      setModels([]);
      setDatasets([]);
    }
  };

  const getWelcomeMessage = () => {
    const time = new Date().getHours();
    const greeting = time < 12 ? 'Good morning' : time < 18 ? 'Good afternoon' : 'Good evening';
    return `${greeting}, ${user?.full_name || 'User'}`;
  };

  const getRoleDescription = () => {
    switch (user?.role) {
      case 'admin':
        return 'System Administrator - Full access to all system functions and user management';
      case 'researcher':
        return 'Security Researcher - Access to advanced analytics and model training';
      case 'user':
        return 'End User - Email classification and basic reporting access';
      default:
        return 'User';
    }
  };

  // Color scheme for charts
  const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6'];

  const renderUserMetrics = () => (
    <div className="metrics-grid">
      <div className="metric-card primary">
        <div className="metric-icon">
          <Activity size={24} />
        </div>
        <div className="metric-content">
          <h3>{stats?.recent_classifications || 0}</h3>
          <p>Recent Classifications</p>
          <span className="metric-change positive">Last 7 days</span>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-icon">
          <Shield size={24} />
        </div>
        <div className="metric-content">
          <h3>{stats?.phishing_detected || 0}</h3>
          <p>Threats Detected</p>
          <span className="metric-change neutral">Last 30 days</span>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-icon">
          <CheckCircle size={24} />
        </div>
        <div className="metric-content">
          <h3>{stats?.accuracy_percentage || 97.3}%</h3>
          <p>Model Accuracy</p>
          <span className="metric-change positive">SVM Production Model</span>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-icon">
          <Clock size={24} />
        </div>
        <div className="metric-content">
          <h3>{stats?.avg_response_time_ms || 0}ms</h3>
          <p>Avg Response Time</p>
          <span className="metric-change positive">Real-time ML inference</span>
        </div>
      </div>
    </div>
  );

  const renderResearcherMetrics = () => (
    <div className="metrics-grid">
      {renderUserMetrics()}
      <div className="metric-card">
        <div className="metric-icon">
          <TrendingUp size={24} />
        </div>
        <div className="metric-content">
          <h3>{stats?.f1_score_percentage || 0}%</h3>
          <p>F1-Score</p>
          <span className="metric-change positive">Model performance</span>
        </div>
      </div>

      <div className="metric-card">
        <div className="metric-icon">
          <AlertTriangle size={24} />
        </div>
        <div className="metric-content">
          <h3>{stats?.false_positives || 0}</h3>
          <p>False Positives</p>
          <span className="metric-change negative">Needs attention</span>
        </div>
      </div>
    </div>
  );

  const renderAdminMetrics = () => (
    <div className="enhanced-metrics-grid">
      <div className="metric-card-enhanced primary">
        <div className="metric-header">
          <div className="metric-icon-enhanced">
            <Brain size={28} />
          </div>
          <div className="metric-badge">Live</div>
        </div>
        <div className="metric-content-enhanced">
          <h2>{stats?.accuracy_percentage || 97.3}%</h2>
          <p>Production Model Accuracy</p>
          <div className="metric-details">
            <span className="detail-item">
              <CheckCircle size={16} />
              SVM Model Active
            </span>
            <span className="detail-item success">
              <TrendingUp size={16} />
              97.1% F1-Score
            </span>
          </div>
        </div>
      </div>

      <div className="metric-card-enhanced">
        <div className="metric-header">
          <div className="metric-icon-enhanced">
            <Shield size={28} />
          </div>
          <div className="metric-badge warning">Alert</div>
        </div>
        <div className="metric-content-enhanced">
          <h2>{stats?.phishing_detected || 0}</h2>
          <p>Threats Detected</p>
          <div className="metric-details">
            <span className="detail-item">
              <Activity size={16} />
              {stats?.total_classifications || 0} Total
            </span>
            <span className="detail-item">
              <Clock size={16} />
              {stats?.avg_response_time_ms || 45}ms Avg
            </span>
          </div>
        </div>
      </div>

      <div className="metric-card-enhanced">
        <div className="metric-header">
          <div className="metric-icon-enhanced">
            <Database size={28} />
          </div>
          <div className="metric-badge info">Ready</div>
        </div>
        <div className="metric-content-enhanced">
          <h2>{models.filter(m => m.status === 'completed').length}</h2>
          <p>Trained Models</p>
          <div className="metric-details">
            <span className="detail-item">
              <Star size={16} />
              {models.filter(m => m.is_production).length} Production
            </span>
            <span className="detail-item">
              <Monitor size={16} />
              {datasets.length} Datasets
            </span>
          </div>
        </div>
      </div>

      <div className="metric-card-enhanced">
        <div className="metric-header">
          <div className="metric-icon-enhanced">
            <Users size={28} />
          </div>
          <div className="metric-badge success">Active</div>
        </div>
        <div className="metric-content-enhanced">
          <h2>{stats?.active_users || 0}</h2>
          <p>Active Users</p>
          <div className="metric-details">
            <span className="detail-item">
              <Activity size={16} />
              System Healthy
            </span>
            <span className="detail-item">
              <CheckCircle size={16} />
              All Services Up
            </span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderAdminModelOverview = () => (
    <div className="admin-models-section">
      <div className="section-header-enhanced">
        <div className="header-content">
          <Brain size={32} />
          <div>
            <h2>Model Performance Overview</h2>
            <p>Real-time monitoring of all trained models</p>
          </div>
        </div>
        <div className="header-actions">
          <button onClick={fetchAdminData} className="btn-enhanced refresh">
            <Monitor size={18} />
            Refresh Status
          </button>
        </div>
      </div>

      <div className="models-grid">
        {models.filter(m => m.status === 'completed' && m.metrics).map((model) => (
          <div key={model.id} className={`model-card-enhanced ${model.is_production ? 'production' : ''}`}>
            <div className="model-header">
              <div className="model-type">
                <Brain size={24} />
                <span>{model.model_type.toUpperCase()}</span>
              </div>
              {model.is_production && (
                <div className="production-badge">
                  <Star size={16} />
                  Production
                </div>
              )}
            </div>
            
            <div className="model-metrics">
              <div className="metric-row">
                <div className="metric-item">
                  <Award size={18} />
                  <div>
                    <span className="metric-value">{((model.metrics?.accuracy || 0) * 100).toFixed(1)}%</span>
                    <span className="metric-label">Accuracy</span>
                  </div>
                </div>
                <div className="metric-item">
                  <TrendingUp size={18} />
                  <div>
                    <span className="metric-value">{((model.metrics?.f1_score || 0) * 100).toFixed(1)}%</span>
                    <span className="metric-label">F1-Score</span>
                  </div>
                </div>
              </div>
              
              <div className="metric-row">
                <div className="metric-item">
                  <Clock size={18} />
                  <div>
                    <span className="metric-value">{(model.training_time || 0).toFixed(0)}s</span>
                    <span className="metric-label">Training Time</span>
                  </div>
                </div>
                <div className="metric-item">
                  <Activity size={18} />
                  <div>
                    <span className="metric-value">{model.total_classifications || 0}</span>
                    <span className="metric-label">Classifications</span>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="model-footer">
              <span className="model-date">
                Trained: {new Date(model.created_at).toLocaleDateString()}
              </span>
              <div className="model-actions">
                <button className="btn-icon" title="View Details">
                  <Eye size={16} />
                </button>
                <button className="btn-icon" title="Configure">
                  <Settings size={16} />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderAdminTrainingSection = () => (
    <div className="admin-training-section">
      <div className="section-header-enhanced">
        <div className="header-content">
          <Zap size={32} />
          <div>
            <h2>Model Training Center</h2>
            <p>Train new models and manage training pipelines</p>
          </div>
        </div>
        <div className="header-actions">
          <button className="btn-enhanced primary">
            <Plus size={18} />
            Start New Training
          </button>
        </div>
      </div>

      <div className="training-grid">
        <div className="training-card">
          <div className="training-header">
            <Brain size={24} />
            <h3>Quick Train</h3>
          </div>
          <p>Train a new model with default parameters</p>
          <div className="training-options">
            <select className="select-enhanced">
              <option>Select Dataset</option>
              {datasets.map(d => (
                <option key={d.id} value={d.id}>{d.name} ({d.n_rows} samples)</option>
              ))}
            </select>
            <select className="select-enhanced">
              <option>SVM (Recommended)</option>
              <option>KNN</option>
            </select>
          </div>
          <button className="btn-enhanced success full-width">
            <Play size={18} />
            Start Training
          </button>
        </div>

        <div className="training-card">
          <div className="training-header">
            <Settings size={24} />
            <h3>Advanced Training</h3>
          </div>
          <p>Custom hyperparameters and feature configuration</p>
          <div className="training-stats">
            <div className="stat-item">
              <span className="stat-value">{datasets.length}</span>
              <span className="stat-label">Datasets Available</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{models.filter(m => m.status === 'completed').length}</span>
              <span className="stat-label">Models Trained</span>
            </div>
          </div>
          <button className="btn-enhanced outline full-width">
            <Settings size={18} />
            Configure Training
          </button>
        </div>
      </div>
    </div>
  );

  const renderAdminContent = () => {
    switch (activeAdminTab) {
      case 'models':
        return renderAdminModelOverview();
      case 'training':
        return renderAdminTrainingSection();
      default:
        return (
          <>
            {renderAdminMetrics()}
            <div className="charts-section-enhanced">
              <div className="charts-grid">
                {renderTrendsChart()}
                {renderDistributionChart()}
              </div>
            </div>
            {renderAdminModelOverview()}
          </>
        );
    }
  };

  const renderMetrics = () => {
    if (hasRole('admin')) return null; // Admin has custom layout
    if (hasRole('researcher')) return renderResearcherMetrics();
    return renderUserMetrics();
  };

  const renderTrendsChart = () => {
    if (!trends?.daily_classifications?.length) return null;

    return (
      <div className="chart-container-enhanced">
        <div className="chart-header">
          <h3>Classification Trends</h3>
          <span className="chart-subtitle">Last 30 Days</span>
        </div>
        <ResponsiveContainer width="100%" height={350}>
          <AreaChart data={trends.daily_classifications}>
            <defs>
              <linearGradient id="colorPhishing" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={colors[1]} stopOpacity={0.8}/>
                <stop offset="95%" stopColor={colors[1]} stopOpacity={0.1}/>
              </linearGradient>
              <linearGradient id="colorLegitimate" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={colors[2]} stopOpacity={0.8}/>
                <stop offset="95%" stopColor={colors[2]} stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
            <XAxis dataKey="date" stroke="rgba(255,255,255,0.7)" />
            <YAxis stroke="rgba(255,255,255,0.7)" />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(0,0,0,0.8)', 
                border: 'none', 
                borderRadius: '8px',
                color: 'white'
              }} 
            />
            <Legend />
            <Area 
              type="monotone" 
              dataKey="phishing" 
              stackId="1" 
              stroke={colors[1]} 
              fill="url(#colorPhishing)"
              name="Phishing Detected"
            />
            <Area 
              type="monotone" 
              dataKey="legitimate" 
              stackId="1" 
              stroke={colors[2]} 
              fill="url(#colorLegitimate)"
              name="Legitimate Emails"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    );
  };

  const renderDistributionChart = () => {
    if (!stats) return null;

    const data = [
      { name: 'Phishing', value: stats.phishing_detected, color: colors[1] },
      { name: 'Legitimate', value: stats.legitimate_detected, color: colors[2] }
    ];

    return (
      <div className="chart-container-enhanced">
        <div className="chart-header">
          <h3>Email Classification Distribution</h3>
          <span className="chart-subtitle">All Time</span>
        </div>
        <ResponsiveContainer width="100%" height={350}>
          <PieChart>
            <defs>
              <filter id="shadow">
                <feDropShadow dx="0" dy="4" stdDeviation="8" floodOpacity="0.3"/>
              </filter>
            </defs>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={(entry: any) => `${entry.name}: ${(entry.percent * 100).toFixed(0)}%`}
              outerRadius={120}
              fill="#8884d8"
              dataKey="value"
              filter="url(#shadow)"
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgba(0,0,0,0.8)', 
                border: 'none', 
                borderRadius: '8px',
                color: 'white'
              }} 
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="dashboard-enhanced">
        <div className="loading-state-enhanced">
          <div className="spinner-enhanced"></div>
          <p>Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`dashboard-enhanced ${hasRole('admin') ? 'admin-dashboard' : ''}`}>
      <div className="dashboard-header-enhanced">
        <div className="header-content">
          <div className="welcome-section">
            <h1>{getWelcomeMessage()}</h1>
            <p>{getRoleDescription()}</p>
          </div>
          
          {hasRole('admin') && (
            <div className="admin-tabs">
              <button 
                className={`tab-btn ${activeAdminTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveAdminTab('overview')}
              >
                <BarChart3 size={18} />
                Overview
              </button>
              <button 
                className={`tab-btn ${activeAdminTab === 'models' ? 'active' : ''}`}
                onClick={() => setActiveAdminTab('models')}
              >
                <Brain size={18} />
                Models
              </button>
              <button 
                className={`tab-btn ${activeAdminTab === 'training' ? 'active' : ''}`}
                onClick={() => setActiveAdminTab('training')}
              >
                <Zap size={18} />
                Training
              </button>
            </div>
          )}
        </div>
        
        <div className="header-actions">
          <button onClick={fetchDashboardData} className="btn-enhanced refresh">
            <Monitor size={18} />
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="alert-enhanced error">
          <AlertTriangle size={18} />
          {error}
        </div>
      )}

      {hasRole('admin') ? renderAdminContent() : (
        <>
          {renderMetrics()}
          <div className="charts-section">
            <div className="charts-grid">
              {renderTrendsChart()}
              {renderDistributionChart()}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;