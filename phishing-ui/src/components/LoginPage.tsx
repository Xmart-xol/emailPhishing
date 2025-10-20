import React, { useState } from 'react';
import { Shield, User, Lock, AlertCircle, Loader2 } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const LoginPage: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const success = await login(username, password);
    if (!success) {
      setError('Invalid username or password');
    }
    setLoading(false);
  };

  const demoUsers = [
    { username: 'admin', role: 'System Administrator', description: 'Full system access, user management, system configuration' },
    { username: 'researcher', role: 'Security Researcher', description: 'Advanced analytics, model training, threat intelligence' },
    { username: 'user', role: 'End User', description: 'Email classification, basic reporting' }
  ];

  return (
    <div className="login-page">
      <div className="login-container">
        <div className="login-header">
          <Shield size={48} className="login-icon" />
          <h1>Phishing Detection System</h1>
          <p>Advanced AI-powered email security platform</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label htmlFor="username">
              <User size={18} />
              Username
            </label>
            <input
              id="username"
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              required
              disabled={loading}
            />
          </div>

          <div className="form-group">
            <label htmlFor="password">
              <Lock size={18} />
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              required
              disabled={loading}
            />
          </div>

          {error && (
            <div className="error-message">
              <AlertCircle size={18} />
              {error}
            </div>
          )}

          <button type="submit" disabled={loading} className="login-button">
            {loading ? (
              <>
                <Loader2 className="spinner" size={18} />
                Signing in...
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <div className="demo-section">
          <h3>Demo Accounts</h3>
          <p>Use password: <strong>demo123</strong></p>
          <div className="demo-users">
            {demoUsers.map((user) => (
              <div key={user.username} className="demo-user-card">
                <div className="demo-user-header">
                  <strong>{user.username}</strong>
                  <span className="demo-user-role">{user.role}</span>
                </div>
                <p className="demo-user-description">{user.description}</p>
                <button
                  type="button"
                  onClick={() => {
                    setUsername(user.username);
                    setPassword('demo123');
                  }}
                  className="demo-user-button"
                  disabled={loading}
                >
                  Use This Account
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;