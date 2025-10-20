import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { 
  Shield, 
  Mail, 
  Lock, 
  Eye, 
  EyeOff, 
  AlertCircle, 
  LogIn,
  User,
  Settings,
  Crown
} from 'lucide-react';

const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const { login, error } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) return;
    
    setIsLoading(true);
    try {
      await login(username.trim(), password);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoLogin = async (demoUsername: string, demoPassword: string) => {
    setUsername(demoUsername);
    setPassword(demoPassword);
    setIsLoading(true);
    try {
      await login(demoUsername, demoPassword);
    } finally {
      setIsLoading(false);
    }
  };

  const demoAccounts = [
    {
      role: 'Admin',
      username: 'admin',
      password: 'demo123',
      description: 'Full system access, user management',
      icon: Crown
    },
    {
      role: 'User',
      username: 'user',
      password: 'demo123',
      description: 'Email classification and reporting',
      icon: User
    }
  ];

  return (
    <div className="login-container">
      <div className="login-background">
        <div className="gradient-overlay" />
      </div>
      
      <div className="login-content">
        <div className="login-card">
          <div className="login-header">
            <div className="login-logo">
              <Shield size={32} />
            </div>
            <h1>Phishing Detector</h1>
            <p>Secure email classification system</p>
          </div>

          {error && (
            <div className="error-message">
              <AlertCircle size={16} />
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="login-form">
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <div className="input-wrapper">
                <Mail className="input-icon" size={18} />
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter your username"
                  disabled={isLoading}
                  autoComplete="username"
                />
              </div>
            </div>

            <div className="form-group">
              <label htmlFor="password">Password</label>
              <div className="input-wrapper">
                <Lock className="input-icon" size={18} />
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  disabled={isLoading}
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  className="password-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                  disabled={isLoading}
                >
                  {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              className="login-button"
              disabled={isLoading || !username.trim() || !password.trim()}
            >
              {isLoading ? (
                <>
                  <div className="spinner" style={{ width: 20, height: 20, border: '2px solid rgba(255,255,255,0.3)', borderTop: '2px solid white', borderRadius: '50%' }} />
                  Signing in...
                </>
              ) : (
                <>
                  <LogIn size={18} />
                  Sign In
                </>
              )}
            </button>
          </form>

          <div className="demo-accounts">
            <h3>Demo Accounts</h3>
            <div className="demo-grid">
              {demoAccounts.map((account) => {
                const IconComponent = account.icon;
                return (
                  <div
                    key={account.username}
                    className="demo-card"
                    onClick={() => !isLoading && handleDemoLogin(account.username, account.password)}
                    style={{ opacity: isLoading ? 0.6 : 1, pointerEvents: isLoading ? 'none' : 'auto' }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                      <IconComponent size={16} />
                      <div className="demo-role">{account.role}</div>
                    </div>
                    <div className="demo-username">{account.username}</div>
                    <div className="demo-description">{account.description}</div>
                  </div>
                );
              })}
            </div>
            <div className="demo-note">
              Click any demo account to login instantly with password: <code>demo123</code>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;