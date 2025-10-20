import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Shield, Home, BarChart3, Settings, User, Brain, LogOut, Mail } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

const Navbar: React.FC = () => {
  const location = useLocation();
  const { user, logout, hasRole } = useAuth();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  const getNavigationItems = () => {
    const items = [
      {
        path: '/dashboard',
        icon: Home,
        label: 'Dashboard',
        roles: ['admin', 'researcher', 'user']
      },
      {
        path: '/classify',
        icon: Mail,
        label: 'Email Detection',
        roles: ['admin', 'researcher', 'user']
      }
    ];

    // Add role-specific items
    if (hasRole('admin') || hasRole('researcher')) {
      items.push({
        path: '/analytics',
        icon: BarChart3,
        label: 'Analytics',
        roles: ['admin', 'researcher']
      });
    }

    if (hasRole('admin') || hasRole('researcher')) {
      items.push({
        path: '/training',
        icon: Brain,
        label: 'Model Training',
        roles: ['admin', 'researcher']
      });
    }

    if (hasRole('admin')) {
      items.push({
        path: '/admin',
        icon: Settings,
        label: 'Administration',
        roles: ['admin']
      });
    }

    return items.filter(item => 
      item.roles.includes(user?.role || '')
    );
  };

  const navigationItems = getNavigationItems();

  return (
    <nav className="navbar">
      <div className="nav-brand">
        <Shield className="nav-icon" />
        <div className="brand-text">
          <h1>PhishGuard</h1>
          <span>AI Security Platform</span>
        </div>
      </div>

      <div className="nav-center">
        <div className="nav-links">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            return (
              <Link 
                key={item.path}
                to={item.path} 
                className={`nav-link ${isActive(item.path) ? 'active' : ''}`}
              >
                <Icon size={18} />
                {item.label}
              </Link>
            );
          })}
        </div>
      </div>

      <div className="nav-user">
        <div className="user-info">
          <div className="user-avatar">
            <User size={20} />
          </div>
          <div className="user-details">
            <span className="user-name">{user?.full_name}</span>
            <span className="user-role">{user?.role}</span>
          </div>
        </div>
        <button onClick={logout} className="logout-btn">
          <LogOut size={18} />
        </button>
      </div>
    </nav>
  );
};

export default Navbar;