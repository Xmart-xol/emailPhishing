import React, { createContext, useContext, useState, useEffect } from 'react';

export interface User {
  id: string;
  username: string;
  email: string;
  full_name: string;
  role: 'admin' | 'researcher' | 'user';
  is_active: boolean;
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  hasRole: (role: string) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Check for stored authentication on app load
  useEffect(() => {
    const storedUser = localStorage.getItem('phishing_user');
    if (storedUser) {
      try {
        const userData = JSON.parse(storedUser);
        setUser(userData);
      } catch (error) {
        localStorage.removeItem('phishing_user');
      }
    }
    setIsLoading(false);
  }, []);

  const login = async (username: string, password: string): Promise<boolean> => {
    try {
      setError(null);
      setIsLoading(true);
      
      // Add a small delay to ensure UI updates
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // For demo purposes, we'll simulate different user types
      // In production, this would make an API call to your backend
      const mockUsers: User[] = [
        {
          id: 'admin-1',
          username: 'admin',
          email: 'admin@phishdetect.com',
          full_name: 'System Administrator',
          role: 'admin',
          is_active: true
        },
        {
          id: 'user-1',
          username: 'user',
          email: 'user@company.com',
          full_name: 'John Doe',
          role: 'user',
          is_active: true
        }
      ];

      const foundUser = mockUsers.find(u => 
        u.username === username && password === 'demo123'
      );

      if (foundUser) {
        setUser(foundUser);
        localStorage.setItem('phishing_user', JSON.stringify(foundUser));
        return true;
      }
      
      setError('Invalid username or password');
      return false;
    } catch (error) {
      console.error('Login error:', error);
      setError('Login failed. Please try again.');
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    setError(null);
    localStorage.removeItem('phishing_user');
  };

  const hasRole = (role: string): boolean => {
    return user?.role === role;
  };

  const value = {
    user,
    isAuthenticated: !!user,
    isLoading,
    error,
    login,
    logout,
    hasRole
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};