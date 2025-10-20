import React, { useState, useEffect } from 'react';
import { Mail, Shield, AlertTriangle, CheckCircle, Copy, Trash2, Send, Loader2, Clock, Target } from 'lucide-react';
import axios from 'axios';
import { getApiUrl } from '../config/api';

interface ClassificationResult {
  label: 'phish' | 'ham';
  score: number;
  probability: number;
  explanation: {
    model_type: string;
    confidence: number;
    risk_level: string;
    processing_time_ms: number;
    features_detected?: {
      suspicious_patterns: string[];
      trust_indicators: string[];
      phishing_indicators_count: number;
      legitimate_indicators_count: number;
    };
    technical_analysis?: {
      domain_analysis: string;
      url_analysis: string;
      header_analysis: string;
    };
  };
}

interface ClassificationHistory {
  id: string;
  email_content: string;
  result: ClassificationResult;
  timestamp: string;
}

const EmailClassifier: React.FC = () => {
  const [emailContent, setEmailContent] = useState('');
  const [emailSubject, setEmailSubject] = useState('');
  const [selectedModel, setSelectedModel] = useState<'svm' | 'knn'>('svm');
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [history, setHistory] = useState<ClassificationHistory[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Sample emails for testing
  const sampleEmails = [
    {
      name: "Legitimate Newsletter",
      content: `Subject: Your Monthly Security Report

Dear Valued Customer,

Thank you for being a subscriber to our cybersecurity newsletter. This month's highlights include:

• New threat intelligence updates
• Best practices for password security
• Upcoming webinar on AI in cybersecurity

Best regards,
The Security Team
security@company.com

Unsubscribe: Click here to unsubscribe from future emails.`
    },
    {
      name: "Suspicious Phishing Email",
      content: `Subject: URGENT: Your account will be suspended

Dear Customer,

Your account has been flagged for suspicious activity. Click the link below immediately to verify your account or it will be permanently suspended within 24 hours.

VERIFY NOW: http://secure-account-verification-urgent.com/verify?id=123

This is a time-sensitive security measure. Do not ignore this message.

Customer Support Team
support@bank-security.net

Note: This is an automated message. Do not reply to this email.`
    },
    {
      name: "Corporate Email",
      content: `Subject: Q4 Budget Review Meeting

Hi Team,

Hope everyone is doing well. I wanted to schedule our Q4 budget review meeting for next Friday at 2 PM in the conference room.

Please review the budget documents I sent earlier and come prepared with your department's projections.

Let me know if you have any conflicts with the timing.

Best,
John Smith
Finance Director
john.smith@company.com`
    }
  ];

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const response = await axios.get(getApiUrl('/api/classify/history'));
      setHistory(response.data.slice(0, 5)); // Show last 5 classifications
    } catch (err) {
      console.error('Failed to load history:', err);
    }
  };

  const classifyEmail = async () => {
    if (!emailContent.trim()) {
      setError('Please enter email content to classify');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(getApiUrl('/api/classify'), {
        email_content: emailContent,
        subject: emailSubject,
        model: selectedModel
      });

      setResult(response.data);
      loadHistory(); // Refresh history
    } catch (err: any) {
      setError('Classification failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  const clearContent = () => {
    setEmailContent('');
    setResult(null);
    setError(null);
  };

  const loadSampleEmail = (sample: typeof sampleEmails[0]) => {
    setEmailContent(sample.content);
    setResult(null);
    setError(null);
  };

  const getRiskColor = (level: string) => {
    switch (level) {
      case 'LOW': return '#10b981';
      case 'MEDIUM': return '#f59e0b';
      case 'HIGH': return '#ef4444';
      case 'CRITICAL': return '#dc2626';
      default: return '#6b7280';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return '#dc2626';
    if (confidence >= 0.7) return '#f59e0b';
    if (confidence >= 0.5) return '#10b981';
    return '#6b7280';
  };

  return (
    <div className="email-classifier">
      <div className="classifier-header">
        <Mail size={48} className="page-icon" />
        <div>
          <h1>Email Security Analysis</h1>
          <p>Advanced AI-powered phishing detection and email security analysis</p>
        </div>
      </div>

      <div className="classifier-container">
        <div className="classifier-main">
          <div className="input-section">
            <div className="section-header">
              <h3>Email Content</h3>
              <div className="input-actions">
                <button 
                  onClick={clearContent} 
                  className="btn btn-secondary btn-small"
                  disabled={loading}
                >
                  <Trash2 size={16} />
                  Clear
                </button>
                <button 
                  onClick={() => copyToClipboard(emailContent)} 
                  className="btn btn-secondary btn-small"
                  disabled={!emailContent}
                >
                  <Copy size={16} />
                  Copy
                </button>
              </div>
            </div>

            <textarea
              value={emailContent}
              onChange={(e) => setEmailContent(e.target.value)}
              placeholder="Paste your email content here for analysis..."
              className="email-input"
              rows={12}
              disabled={loading}
            />

            <div className="sample-emails">
              <h4>Try Sample Emails:</h4>
              <div className="sample-buttons">
                {sampleEmails.map((sample, index) => (
                  <button
                    key={index}
                    onClick={() => loadSampleEmail(sample)}
                    className="sample-button"
                    disabled={loading}
                  >
                    {sample.name}
                  </button>
                ))}
              </div>
            </div>

            <div className="model-selection">
              <h4>Classification Model:</h4>
              <div className="model-options">
                <label className="model-option">
                  <input
                    type="radio"
                    value="svm"
                    checked={selectedModel === 'svm'}
                    onChange={(e) => setSelectedModel(e.target.value as 'svm' | 'knn')}
                    disabled={loading}
                  />
                  <span>SVM (Support Vector Machine)</span>
                  <small>High accuracy, good for complex patterns</small>
                </label>
                <label className="model-option">
                  <input
                    type="radio"
                    value="knn"
                    checked={selectedModel === 'knn'}
                    onChange={(e) => setSelectedModel(e.target.value as 'svm' | 'knn')}
                    disabled={loading}
                  />
                  <span>KNN (K-Nearest Neighbors)</span>
                  <small>Fast classification, similarity-based</small>
                </label>
              </div>
            </div>

            <button 
              onClick={classifyEmail}
              disabled={loading || !emailContent.trim()}
              className="classify-button"
            >
              {loading ? (
                <>
                  <Loader2 className="spinner" size={20} />
                  Analyzing Email...
                </>
              ) : (
                <>
                  <Send size={20} />
                  Analyze Email Security
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="alert alert-error">
              <AlertTriangle size={20} />
              {error}
            </div>
          )}

          {result && (
            <div className="results-section">
              <div className="result-header">
                <div className={`threat-indicator ${result.label === 'phish' ? 'phishing' : 'legitimate'}`}>
                  {result.label === 'phish' ? (
                    <>
                      <AlertTriangle size={24} />
                      <span>PHISHING DETECTED</span>
                    </>
                  ) : (
                    <>
                      <CheckCircle size={24} />
                      <span>LEGITIMATE EMAIL</span>
                    </>
                  )}
                </div>
                
                <div className="result-metrics">
                  <div className="metric">
                    <Target size={18} />
                    <span>Confidence: {(result.explanation.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric">
                    <Shield size={18} />
                    <span style={{ color: getRiskColor(result.explanation.risk_level) }}>
                      Risk: {result.explanation.risk_level}
                    </span>
                  </div>
                  <div className="metric">
                    <Clock size={18} />
                    <span>{result.explanation.processing_time_ms}ms</span>
                  </div>
                </div>
              </div>

              <div className="analysis-details">
                {result.explanation.features_detected?.suspicious_patterns && result.explanation.features_detected.suspicious_patterns.length > 0 && (
                  <div className="analysis-section suspicious">
                    <h4>🚨 Suspicious Patterns Detected</h4>
                    <ul>
                      {result.explanation.features_detected.suspicious_patterns.map((pattern, index) => (
                        <li key={index}>{pattern}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {result.explanation.features_detected?.trust_indicators && result.explanation.features_detected.trust_indicators.length > 0 && (
                  <div className="analysis-section trusted">
                    <h4>✅ Trust Indicators</h4>
                    <ul>
                      {result.explanation.features_detected.trust_indicators.map((indicator, index) => (
                        <li key={index}>{indicator}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="technical-analysis">
                  <h4>🔍 Technical Analysis</h4>
                  <div className="analysis-grid">
                    <div className="analysis-item">
                      <h5>Domain Analysis</h5>
                      <p>{result.explanation.technical_analysis?.domain_analysis}</p>
                    </div>
                    <div className="analysis-item">
                      <h5>URL Analysis</h5>
                      <p>{result.explanation.technical_analysis?.url_analysis}</p>
                    </div>
                    <div className="analysis-item">
                      <h5>Header Analysis</h5>
                      <p>{result.explanation.technical_analysis?.header_analysis}</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="confidence-bar">
                <div className="confidence-label">
                  Confidence Score: {(result.explanation.confidence * 100).toFixed(1)}%
                </div>
                <div className="confidence-track">
                  <div 
                    className="confidence-fill"
                    style={{ 
                      width: `${result.explanation.confidence * 100}%`,
                      backgroundColor: getConfidenceColor(result.explanation.confidence)
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="classifier-sidebar">
          <div className="history-section">
            <h3>Recent Classifications</h3>
            {history.length > 0 ? (
              <div className="history-list">
                {history.map((item) => (
                  <div key={item.id} className="history-item">
                    <div className="history-header">
                      <span className={`history-badge ${item.result.label === 'phish' ? 'phishing' : 'legitimate'}`}>
                        {item.result.label === 'phish' ? 'Phishing' : 'Legitimate'}
                      </span>
                      <span className="history-time">
                        {new Date(item.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <p className="history-content">
                      {item.email_content.substring(0, 100)}...
                    </p>
                    <div className="history-confidence">
                      Confidence: {(item.result.explanation.confidence * 100).toFixed(1)}%
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="no-history">No recent classifications</p>
            )}
          </div>

          <div className="info-section">
            <h3>Security Tips</h3>
            <div className="tips-list">
              <div className="tip">
                <AlertTriangle size={16} />
                <p>Always verify sender identity through separate channels</p>
              </div>
              <div className="tip">
                <Shield size={16} />
                <p>Hover over links to preview destinations before clicking</p>
              </div>
              <div className="tip">
                <CheckCircle size={16} />
                <p>Be suspicious of urgent requests for personal information</p>
              </div>
              <div className="tip">
                <Mail size={16} />
                <p>Check email headers for authentication records</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmailClassifier;