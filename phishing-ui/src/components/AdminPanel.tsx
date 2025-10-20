import React, { useState, useEffect } from 'react';
import { Upload, Database, Brain, BarChart3, Settings, Loader2, CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react';
import axios from 'axios';
import { getApiUrl } from '../config/api';

interface Dataset {
  id: string;
  name: string;
  n_rows: number;
  n_phish: number;
  n_ham: number;
  created_at: string;
}

interface TrainingRun {
  id: string;
  model_type: string;
  status: string;
  metrics?: {
    accuracy_mean: number;
    f1_mean: number;
  };
  training_time?: number;
  created_at: string;
  is_production: boolean;
}

interface TrainingConfig {
  model_type: string;
  features: {
    bow: boolean;
    char_ngrams: boolean;
    heuristics: boolean;
    hash_dim: number;
    ngram_min: number;
    ngram_max: number;
    use_tfidf: boolean;
  };
  hyperparams: any;
  cv: {
    type: string;
    k: number;
    shuffle: boolean;
    seed: number;
  };
}

const AdminPanel: React.FC = () => {
  const [activeTab, setActiveTab] = useState('datasets');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [runs, setRuns] = useState<TrainingRun[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Dataset upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadName, setUploadName] = useState('');
  const [uploading, setUploading] = useState(false);

  // Training state
  const [selectedDataset, setSelectedDataset] = useState('');
  const [training, setTraining] = useState(false);
  const [trainingConfig, setTrainingConfig] = useState<TrainingConfig>({
    model_type: 'svm',
    features: {
      bow: true,
      char_ngrams: true,
      heuristics: true,
      hash_dim: 10000,
      ngram_min: 3,
      ngram_max: 5,
      use_tfidf: true
    },
    hyperparams: {
      C: 1.0,
      kernel: 'linear',
      tol: 0.001,
      max_passes: 5
    },
    cv: {
      type: 'stratified',
      k: 5,
      shuffle: true,
      seed: 42
    }
  });

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [datasetsResponse, runsResponse] = await Promise.all([
        axios.get(getApiUrl('/api/admin/datasets')),
        axios.get(getApiUrl('/api/admin/runs'))
      ]);

      setDatasets(datasetsResponse.data.datasets || []);
      setRuns(runsResponse.data.runs || []);
      setError(null);
    } catch (err) {
      setError('Failed to load admin data');
      console.error('Admin error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!uploadFile) {
      setError('Please select a file');
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);

    const formData = new FormData();
    formData.append('file', uploadFile);
    if (uploadName) formData.append('name', uploadName);

    try {
      const response = await axios.post(getApiUrl('/api/admin/datasets'), formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setSuccess('Dataset uploaded successfully!');
      setDatasets([response.data.summary, ...datasets]);
      
      // Clear form
      setUploadFile(null);
      setUploadName('');
      
    } catch (err: any) {
      setError('Upload failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setUploading(false);
    }
  };

  const handleTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedDataset) {
      setError('Please select a dataset');
      return;
    }

    setTraining(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await axios.post(getApiUrl('/api/admin/train'), {
        dataset_id: selectedDataset,
        ...trainingConfig
      });

      setSuccess(`Training started! Run ID: ${response.data.run_id}`);
      
      // Refresh runs to show new training
      setTimeout(() => fetchData(), 1000);
      
    } catch (err: any) {
      setError('Training failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setTraining(false);
    }
  };

  const updateHyperparams = (key: string, value: any) => {
    setTrainingConfig(prev => ({
      ...prev,
      hyperparams: {
        ...prev.hyperparams,
        [key]: value
      }
    }));
  };

  const updateFeatures = (key: string, value: any) => {
    setTrainingConfig(prev => ({
      ...prev,
      features: {
        ...prev.features,
        [key]: value
      }
    }));
  };

  const updateCV = (key: string, value: any) => {
    setTrainingConfig(prev => ({
      ...prev,
      cv: {
        ...prev.cv,
        [key]: value
      }
    }));
  };

  const renderDatasetsTab = () => (
    <div className="tab-content">
      <div className="section-header">
        <Database size={24} />
        <h3>Dataset Management</h3>
      </div>
      
      <div className="upload-section">
        <h4>Upload New Dataset</h4>
        <form onSubmit={handleFileUpload} className="upload-form">
          <div className="form-group">
            <label>CSV File</label>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
              className="file-input"
              required
            />
          </div>
          
          <div className="form-group">
            <label>Dataset Name (optional)</label>
            <input
              type="text"
              value={uploadName}
              onChange={(e) => setUploadName(e.target.value)}
              placeholder="e.g., PhishTank Dataset 2024"
              className="form-input"
            />
          </div>

          <button type="submit" disabled={uploading || !uploadFile} className="btn btn-primary">
            {uploading ? (
              <>
                <Loader2 className="spinner" size={18} />
                Uploading...
              </>
            ) : (
              <>
                <Upload size={18} />
                Upload Dataset
              </>
            )}
          </button>
        </form>
      </div>

      <div className="datasets-grid">
        {datasets.map((dataset) => (
          <div key={dataset.id} className="dataset-card">
            <h5>{dataset.name}</h5>
            <div className="dataset-stats">
              <div className="stat">
                <div className="stat-value">{dataset.n_rows}</div>
                <div className="stat-label">Total</div>
              </div>
              <div className="stat">
                <div className="stat-value phishing">{dataset.n_phish}</div>
                <div className="stat-label">Phishing</div>
              </div>
              <div className="stat">
                <div className="stat-value legitimate">{dataset.n_ham}</div>
                <div className="stat-label">Legitimate</div>
              </div>
            </div>
            <div className="dataset-date">
              Created: {new Date(dataset.created_at).toLocaleDateString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderTrainingTab = () => (
    <div className="tab-content">
      <div className="section-header">
        <Brain size={24} />
        <h3>Model Training</h3>
      </div>
      
      <form onSubmit={handleTraining} className="training-form">
        <div className="form-group">
          <label>Select Dataset</label>
          <select
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
            className="form-select"
            required
          >
            <option value="">Choose a dataset...</option>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.name} ({dataset.n_rows} samples)
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Model Type</label>
          <select
            value={trainingConfig.model_type}
            onChange={(e) => {
              const modelType = e.target.value;
              setTrainingConfig(prev => ({
                ...prev,
                model_type: modelType,
                hyperparams: modelType === 'svm' ? {
                  C: 1.0,
                  kernel: 'linear',
                  tol: 0.001,
                  max_passes: 5
                } : {
                  k: 5,
                  metric: 'cosine',
                  weighting: 'uniform'
                }
              }));
            }}
            className="form-select"
          >
            <option value="svm">SVM (Support Vector Machine)</option>
            <option value="knn">KNN (K-Nearest Neighbors)</option>
          </select>
        </div>

        <div className="form-group">
          <label>Feature Configuration</label>
          <div className="checkbox-grid">
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={trainingConfig.features.bow}
                onChange={(e) => updateFeatures('bow', e.target.checked)}
              />
              Bag of Words
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={trainingConfig.features.char_ngrams}
                onChange={(e) => updateFeatures('char_ngrams', e.target.checked)}
              />
              Character N-grams
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={trainingConfig.features.heuristics}
                onChange={(e) => updateFeatures('heuristics', e.target.checked)}
              />
              Heuristic Features
            </label>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={trainingConfig.features.use_tfidf}
                onChange={(e) => updateFeatures('use_tfidf', e.target.checked)}
              />
              TF-IDF Weighting
            </label>
          </div>
        </div>

        <div className="form-group">
          <label>Hyperparameters</label>
          {trainingConfig.model_type === 'svm' ? (
            <div className="form-row">
              <div>
                <label>C (Regularization)</label>
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  value={trainingConfig.hyperparams.C}
                  onChange={(e) => updateHyperparams('C', parseFloat(e.target.value))}
                  className="form-input"
                />
              </div>
              <div>
                <label>Kernel</label>
                <select
                  value={trainingConfig.hyperparams.kernel}
                  onChange={(e) => updateHyperparams('kernel', e.target.value)}
                  className="form-select"
                >
                  <option value="linear">Linear</option>
                  <option value="rbf">RBF</option>
                </select>
              </div>
            </div>
          ) : (
            <div className="form-row">
              <div>
                <label>K (Neighbors)</label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={trainingConfig.hyperparams.k}
                  onChange={(e) => updateHyperparams('k', parseInt(e.target.value))}
                  className="form-input"
                />
              </div>
              <div>
                <label>Distance Metric</label>
                <select
                  value={trainingConfig.hyperparams.metric}
                  onChange={(e) => updateHyperparams('metric', e.target.value)}
                  className="form-select"
                >
                  <option value="cosine">Cosine</option>
                  <option value="euclidean">Euclidean</option>
                </select>
              </div>
            </div>
          )}
        </div>

        <div className="form-group">
          <label>Cross-Validation</label>
          <div className="form-row">
            <div>
              <label>CV Type</label>
              <select
                value={trainingConfig.cv.type}
                onChange={(e) => updateCV('type', e.target.value)}
                className="form-select"
              >
                <option value="stratified">Stratified K-Fold</option>
                <option value="kfold">K-Fold</option>
              </select>
            </div>
            <div>
              <label>Number of Folds</label>
              <input
                type="number"
                min="2"
                max="10"
                value={trainingConfig.cv.k}
                onChange={(e) => updateCV('k', parseInt(e.target.value))}
                className="form-input"
              />
            </div>
          </div>
        </div>

        <button type="submit" disabled={training || !selectedDataset} className="btn btn-primary btn-large">
          {training ? (
            <>
              <Loader2 className="spinner" size={20} />
              Training Model...
            </>
          ) : (
            <>
              <Brain size={20} />
              Start Training
            </>
          )}
        </button>
      </form>
    </div>
  );

  const renderRunsTab = () => (
    <div className="tab-content">
      <div className="runs-header">
        <div className="section-header">
          <BarChart3 size={24} />
          <h3>Training Runs</h3>
        </div>
        <button onClick={fetchData} className="btn btn-secondary">
          <RefreshCw size={18} />
          Refresh
        </button>
      </div>

      <div className="runs-list">
        {runs.map((run) => (
          <div key={run.id} className="run-card">
            <div className="run-header">
              <div>
                <strong>{run.model_type.toUpperCase()} Model</strong>
                {run.is_production && <span className="production-badge">⭐ Production</span>}
                <div className="run-id">ID: {run.id}</div>
              </div>
              <span className={`status-badge ${run.status}`}>
                {run.status}
              </span>
            </div>

            {run.metrics && (
              <div className="metrics-grid">
                <div className="metric">
                  <div className="metric-label">Accuracy</div>
                  <div className="metric-value">{(run.metrics.accuracy_mean * 100).toFixed(1)}%</div>
                </div>
                <div className="metric">
                  <div className="metric-label">F1-Score</div>
                  <div className="metric-value">{(run.metrics.f1_mean * 100).toFixed(1)}%</div>
                </div>
                <div className="metric">
                  <div className="metric-label">Training Time</div>
                  <div className="metric-value">{run.training_time?.toFixed(2)}s</div>
                </div>
              </div>
            )}

            <div className="run-date">
              Created: {new Date(run.created_at).toLocaleString()}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div className="admin-panel">
      <div className="page-header">
        <Settings size={48} className="page-icon" />
        <h2>Admin Panel</h2>
        <p>Manage datasets, train models, and monitor system performance</p>
      </div>

      {error && (
        <div className="alert alert-error">
          <AlertTriangle size={20} />
          {error}
        </div>
      )}

      {success && (
        <div className="alert alert-success">
          <CheckCircle size={20} />
          {success}
        </div>
      )}

      <div className="admin-tabs">
        <button
          className={`tab-button ${activeTab === 'datasets' ? 'active' : ''}`}
          onClick={() => setActiveTab('datasets')}
        >
          <Database size={18} />
          Datasets
        </button>
        <button
          className={`tab-button ${activeTab === 'training' ? 'active' : ''}`}
          onClick={() => setActiveTab('training')}
        >
          <Brain size={18} />
          Training
        </button>
        <button
          className={`tab-button ${activeTab === 'runs' ? 'active' : ''}`}
          onClick={() => setActiveTab('runs')}
        >
          <BarChart3 size={18} />
          Runs
        </button>
      </div>

      <div className="admin-content">
        {loading && (
          <div className="loading-state">
            <Loader2 className="spinner" size={32} />
            Loading...
          </div>
        )}

        {activeTab === 'datasets' && renderDatasetsTab()}
        {activeTab === 'training' && renderTrainingTab()}
        {activeTab === 'runs' && renderRunsTab()}
      </div>
    </div>
  );
};

export default AdminPanel;