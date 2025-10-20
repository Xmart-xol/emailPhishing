# Email Phishing Detection System

A comprehensive machine learning-based email phishing detection system with real-time classification capabilities, built with FastAPI backend and React frontend.

## 🏗️ Project Structure

```
emailPhishing/
├── app/                        # Backend (FastAPI)
│   ├── main.py                # Main FastAPI application
│   ├── api/                   # API endpoints
│   │   ├── admin.py          # Admin endpoints (datasets, training)
│   │   ├── analytics.py      # Analytics and reporting
│   │   └── classify.py       # Email classification endpoint
│   ├── ml/                    # Machine learning components
│   │   ├── features.py       # Feature extraction
│   │   ├── knn.py           # K-Nearest Neighbors implementation
│   │   ├── svm.py           # Support Vector Machine implementation
│   │   └── metrics.py       # Evaluation metrics
│   ├── schemas/              # Data models and DTOs
│   │   ├── models.py        # Database models
│   │   └── dto.py           # Data transfer objects
│   └── services/             # Business logic services
│       ├── storage.py       # Database operations
│       ├── training.py      # Model training service
│       ├── analytics.py     # Analytics service
│       └── auth.py          # Authentication service
├── phishing-ui/               # Frontend (React + TypeScript)
│   ├── package.json          # Frontend dependencies
│   ├── src/                  # React source code
│   └── public/               # Public assets
├── data/                      # Data storage
│   ├── datasets/             # Uploaded datasets
│   ├── models/               # Trained model artifacts
│   └── artifacts/            # Training artifacts
├── tests/                     # Test suites
├── docs/                      # Documentation
├── requirements.txt           # Python dependencies
├── start_server.py           # Server startup script
└── README.md                 # This file
```

## 🚀 Features

- **Real-time Email Classification**: Classify emails as phishing or legitimate using trained ML models
- **Multiple ML Algorithms**: Support for KNN and SVM classifiers
- **Advanced Feature Engineering**: Bag of Words, Character N-grams, and Heuristic features
- **Model Management**: Train, evaluate, and deploy models through web interface
- **Analytics Dashboard**: Comprehensive analytics and reporting capabilities
- **Dataset Management**: Upload and manage training datasets
- **Production Deployment**: Ready for production with proper model versioning

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: SQL toolkit and ORM
- **scikit-learn**: Machine learning library
- **SQLite**: Lightweight database for development
- **Pydantic**: Data validation and settings management

### Frontend
- **React 19**: Modern React with latest features
- **TypeScript**: Type-safe JavaScript
- **Axios**: HTTP client for API requests
- **Recharts**: Data visualization and charts
- **Lucide React**: Beautiful icons

### Machine Learning
- **Custom KNN Implementation**: K-Nearest Neighbors from scratch
- **Custom SVM Implementation**: Support Vector Machine from scratch
- **Feature Engineering**: Advanced text processing and feature extraction
- **Cross-validation**: Robust model evaluation

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- npm or yarn

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
python start_server.py
```

The backend will be available at `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd phishing-ui

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will be available at `http://localhost:3000`

## 📊 Model Performance

Current production models achieve the following performance:

- **SVM Model**: 97.3% accuracy, 97.1% F1-score
- **KNN Model**: 92.9% accuracy, 91.8% F1-score

## 🔧 API Endpoints

### Classification
- `POST /api/classify` - Classify email content
- `GET /api/classify/history` - Get classification history

### Admin
- `POST /api/admin/datasets/upload` - Upload training dataset
- `GET /api/admin/datasets` - List all datasets
- `POST /api/admin/train` - Start model training
- `GET /api/admin/runs` - Get training runs
- `POST /api/admin/runs/{run_id}/promote` - Promote model to production

### Analytics
- `GET /api/analytics/dashboard/stats` - Dashboard statistics
- `GET /api/analytics/trends` - Analytics trends

## 🗄️ Database Schema

The system uses SQLite with the following main tables:
- `datasets`: Uploaded training datasets
- `runs`: Model training runs and metadata
- `artifacts`: Model artifacts and files
- `classifications`: Email classification history

## 🧪 Testing

```bash
# Run backend tests
python -m pytest tests/

# Test trained models
python test_trained_model.py

# System integration test
python test_system.py
```

## 📈 Model Training

1. Upload a dataset through the web interface
2. Configure feature extraction and hyperparameters
3. Start training process
4. Monitor training progress
5. Evaluate model performance
6. Promote to production if satisfactory

## 🔒 Security Features

- Rate limiting on classification endpoints
- Input validation and sanitization
- SQL injection prevention
- CORS configuration for frontend

## 🚀 Deployment

The system is designed for easy deployment with:
- Docker containerization support
- Environment-based configuration
- Production-ready database migrations
- Static file serving for frontend

## 📖 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive API docs
- [Deployment Guide](deployment_deliverable.md) - Deployment instructions
- [Video Presentation](video_presentation.md) - System overview

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is developed for educational and research purposes.

## 📞 Support

For questions or issues, please refer to the documentation or create an issue in the repository.