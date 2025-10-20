"""
Database models and schemas for the email phishing detection system.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Dataset(Base):
    """Dataset table for storing uploaded datasets."""
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    n_rows = Column(Integer, nullable=False)
    n_phish = Column(Integer, nullable=False)
    n_ham = Column(Integer, nullable=False)
    notes = Column(Text)
    file_path = Column(String)
    
    # Relationships
    runs = relationship("Run", back_populates="dataset")


class Run(Base):
    """Training run table for storing model training metadata."""
    __tablename__ = "runs"
    
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    model_type = Column(String, nullable=False)  # "knn" or "svm"
    features_json = Column(Text, nullable=False)  # JSON string of feature config
    hyperparams_json = Column(Text, nullable=False)  # JSON string of hyperparameters
    metrics_json = Column(Text)  # JSON string of evaluation metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    is_production = Column(Boolean, default=False)
    status = Column(String, default="pending")  # "pending", "training", "completed", "failed"
    training_time = Column(Float)  # Training time in seconds
    total_classifications = Column(Integer, default=0)  # Number of classifications performed
    
    # Add dedicated columns for key metrics
    accuracy_mean = Column(Float)  # Mean accuracy from cross-validation or single run
    f1_score_mean = Column(Float)  # Mean F1-score from cross-validation or single run
    
    # Relationships
    dataset = relationship("Dataset", back_populates="runs")
    artifacts = relationship("Artifact", back_populates="run")


class Artifact(Base):
    """Artifact table for storing model artifacts."""
    __tablename__ = "artifacts"
    
    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("runs.id"), nullable=False)
    path = Column(String, nullable=False)
    kind = Column(String, nullable=False)  # "vectorizer", "svm_params", "knn_store", "scaler", "idf"
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    run = relationship("Run", back_populates="artifacts")