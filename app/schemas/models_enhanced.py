"""
Database models and schemas for the email phishing detection system.
Enhanced with user management, classification logging, and analytics.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum

Base = declarative_base()


# Enums for better type safety
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    ANALYST = "analyst"


class ClassificationStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"


class FeedbackType(str, Enum):
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    CORRECT_CLASSIFICATION = "correct"
    GENERAL_FEEDBACK = "general"


# User Management Tables
class User(Base):
    """User table for authentication and authorization."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(String, nullable=False, default=UserRole.USER)  # admin, user, analyst
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    login_count = Column(Integer, default=0)
    
    # Relationships
    classifications = relationship("Classification", back_populates="user")
    feedback_reports = relationship("FeedbackReport", back_populates="user", foreign_keys="FeedbackReport.user_id")
    user_sessions = relationship("UserSession", back_populates="user")

    __table_args__ = (
        Index('idx_users_role_active', 'role', 'is_active'),
    )


class UserSession(Base):
    """Track user sessions for analytics."""
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_token = Column(String, unique=True, nullable=False)
    ip_address = Column(String)
    user_agent = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="user_sessions")


# Classification and Analytics Tables
class Classification(Base):
    """Log all email classifications for analytics and audit."""
    __tablename__ = "classifications"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Nullable for anonymous users
    run_id = Column(String, ForeignKey("runs.id"), nullable=False)  # Which model was used
    
    # Email content (hashed or anonymized for privacy)
    email_hash = Column(String, nullable=False, index=True)  # SHA256 hash of email content
    email_subject_hash = Column(String, index=True)  # Hash of subject for analysis
    sender_domain = Column(String, index=True)  # Extract domain for stats
    
    # Classification results
    predicted_label = Column(String, nullable=False)  # "phish" or "ham"
    confidence_score = Column(Float, nullable=False)
    model_type = Column(String, nullable=False)  # "knn", "svm"
    processing_time_ms = Column(Float)  # Response time in milliseconds
    
    # Feedback and verification
    user_feedback = Column(String)  # User's reported actual label
    status = Column(String, default=ClassificationStatus.PENDING)
    is_disputed = Column(Boolean, default=False)
    
    # Metadata
    ip_address = Column(String)
    user_agent = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Feature analysis (JSON field for storing top features)
    top_features = Column(JSON)  # Store top contributing features
    
    # Relationships
    user = relationship("User", back_populates="classifications")
    run = relationship("Run", back_populates="classifications")
    feedback_reports = relationship("FeedbackReport", back_populates="classification")

    __table_args__ = (
        Index('idx_classifications_created_at', 'created_at'),
        Index('idx_classifications_label_confidence', 'predicted_label', 'confidence_score'),
        Index('idx_classifications_status', 'status'),
        Index('idx_classifications_user_created', 'user_id', 'created_at'),
    )


class FeedbackReport(Base):
    """User feedback and false positive/negative reports."""
    __tablename__ = "feedback_reports"
    
    id = Column(String, primary_key=True)
    classification_id = Column(String, ForeignKey("classifications.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    feedback_type = Column(String, nullable=False)  # false_positive, false_negative, correct, general
    reported_label = Column(String)  # What the user says the correct label should be
    confidence_level = Column(Integer)  # User's confidence 1-5
    
    # Detailed feedback
    comments = Column(Text)
    evidence_url = Column(String)  # Link to external evidence if provided
    
    # Admin review
    admin_reviewed = Column(Boolean, default=False)
    admin_notes = Column(Text)
    admin_decision = Column(String)  # accepted, rejected, pending
    reviewed_by = Column(String, ForeignKey("users.id"))
    reviewed_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    classification = relationship("Classification", back_populates="feedback_reports")
    user = relationship("User", back_populates="feedback_reports", foreign_keys=[user_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])

    __table_args__ = (
        Index('idx_feedback_type_created', 'feedback_type', 'created_at'),
        Index('idx_feedback_admin_review', 'admin_reviewed', 'created_at'),
    )


class SystemMetrics(Base):
    """Store system performance metrics and analytics."""
    __tablename__ = "system_metrics"
    
    id = Column(String, primary_key=True)
    metric_type = Column(String, nullable=False, index=True)  # daily_stats, model_performance, etc.
    metric_name = Column(String, nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    metric_data = Column(JSON)  # Additional structured data
    
    # Time-based partitioning
    date_recorded = Column(DateTime, default=datetime.utcnow, index=True)
    hour_bucket = Column(Integer, index=True)  # 0-23 for hourly aggregation
    day_bucket = Column(String, index=True)   # YYYY-MM-DD for daily aggregation
    
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_metrics_type_date', 'metric_type', 'date_recorded'),
        Index('idx_metrics_daily', 'day_bucket', 'metric_name'),
    )


class AuditLog(Base):
    """Audit trail for security and compliance."""
    __tablename__ = "audit_logs"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    action = Column(String, nullable=False, index=True)  # login, classify, upload, train, etc.
    resource_type = Column(String, index=True)  # user, dataset, model, etc.
    resource_id = Column(String, index=True)
    
    # Action details
    details = Column(JSON)  # Structured details about the action
    ip_address = Column(String)
    user_agent = Column(String)
    success = Column(Boolean, default=True)
    error_message = Column(Text)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User")

    __table_args__ = (
        Index('idx_audit_user_action', 'user_id', 'action'),
        Index('idx_audit_created_at', 'created_at'),
    )


# Enhanced existing tables
class Dataset(Base):
    """Dataset table for storing uploaded datasets."""
    __tablename__ = "datasets"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, ForeignKey("users.id"))  # Track who uploaded
    n_rows = Column(Integer, nullable=False)
    n_phish = Column(Integer, nullable=False)
    n_ham = Column(Integer, nullable=False)
    notes = Column(Text)
    file_path = Column(String)
    
    # Dataset metadata for analytics
    size_bytes = Column(Integer)
    checksum = Column(String)  # File integrity
    is_active = Column(Boolean, default=True)
    
    # Relationships
    runs = relationship("Run", back_populates="dataset")
    creator = relationship("User")


class Run(Base):
    """Training run table for storing model training metadata."""
    __tablename__ = "runs"
    
    id = Column(String, primary_key=True)
    dataset_id = Column(String, ForeignKey("datasets.id"), nullable=False)
    created_by = Column(String, ForeignKey("users.id"))  # Track who started training
    model_type = Column(String, nullable=False)  # "knn" or "svm"
    features_json = Column(Text, nullable=False)  # JSON string of feature config
    hyperparams_json = Column(Text, nullable=False)  # JSON string of hyperparameters
    metrics_json = Column(Text)  # JSON string of evaluation metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    is_production = Column(Boolean, default=False)
    status = Column(String, default="pending")  # "pending", "training", "completed", "failed"
    training_time = Column(Float)  # Training time in seconds
    
    # Performance tracking
    accuracy_mean = Column(Float)  # Quick access to accuracy
    f1_score_mean = Column(Float)  # Quick access to F1 score
    total_classifications = Column(Integer, default=0)  # Usage counter
    
    # Relationships
    dataset = relationship("Dataset", back_populates="runs")
    artifacts = relationship("Artifact", back_populates="run")
    classifications = relationship("Classification", back_populates="run")
    creator = relationship("User")


class Artifact(Base):
    """Artifact table for storing model artifacts."""
    __tablename__ = "artifacts"
    
    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("runs.id"), nullable=False)
    path = Column(String, nullable=False)
    kind = Column(String, nullable=False)  # "vectorizer", "svm_params", "knn_store", "scaler", "idf"
    created_at = Column(DateTime, default=datetime.utcnow)
    size_bytes = Column(Integer)  # File size tracking
    checksum = Column(String)  # File integrity
    
    # Relationships
    run = relationship("Run", back_populates="artifacts")