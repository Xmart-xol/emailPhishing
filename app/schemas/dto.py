"""
Pydantic schemas for API data transfer objects.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal, Union
from datetime import datetime


class FeatureConfigDTO(BaseModel):
    """Feature configuration schema."""
    bow: bool = True
    char_ngrams: bool = True
    heuristics: bool = True
    hash_dim: int = Field(default=20000, ge=1000, le=100000)
    ngram_min: int = Field(default=3, ge=1, le=10)
    ngram_max: int = Field(default=5, ge=1, le=10)
    use_tfidf: bool = True


class CrossValidationConfigDTO(BaseModel):
    """Cross-validation configuration schema."""
    type: Literal["kfold", "stratified"] = "stratified"
    k: int = Field(default=5, ge=2, le=10)
    shuffle: bool = True
    seed: int = 42


class KNNHyperparamsDTO(BaseModel):
    """KNN hyperparameters schema."""
    k: int = Field(ge=1, le=50)
    metric: Literal["cosine", "euclidean"] = "cosine"
    weighting: Literal["uniform", "distance"] = "uniform"


class SVMHyperparamsDTO(BaseModel):
    """SVM hyperparameters schema."""
    C: float = Field(default=1.0, gt=0)
    kernel: Literal["linear", "rbf"] = "linear"
    gamma: Optional[float] = Field(default=0.1, gt=0)
    tol: float = Field(default=1e-3, gt=0)
    max_passes: int = Field(default=5, ge=1, le=20)


class DataCleaningConfigDTO(BaseModel):
    """Data cleaning configuration schema."""
    remove_duplicates: bool = True
    min_text_length: int = Field(default=10, ge=1, le=1000)
    max_text_length: int = Field(default=10000, ge=100, le=100000)
    remove_html: bool = True
    normalize_whitespace: bool = True
    remove_empty: bool = True
    balance_classes: bool = False
    balance_method: Literal["undersample", "oversample"] = "undersample"
    outlier_detection: bool = True
    outlier_threshold: float = Field(default=3.0, ge=1.0, le=5.0)


class TransformationConfigDTO(BaseModel):
    """Data transformation configuration schema."""
    combine_text_fields: bool = True
    extract_metadata_features: bool = False
    normalize_labels: bool = True
    feature_scaling: bool = True
    handle_missing_values: bool = True
    missing_value_strategy: Literal["fill", "drop", "interpolate"] = "fill"


class TrainRequestDTO(BaseModel):
    """Training request schema with preprocessing support."""
    dataset_id: str
    features: FeatureConfigDTO
    model_type: Literal["knn", "svm"]
    hyperparams: Union[KNNHyperparamsDTO, SVMHyperparamsDTO]
    cv: CrossValidationConfigDTO
    # New preprocessing parameters
    dataset_format: Literal["simple", "rich"] = "simple"
    cleaning_config: Optional[DataCleaningConfigDTO] = None
    transform_config: Optional[TransformationConfigDTO] = None


class ClassifyRequestDTO(BaseModel):
    """Classification request schema."""
    # Support both 'text' and 'email_content' for compatibility
    text: Optional[str] = Field(None, max_length=50000)
    email_content: Optional[str] = Field(None, max_length=50000)
    subject: Optional[str] = Field(None, max_length=1000)
    content: Optional[str] = Field(None, max_length=50000)
    model: Literal["knn", "svm"] = "svm"  # Default to SVM
    run_id: Optional[str] = None
    
    def get_text_content(self) -> str:
        """Get the text content from various possible fields."""
        # Try different field names the frontend might send
        content = (
            self.email_content or 
            self.content or 
            self.text or 
            ""
        )
        
        # Combine with subject if available
        if self.subject:
            content = f"Subject: {self.subject}\n\n{content}"
            
        return content.strip()


class DatasetSummaryDTO(BaseModel):
    """Dataset summary schema."""
    id: str
    name: str
    created_at: datetime
    n_rows: int
    n_phish: int
    n_ham: int
    notes: Optional[str]
    
    class Config:
        from_attributes = True


class MetricsDTO(BaseModel):
    """Evaluation metrics schema."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float]
    confusion_matrix: Dict[str, Any]
    class_metrics: Dict[str, Dict[str, float]]


class RunDTO(BaseModel):
    """Training run schema."""
    id: str
    dataset_id: str
    model_type: str
    features: FeatureConfigDTO
    hyperparams: Union[KNNHyperparamsDTO, SVMHyperparamsDTO]
    metrics: Optional[MetricsDTO]
    created_at: datetime
    is_production: bool
    status: str
    training_time: Optional[float]
    total_classifications: Optional[int] = 0
    
    class Config:
        from_attributes = True


class ClassificationResultDTO(BaseModel):
    """Classification result schema."""
    label: Literal["phish", "ham"]
    score: float
    probability: float
    explanation: Dict[str, Any]


class ExplanationDTO(BaseModel):
    """Explanation schema for model predictions."""
    model_type: str
    top_features: List[Dict[str, Union[str, float]]]
    confidence: float
    decision_boundary_distance: Optional[float]
    neighbors: Optional[List[Dict[str, Any]]]  # For KNN
    support_vectors: Optional[Dict[str, Any]]  # For SVM


class DatasetUploadResponseDTO(BaseModel):
    """Dataset upload response schema."""
    dataset_id: str
    message: str
    summary: DatasetSummaryDTO


class TrainingProgressDTO(BaseModel):
    """Training progress schema."""
    run_id: str
    status: str
    progress: float
    current_step: str
    elapsed_time: float
    estimated_remaining: Optional[float]


class ErrorResponseDTO(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str]
    code: Optional[str]


class FeedbackRequest(BaseModel):
    """Feedback request schema for user reports."""
    classification_id: str
    feedback_type: Literal["false_positive", "false_negative", "correct"]
    reported_label: Optional[Literal["phish", "ham"]] = None
    confidence_level: Optional[int] = Field(None, ge=1, le=5)
    comments: Optional[str] = Field(None, max_length=1000)
    evidence_url: Optional[str] = None


class AnalyticsRequest(BaseModel):
    """Analytics request schema."""
    days: Optional[int] = Field(30, ge=1, le=365)
    include_trends: bool = True
    include_threat_intel: bool = True
    include_user_analytics: bool = True