"""
Database and storage service for managing datasets and model artifacts.
"""
import os
import json
import sqlite3
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import uuid
from datetime import datetime
import pandas as pd

from ..schemas.models import Base, Dataset, Run, Artifact
from ..schemas.dto import DatasetSummaryDTO, RunDTO


class DatabaseService:
    """Service for database operations."""
    
    def __init__(self, database_url: str = "sqlite:///./phishing_detector.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.engine)
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.SessionLocal = SessionLocal

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()
    
    def create_dataset(
        self, 
        name: str, 
        n_rows: int, 
        n_phish: int, 
        n_ham: int,
        file_path: str = None,
        notes: str = None
    ) -> str:
        """Create a new dataset record."""
        dataset_id = str(uuid.uuid4())
        
        dataset = Dataset(
            id=dataset_id,
            name=name,
            n_rows=n_rows,
            n_phish=n_phish,
            n_ham=n_ham,
            file_path=file_path,
            notes=notes
        )
        
        with self.get_session() as session:
            session.add(dataset)
            session.commit()
        
        return dataset_id
    
    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        """Get a dataset by ID."""
        with self.get_session() as session:
            return session.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    def get_all_datasets(self) -> List[Dataset]:
        """Get all datasets."""
        with self.get_session() as session:
            return session.query(Dataset).order_by(Dataset.created_at.desc()).all()
    
    def create_run(
        self,
        dataset_id: str,
        model_type: str,
        features_config: Dict[str, Any],
        hyperparams: Dict[str, Any]
    ) -> str:
        """Create a new training run record."""
        run_id = str(uuid.uuid4())
        
        run = Run(
            id=run_id,
            dataset_id=dataset_id,
            model_type=model_type,
            features_json=json.dumps(features_config),
            hyperparams_json=json.dumps(hyperparams),
            status="pending"
        )
        
        with self.get_session() as session:
            session.add(run)
            session.commit()
        
        return run_id
    
    def update_run_status(self, run_id: str, status: str, training_time: float = None):
        """Update run status and training time."""
        with self.get_session() as session:
            run = session.query(Run).filter(Run.id == run_id).first()
            if run:
                run.status = status
                if training_time is not None:
                    run.training_time = training_time
                session.commit()
    
    def update_run_metrics(self, run_id: str, metrics: Dict[str, Any]):
        """Update run metrics and extract key performance indicators."""
        with self.get_session() as session:
            run = session.query(Run).filter(Run.id == run_id).first()
            if run:
                # Store full metrics as JSON
                run.metrics_json = json.dumps(metrics)
                
                # Extract and store key metrics to dedicated columns
                if 'accuracy' in metrics:
                    run.accuracy_mean = float(metrics['accuracy'])
                elif 'accuracy_mean' in metrics:
                    run.accuracy_mean = float(metrics['accuracy_mean'])
                
                if 'f1' in metrics:
                    run.f1_score_mean = float(metrics['f1'])
                elif 'f1_score_mean' in metrics:
                    run.f1_score_mean = float(metrics['f1_score_mean'])
                elif 'f1_mean' in metrics:
                    run.f1_score_mean = float(metrics['f1_mean'])
                
                session.commit()
                print(f"✅ Stored metrics - Accuracy: {run.accuracy_mean}, F1: {run.f1_score_mean}")
    
    def update_run_metadata(self, run_id: str, metadata: Dict[str, Any]):
        """Update run metadata."""
        with self.get_session() as session:
            run = session.query(Run).filter(Run.id == run_id).first()
            if run:
                # For now, we can store preprocessing metadata in the status or ignore it
                # The main purpose is to not crash during training
                pass
    
    def get_run(self, run_id: str) -> Optional[Run]:
        """Get a run by ID."""
        with self.get_session() as session:
            return session.query(Run).filter(Run.id == run_id).first()
    
    def get_runs_by_dataset(self, dataset_id: str) -> List[Run]:
        """Get all runs for a dataset."""
        with self.get_session() as session:
            return session.query(Run).filter(Run.dataset_id == dataset_id).order_by(Run.created_at.desc()).all()
    
    def get_all_runs(self) -> List[Run]:
        """Get all runs."""
        with self.get_session() as session:
            return session.query(Run).order_by(Run.created_at.desc()).all()
    
    def promote_run_to_production(self, run_id: str):
        """Promote a run to production status."""
        with self.get_session() as session:
            run = session.query(Run).filter(Run.id == run_id).first()
            if not run:
                raise ValueError(f"Run {run_id} not found")
            
            # Demote any existing production runs of the same model type
            session.query(Run).filter(
                Run.model_type == run.model_type,
                Run.is_production == True
            ).update({"is_production": False})
            
            # Promote this run
            run.is_production = True
            session.commit()
    
    def get_production_run(self, model_type: str) -> Optional[Run]:
        """Get the current production run for a model type."""
        with self.get_session() as session:
            return session.query(Run).filter(
                Run.model_type == model_type,
                Run.is_production == True,
                Run.status == "completed"
            ).first()
    
    def create_artifact(self, run_id: str, path: str, kind: str) -> str:
        """Create an artifact record."""
        artifact_id = str(uuid.uuid4())
        
        artifact = Artifact(
            id=artifact_id,
            run_id=run_id,
            path=path,
            kind=kind
        )
        
        with self.get_session() as session:
            session.add(artifact)
            session.commit()
        
        return artifact_id
    
    def get_artifacts_by_run(self, run_id: str) -> List[Artifact]:
        """Get all artifacts for a run."""
        with self.get_session() as session:
            return session.query(Artifact).filter(Artifact.run_id == run_id).all()


class StorageService:
    """Service for file storage operations."""
    
    def __init__(self, base_path: str = "./data"):
        self.base_path = base_path
        self.datasets_path = os.path.join(base_path, "datasets")
        self.models_path = os.path.join(base_path, "models")
        self.artifacts_path = os.path.join(base_path, "artifacts")
        
        # Create directories
        os.makedirs(self.datasets_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.artifacts_path, exist_ok=True)
    
    def save_dataset(self, dataset_id: str, data: bytes, filename: str) -> str:
        """Save dataset file."""
        file_path = os.path.join(self.datasets_path, f"{dataset_id}_{filename}")
        with open(file_path, "wb") as f:
            f.write(data)
        return file_path
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset from file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Try to read as CSV
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")
        
        # Check for label column (required)
        if "label" not in df.columns:
            raise ValueError("Dataset must contain a 'label' column")
        
        # Check for text content columns and create 'text' column if needed
        if "text" not in df.columns:
            text_parts = []
            
            # Priority order for combining text fields
            if "subject" in df.columns:
                subject_text = df["subject"].fillna("").astype(str)
                subject_text = subject_text[subject_text.str.strip() != ""]
                if len(subject_text) > 0:
                    text_parts.append("Subject: " + subject_text)
            
            if "body" in df.columns:
                body_text = df["body"].fillna("").astype(str)
                body_text = body_text[body_text.str.strip() != ""]
                if len(body_text) > 0:
                    text_parts.append("Body: " + body_text)
            
            if "sender" in df.columns:
                sender_text = df["sender"].fillna("").astype(str)
                sender_text = sender_text[sender_text.str.strip() != ""]
                if len(sender_text) > 0:
                    text_parts.append("From: " + sender_text)
            
            # If we have text parts, combine them
            if text_parts:
                # Combine all available text fields for each row
                combined_texts = []
                for idx in df.index:
                    row_text_parts = []
                    
                    if "subject" in df.columns and pd.notna(df.loc[idx, "subject"]) and str(df.loc[idx, "subject"]).strip():
                        row_text_parts.append(f"Subject: {str(df.loc[idx, 'subject']).strip()}")
                    
                    if "body" in df.columns and pd.notna(df.loc[idx, "body"]) and str(df.loc[idx, "body"]).strip():
                        row_text_parts.append(f"Body: {str(df.loc[idx, 'body']).strip()}")
                    
                    if "sender" in df.columns and pd.notna(df.loc[idx, "sender"]) and str(df.loc[idx, "sender"]).strip():
                        row_text_parts.append(f"From: {str(df.loc[idx, 'sender']).strip()}")
                    
                    # Join with separator or use fallback
                    if row_text_parts:
                        combined_texts.append(" | ".join(row_text_parts))
                    else:
                        # Fallback to any non-empty column value
                        fallback_text = ""
                        for col in ["subject", "body", "content", "message", "email_text"]:
                            if col in df.columns and pd.notna(df.loc[idx, col]):
                                fallback_text = str(df.loc[idx, col]).strip()
                                if fallback_text:
                                    break
                        combined_texts.append(fallback_text if fallback_text else "No content")
                
                df["text"] = combined_texts
            else:
                # Look for other common text columns
                possible_text_cols = ["content", "message", "email_text"]
                found_text_col = None
                
                for col in possible_text_cols:
                    if col in df.columns:
                        found_text_col = col
                        break
                
                if found_text_col:
                    df["text"] = df[found_text_col].fillna("").astype(str)
                else:
                    raise ValueError("Dataset must contain at least one text column: 'text', 'body', 'subject', 'content', 'message', or 'email_text'")
        
        # Normalize labels
        df["label"] = df["label"].astype(str).str.lower().str.strip()
        
        # Convert numeric or various text labels to standard format
        label_mapping = {
            "1": "phish", "1.0": "phish", "true": "phish", 
            "phishing": "phish", "spam": "phish", "phish": "phish",
            "0": "ham", "0.0": "ham", "false": "ham", 
            "legitimate": "ham", "ham": "ham", "normal": "ham", "legit": "ham"
        }
        
        df["label"] = df["label"].map(label_mapping)
        
        # Check for unmapped labels
        if df["label"].isna().any():
            unmapped = df[df["label"].isna()]["label"].iloc[0] if len(df[df["label"].isna()]) > 0 else "unknown"
            raise ValueError(f"Cannot map label '{unmapped}' to 'phish' or 'ham'. Supported: 0/1, true/false, phish/ham, spam/legitimate, etc.")
        
        # Validate labels
        valid_labels = {"phish", "ham"}
        unique_labels = set(df["label"].unique())
        invalid_labels = unique_labels - valid_labels
        if invalid_labels:
            raise ValueError(f"Invalid labels found: {invalid_labels}. Expected: {valid_labels}")
        
        return df
    
    def save_model_artifact(self, run_id: str, artifact_name: str, data: Any) -> str:
        """Save model artifact."""
        artifact_dir = os.path.join(self.artifacts_path, run_id)
        os.makedirs(artifact_dir, exist_ok=True)
        
        file_path = os.path.join(artifact_dir, artifact_name)
        
        # Save based on data type
        if isinstance(data, dict):
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
        elif isinstance(data, str):
            with open(file_path, "w") as f:
                f.write(data)
        else:
            # For complex objects, use pickle
            import pickle
            with open(file_path, "wb") as f:
                pickle.dump(data, f)
        
        return file_path
    
    def load_model_artifact(self, artifact_path: str) -> Any:
        """Load model artifact."""
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")
        
        # Try JSON first
        try:
            with open(artifact_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
        
        # Try pickle
        try:
            import pickle
            with open(artifact_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
        
        # Fall back to text
        with open(artifact_path, "r") as f:
            return f.read()
    
    def get_artifact_path(self, run_id: str, artifact_name: str) -> str:
        """Get path for an artifact."""
        return os.path.join(self.artifacts_path, run_id, artifact_name)


def validate_dataset_file(file_path: str) -> Dict[str, Any]:
    """
    Validate that a dataset file has the required format.
    Returns summary statistics.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
    
    if df.empty:
        raise ValueError("Dataset file is empty")
    
    # Check for label column (required)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column")
    
    # Check for text content columns - flexible approach
    has_text_content = False
    text_columns_found = []
    
    if "text" in df.columns:
        has_text_content = True
        text_columns_found.append("text")
    else:
        # Check for other text columns that we can use
        possible_text_cols = ["body", "subject", "content", "message", "email_text", "sender"]
        for col in possible_text_cols:
            if col in df.columns:
                has_text_content = True
                text_columns_found.append(col)
    
    if not has_text_content:
        raise ValueError("Dataset must contain at least one text column: 'text', 'body', 'subject', 'content', 'message', 'email_text', or 'sender'")
    
    # Validate labels - handle numeric and text labels
    df_copy = df.copy()
    df_copy["label"] = df_copy["label"].astype(str).str.lower().str.strip()
    
    # Convert various label formats to standard format
    label_mapping = {
        "1": "phish", "1.0": "phish", "true": "phish", 
        "phishing": "phish", "spam": "phish", "phish": "phish",
        "0": "ham", "0.0": "ham", "false": "ham", 
        "legitimate": "ham", "ham": "ham", "normal": "ham", "legit": "ham"
    }
    
    df_copy["label"] = df_copy["label"].map(label_mapping)
    
    # Check for unmapped labels
    if df_copy["label"].isna().any():
        unmapped_labels = df[df_copy["label"].isna()]["label"].unique()
        raise ValueError(f"Cannot map label(s) {list(unmapped_labels)} to 'phish' or 'ham'. Supported formats: 0/1, true/false, phish/ham, spam/legitimate, etc.")
    
    # Calculate statistics and convert to native Python types
    label_counts = df_copy["label"].value_counts()
    n_phish = int(label_counts.get("phish", 0))  # Convert numpy.int64 to int
    n_ham = int(label_counts.get("ham", 0))      # Convert numpy.int64 to int
    
    return {
        "n_rows": int(len(df)),  # Convert numpy.int64 to int
        "n_phish": n_phish,
        "n_ham": n_ham,
        "columns": list(df.columns),
        "text_columns_found": text_columns_found,
        "valid": True
    }