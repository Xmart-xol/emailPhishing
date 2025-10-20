"""
Training service for managing model training and evaluation.
Enhanced with data cleaning and transformation pipeline.
"""
import time
import json
import re
import gc  # Add garbage collection
import threading
import psutil  # Add psutil import at top level
from typing import Dict, Any, List, Tuple, Optional, Union
import pickle
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import sys
from sklearn.model_selection import train_test_split

from app.ml.features import Vectorizer
from app.ml.knn import KNNClassifier
from app.ml.svm import SVMClassifier
from app.ml.utils import stratified_k_fold_split
from app.ml.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score
)
from app.services.storage import StorageService, DatabaseService


class ModelRegistry:
    """Registry for managing trained models and their artifacts."""
    
    def __init__(self, db_service: DatabaseService, storage_service: StorageService):
        self.db_service = db_service
        self.storage_service = storage_service
        self._loaded_models = {}  # Cache for loaded models
    
    def get_production_model(self, model_type: str):
        """Get the current production model for a given type."""
        # Get production run
        run = self.db_service.get_production_run(model_type)
        if not run:
            raise ValueError(f"No production model found for {model_type}")
        
        # Load model if not cached
        if run.id not in self._loaded_models:
            model_path = f"data/models/{run.id}_model.pkl"
            vectorizer_path = f"data/models/{run.id}_vectorizer.pkl"
            
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Model artifacts not found for run {run.id}")
            
            # Load model and vectorizer
            import pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            
            self._loaded_models[run.id] = {
                'model': model,
                'vectorizer': vectorizer,
                'run': run
            }
        
        return self._loaded_models[run.id]
    
    def get_model(self, model_type: str, run_id: str = None):
        """Get a model by type and optional run_id. If no run_id, get production model."""
        if run_id:
            # Load specific model by run_id
            if run_id not in self._loaded_models:
                model_path = f"data/models/{run_id}_model.pkl"
                vectorizer_path = f"data/models/{run_id}_vectorizer.pkl"
                
                if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                    raise FileNotFoundError(f"Model artifacts not found for run {run_id}")
                
                # Load model and vectorizer
                import pickle
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                
                # Get run info
                run = self.db_service.get_run(run_id)
                if not run:
                    raise ValueError(f"Run {run_id} not found")
                
                self._loaded_models[run_id] = {
                    'model': model,
                    'vectorizer': vectorizer,
                    'run': run
                }
            
            return self._loaded_models[run_id]
        else:
            # Get production model
            return self.get_production_model(model_type)
    
    def promote_model_to_production(self, run_id: str):
        """Promote a model to production status."""
        self.db_service.promote_run_to_production(run_id)
        # Clear cache to force reload
        self._loaded_models.clear()
    
    def list_available_models(self, model_type: str = None):
        """List all available trained models."""
        if model_type:
            runs = self.db_service.get_all_runs()
            return [run for run in runs if run.model_type == model_type and run.status == "completed"]
        else:
            return [run for run in self.db_service.get_all_runs() if run.status == "completed"]
    
    def clear_cache(self):
        """Clear the model cache."""
        self._loaded_models.clear()


class DatasetFormat(Enum):
    """Enum for dataset format types."""
    SIMPLE = "simple"
    RICH = "rich"


@dataclass
class DataCleaningConfig:
    """Configuration for data cleaning."""
    remove_duplicates: bool = True
    remove_empty: bool = True
    normalize_text: bool = True
    remove_html: bool = True
    
    def __init__(self, **kwargs):
        self.remove_duplicates = kwargs.get('remove_duplicates', True)
        self.remove_empty = kwargs.get('remove_empty', True)
        self.normalize_text = kwargs.get('normalize_text', True)
        self.remove_html = kwargs.get('remove_html', True)


@dataclass
class TransformationConfig:
    """Configuration for data transformation."""
    lowercase: bool = True
    remove_special_chars: bool = False
    min_length: int = 10
    
    def __init__(self, **kwargs):
        self.lowercase = kwargs.get('lowercase', True)
        self.remove_special_chars = kwargs.get('remove_special_chars', False)
        self.min_length = kwargs.get('min_length', 10)


class DataPreprocessor:
    """Simple data preprocessor for email text."""
    
    def __init__(self, cleaning_config: DataCleaningConfig, transform_config: TransformationConfig):
        self.cleaning_config = cleaning_config
        self.transform_config = transform_config
        self.preprocessing_stats = {}
    
    def preprocess_dataset(self, df: pd.DataFrame, dataset_format: DatasetFormat) -> pd.DataFrame:
        """Preprocess a dataset with basic cleaning and transformation."""
        print(f"   🧹 Preprocessing {len(df)} samples...")
        
        original_size = len(df)
        df_clean = df.copy()
        
        # Basic text cleaning
        if 'text' in df_clean.columns:
            # Remove HTML tags if configured
            if self.cleaning_config.remove_html:
                df_clean['text'] = df_clean['text'].str.replace(r'<[^>]+>', '', regex=True)
            
            # Remove empty texts if configured
            if self.cleaning_config.remove_empty:
                df_clean = df_clean[df_clean['text'].notna()]
                df_clean = df_clean[df_clean['text'].str.len() >= self.transform_config.min_length]
            
            # Normalize text if configured
            if self.transform_config.lowercase:
                df_clean['text'] = df_clean['text'].str.lower()
            
            # Remove special characters if configured
            if self.transform_config.remove_special_chars:
                df_clean['text'] = df_clean['text'].str.replace(r'[^\w\s]', '', regex=True)
        
        # Remove duplicates if configured
        if self.cleaning_config.remove_duplicates and 'text' in df_clean.columns:
            df_clean = df_clean.drop_duplicates(subset=['text'])
        
        # Store preprocessing statistics
        final_size = len(df_clean)
        self.preprocessing_stats = {
            'original_samples': original_size,
            'final_samples': final_size,
            'removed_samples': original_size - final_size,
            'removal_rate': (original_size - final_size) / original_size if original_size > 0 else 0
        }
        
        print(f"   ✅ Preprocessing complete: {final_size} samples remaining")
        return df_clean
    
    def partial_fit(self, texts: List[str]):
        """Partial fit for streaming processing (placeholder)."""
        pass


class MemoryMonitor:
    """Memory monitoring and management utility."""
    
    def __init__(self, max_memory_gb: float = 2.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process(os.getpid())
        self._lock = threading.Lock()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        with self._lock:
            memory_info = self.process.memory_info()
            return {
                'rss_mb': memory_info.rss / (1024 * 1024),
                'vms_mb': memory_info.vms / (1024 * 1024),
                'percent': self.process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / (1024 * 1024)
            }
    
    def is_memory_critical(self, threshold_percent: float = 80.0) -> bool:
        """Check if memory usage is critical."""
        memory_info = self.get_memory_usage()
        return memory_info['percent'] > threshold_percent
    
    def force_cleanup(self):
        """Force aggressive memory cleanup."""
        print("🧹 Forcing aggressive memory cleanup...")
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Clear any numpy caches
        try:
            import numpy as np
            if hasattr(np, 'random'):
                np.random.seed()  # Reset random state to free memory
        except:
            pass
        
        # Platform-specific memory optimization
        try:
            import ctypes
            if sys.platform == "linux":
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
        except:
            pass
        
        print(f"   Memory after cleanup: {self.get_memory_usage()['rss_mb']:.1f}MB")


class TrainingService:
    """Service for model training and evaluation with integrated preprocessing."""
    
    def __init__(self, db_service: DatabaseService, storage_service: StorageService):
        self.db_service = db_service
        self.storage_service = storage_service
        
        # Enhanced memory management
        self.memory_monitor = MemoryMonitor(max_memory_gb=2.0)  # 2GB limit
        self.chunk_size = 500  # Smaller chunks for better memory management
        self.max_samples_in_memory = 5000  # Maximum samples to process at once
        
    def _check_memory_and_cleanup(self, operation: str = "operation") -> float:
        """Check memory usage and perform cleanup if needed."""
        memory_info = self.memory_monitor.get_memory_usage()
        memory_mb = memory_info['rss_mb']
        
        print(f"   💾 Memory usage during {operation}: {memory_mb:.1f}MB ({memory_info['percent']:.1f}%)")
        
        # Force cleanup if memory is critical
        if self.memory_monitor.is_memory_critical(threshold_percent=75.0):
            print(f"⚠️  Critical memory usage detected during {operation}")
            self.memory_monitor.force_cleanup()
            
            # Check memory again after cleanup
            new_memory_info = self.memory_monitor.get_memory_usage()
            print(f"   Memory after cleanup: {new_memory_info['rss_mb']:.1f}MB")
            
            # If still critical, reduce processing parameters
            if new_memory_info['percent'] > 85.0:
                print("⚠️  Memory still critical after cleanup, reducing chunk size")
                self.chunk_size = max(100, self.chunk_size // 2)
                self.max_samples_in_memory = max(1000, self.max_samples_in_memory // 2)
        
        return memory_mb
    
    def train_model(
        self, 
        run_id: str, 
        dataset_path: str, 
        features_config: Dict[str, Any], 
        model_type: str, 
        hyperparams: Dict[str, Any],
        cv_config: Dict[str, Any],
        dataset_format: str = "simple",
        cleaning_config: Dict[str, Any] = None,
        transform_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Train a model with comprehensive memory management."""
        start_time = time.time()
        
        try:
            print(f"🚀 Starting memory-efficient training for run {run_id}")
            initial_memory = self._check_memory_and_cleanup("training_start")
            
            # Update run status
            self.db_service.update_run_status(run_id, "preprocessing")
            
            # Load and check dataset size first
            print(f"📁 Loading dataset: {dataset_path}")
            try:
                # Try to load just the header first to check size
                sample_df = pd.read_csv(dataset_path, nrows=5)
                print(f"   Dataset columns: {list(sample_df.columns)}")
                
                # Get file size to estimate memory requirements
                file_size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
                print(f"   Dataset file size: {file_size_mb:.1f}MB")
                
                # Load full dataset with memory monitoring
                df = self.storage_service.load_dataset(dataset_path)
                dataset_size = len(df)
                print(f"   Loaded {dataset_size} samples")
                
                self._check_memory_and_cleanup("dataset_load")
                
            except Exception as e:
                raise ValueError(f"Failed to load dataset: {str(e)}")
            
            # Determine processing strategy based on dataset size and available memory
            memory_info = self.memory_monitor.get_memory_usage()
            available_memory_mb = memory_info['available_mb']
            
            if dataset_size > self.max_samples_in_memory or file_size_mb > available_memory_mb * 0.3:
                print(f"⚠️  Large dataset detected ({dataset_size} samples, {file_size_mb:.1f}MB)")
                print(f"    Available memory: {available_memory_mb:.1f}MB, using streaming processing")
                return self._train_model_streaming(
                    run_id, dataset_path, features_config, model_type, hyperparams, cv_config,
                    dataset_format, cleaning_config, transform_config, start_time
                )
            else:
                print(f"✅ Dataset size manageable ({dataset_size} samples), using standard processing")
                return self._train_model_standard(
                    run_id, df, features_config, model_type, hyperparams, cv_config,
                    dataset_format, cleaning_config, transform_config, start_time
                )
            
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
            # Force cleanup on error
            self.memory_monitor.force_cleanup()
            
            # Update run status to failed
            training_time = time.time() - start_time
            self.db_service.update_run_status(run_id, "failed", training_time)
            raise e
    
    def _train_model_streaming(
        self, run_id: str, dataset_path: str, features_config: Dict[str, Any], 
        model_type: str, hyperparams: Dict[str, Any], cv_config: Dict[str, Any],
        dataset_format: str, cleaning_config: Dict[str, Any], 
        transform_config: Dict[str, Any], start_time: float
    ) -> Dict[str, Any]:
        """Train model using streaming processing for very large datasets."""
        print("🌊 Using streaming processing strategy...")
        
        # Set up preprocessing configurations
        cleaning_cfg = DataCleaningConfig(**(cleaning_config or {}))
        transform_cfg = TransformationConfig(**(transform_config or {}))
        preprocessor = DataPreprocessor(cleaning_cfg, transform_cfg)
        
        # Process dataset in streaming chunks
        processed_texts = []
        processed_labels = []
        total_processed = 0
        
        # First, load the full dataset using storage service to get proper text column
        print("📁 Loading full dataset to create text column...")
        df_full = self.storage_service.load_dataset(dataset_path)
        
        # Now process in chunks from the properly formatted dataset
        total_rows = len(df_full)
        
        for start_idx in range(0, total_rows, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, total_rows)
            chunk_num = (start_idx // self.chunk_size) + 1
            
            print(f"   Processing chunk {chunk_num} ({end_idx - start_idx} samples)...")
            
            try:
                # Get chunk from the properly formatted dataset
                chunk = df_full.iloc[start_idx:end_idx].copy()
                
                # Process chunk
                dataset_fmt = DatasetFormat.RICH if dataset_format == "rich" else DatasetFormat.SIMPLE
                chunk_clean = preprocessor.preprocess_dataset(chunk, dataset_fmt)
                
                if len(chunk_clean) > 0:
                    # Extract texts and labels
                    chunk_texts = chunk_clean["text"].tolist()
                    chunk_labels = [1 if label == "phish" else 0 for label in chunk_clean["label"]]
                    
                    processed_texts.extend(chunk_texts)
                    processed_labels.extend(chunk_labels)
                    total_processed += len(chunk_clean)
                
                # Cleanup chunk data immediately
                del chunk, chunk_clean
                if 'chunk_texts' in locals():
                    del chunk_texts, chunk_labels
                
                # Check memory after each chunk
                self._check_memory_and_cleanup(f"chunk_{chunk_num}")
                
                # If memory is still critical, process what we have so far
                if self.memory_monitor.is_memory_critical(threshold_percent=80.0) and len(processed_texts) > 1000:
                    print("⚠️  Memory critical, processing accumulated data...")
                    break
                    
            except Exception as e:
                print(f"   Error processing chunk {chunk_num}: {str(e)}")
                continue
        
        # Free the full dataset
        del df_full
        self._check_memory_and_cleanup("dataset_cleanup")
        
        print(f"📊 Streaming processing complete: {total_processed} samples processed")
        
        if len(processed_texts) == 0:
            raise ValueError("No valid samples found after preprocessing")
        
        # Continue with standard training on processed data
        return self._complete_streaming_training(
            run_id, processed_texts, processed_labels, features_config, 
            model_type, hyperparams, cv_config, preprocessor, start_time
        )
    
    def _complete_streaming_training(
        self, run_id: str, texts: List[str], labels: List[int], 
        features_config: Dict[str, Any], model_type: str, hyperparams: Dict[str, Any],
        cv_config: Dict[str, Any], preprocessor: DataPreprocessor, start_time: float
    ) -> Dict[str, Any]:
        """Complete training with streaming-processed data."""
        
        print(f"🎯 Completing training with {len(texts)} processed samples")
        self._check_memory_and_cleanup("streaming_training_start")
        
        # Save preprocessing statistics
        preprocessing_stats = preprocessor.preprocessing_stats
        self.db_service.update_run_metadata(run_id, {
            'preprocessing_stats': preprocessing_stats,
            'processing_method': 'streaming',
            'final_dataset_size': len(texts)
        })
        
        # Update status to training
        self.db_service.update_run_status(run_id, "training")
        
        # Create and fit vectorizer with streaming approach
        print("🔤 Creating vectorizer for streaming data...")
        vectorizer = Vectorizer(features_config, seed=42)
        
        # For large datasets, fit on all texts at once (vectorizer doesn't support partial_fit)
        print("   Fitting vectorizer on all processed texts...")
        vectorizer.fit(texts)
        
        self._check_memory_and_cleanup("vectorizer_fit")
        
        # Transform data in chunks to manage memory
        print("🔄 Transforming text data in chunks...")
        X_chunks = []
        chunk_size = 1000
        
        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i+chunk_size]
            X_chunk = vectorizer.transform(chunk_texts)
            X_chunks.append(X_chunk)
            
            # Monitor memory during transformation
            self._check_memory_and_cleanup(f"transform_chunk_{i//chunk_size + 1}")
        
        # Combine chunks efficiently
        print("🔗 Combining feature chunks...")
        if len(X_chunks) == 1:
            X = X_chunks[0]
        else:
            # Convert custom SparseMatrix to scipy sparse format for proper combination
            from scipy import sparse
            import numpy as np
            
            # Convert each chunk to scipy sparse matrix
            scipy_chunks = []
            for chunk in X_chunks:
                if hasattr(chunk, 'vectors'):  # Custom SparseMatrix
                    # Convert to scipy csr_matrix
                    n_samples = len(chunk.vectors)
                    n_features = chunk.vectors[0].dim if chunk.vectors else 0
                    
                    row_indices = []
                    col_indices = []
                    data = []
                    
                    for i, vector in enumerate(chunk.vectors):
                        for col_idx, value in zip(vector.indices, vector.values):
                            row_indices.append(i)
                            col_indices.append(col_idx)
                            data.append(float(value))  # Ensure float type
                    
                    # Create scipy sparse matrix with proper dtype
                    if data:  # Only create matrix if we have data
                        scipy_chunk = sparse.csr_matrix(
                            (np.array(data, dtype=np.float64), 
                             (np.array(row_indices, dtype=np.int32), 
                              np.array(col_indices, dtype=np.int32))),
                            shape=(n_samples, n_features),
                            dtype=np.float64
                        )
                    else:
                        # Create empty matrix with correct shape and dtype
                        scipy_chunk = sparse.csr_matrix(
                            (n_samples, n_features), dtype=np.float64
                        )
                    
                    scipy_chunks.append(scipy_chunk)
                else:
                    # Already a scipy matrix
                    scipy_chunks.append(chunk.astype(np.float64))
            
            # Now combine using scipy's vstack
            scipy_combined = sparse.vstack(scipy_chunks, format='csr', dtype=np.float64)
            
            # Convert back to custom SparseMatrix for the SVM
            print("🔄 Converting back to custom SparseMatrix format...")
            from app.ml.features import SparseVector, SparseMatrix
            
            custom_vectors = []
            scipy_csr = scipy_combined.tocsr()
            
            for i in range(scipy_csr.shape[0]):
                row_start = scipy_csr.indptr[i]
                row_end = scipy_csr.indptr[i + 1]
                
                indices = scipy_csr.indices[row_start:row_end].tolist()
                values = scipy_csr.data[row_start:row_end].tolist()
                
                vector = SparseVector(indices, values, scipy_csr.shape[1])
                custom_vectors.append(vector)
            
            X = SparseMatrix(custom_vectors)
        
        # Free chunk data
        del X_chunks, texts
        self._check_memory_and_cleanup("vectorization_complete")
        
        # Perform memory-efficient cross-validation or train/test split
        if cv_config["type"] == "kfold" or cv_config["type"] == "stratified":
            print("🔄 Starting memory-efficient cross-validation...")
            metrics = self._cross_validate_model_efficient(
                X, labels, model_type, hyperparams, cv_config
            )
            
            # Train final model
            print("🎯 Training final model...")
            model = self._create_model(model_type, hyperparams)
            model.fit(X, labels)
            
        else:
            # Simple train/test split
            print("🔄 Performing memory-efficient train/test split...")
            
            # Use smaller test size for large datasets to save memory
            test_size = 0.1 if len(labels) > 10000 else 0.2
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            # Train model
            model = self._create_model(model_type, hyperparams)
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = self._evaluate_model(model, X_test, y_test)
            
            # Immediate cleanup of test data
            del X_train, X_test, y_train, y_test
            self._check_memory_and_cleanup("training_complete")
        
        # Add preprocessing stats to metrics
        metrics['preprocessing'] = preprocessing_stats
        metrics['processing_method'] = 'streaming'
        
        # Save artifacts with memory monitoring
        print("💾 Saving model artifacts...")
        self._save_model_artifacts(run_id, model, vectorizer, metrics, preprocessor)
        
        # Update run with results
        training_time = time.time() - start_time
        self.db_service.update_run_status(run_id, "completed", training_time)
        self.db_service.update_run_metrics(run_id, metrics)
        
        # Final cleanup
        del X, labels, model, vectorizer
        self.memory_monitor.force_cleanup()
        
        print(f"✅ Streaming training completed in {training_time:.2f} seconds")
        return metrics

    def _train_model_standard(
        self, run_id: str, df: pd.DataFrame, features_config: Dict[str, Any], 
        model_type: str, hyperparams: Dict[str, Any], cv_config: Dict[str, Any],
        dataset_format: str, cleaning_config: Dict[str, Any], 
        transform_config: Dict[str, Any], start_time: float
    ) -> Dict[str, Any]:
        """Train model using standard processing with enhanced memory management."""
        print("🏃 Using standard processing strategy...")
        
        try:
            # Set up preprocessing configurations  
            cleaning_cfg = DataCleaningConfig(**(cleaning_config or {}))
            transform_cfg = TransformationConfig(**(transform_config or {}))
            preprocessor = DataPreprocessor(cleaning_cfg, transform_cfg)
            
            # Preprocess dataset
            self.db_service.update_run_status(run_id, "preprocessing")
            print("🧹 Preprocessing dataset...")
            
            dataset_fmt = DatasetFormat.RICH if dataset_format == "rich" else DatasetFormat.SIMPLE  
            df_clean = preprocessor.preprocess_dataset(df, dataset_fmt)
            
            # Free original dataframe immediately
            del df
            self._check_memory_and_cleanup("preprocessing")
            
            if len(df_clean) == 0:
                raise ValueError("No valid samples found after preprocessing")
            
            print(f"✅ Preprocessing complete: {len(df_clean)} samples")
            
            # Save preprocessing statistics
            preprocessing_stats = preprocessor.preprocessing_stats
            self.db_service.update_run_metadata(run_id, {
                'preprocessing_stats': preprocessing_stats,
                'processing_method': 'standard',
                'final_dataset_size': len(df_clean)
            })
            
            # Extract features efficiently
            print("🔤 Extracting features...")
            self.db_service.update_run_status(run_id, "feature_extraction")
            
            texts = df_clean["text"].tolist()
            labels = [1 if label == "phish" else 0 for label in df_clean["label"]]
            
            # Free preprocessed dataframe
            del df_clean
            self._check_memory_and_cleanup("feature_extraction_prep")
            
            # Create and fit vectorizer
            vectorizer = Vectorizer(features_config, seed=42)
            vectorizer.fit(texts)
            
            self._check_memory_and_cleanup("vectorizer_fit")
            
            # Transform texts to features
            X = vectorizer.transform(texts)
            
            # Free text data immediately after transformation
            del texts
            self._check_memory_and_cleanup("vectorization")
            
            print(f"   Feature matrix shape: {X.shape}")
            
            # Training phase
            print("🎯 Starting model training...")
            self.db_service.update_run_status(run_id, "training")
            
            # Choose training strategy based on cross-validation config
            if cv_config["type"] == "kfold" or cv_config["type"] == "stratified":
                print("🔄 Performing cross-validation...")
                metrics = self._cross_validate_model_efficient(X, labels, model_type, hyperparams, cv_config)
                
                # Train final model on all data
                print("🎯 Training final model on complete dataset...")
                model = self._create_model(model_type, hyperparams)
                model.fit(X, labels)
                
            else:
                # Simple train/test split
                print("🔄 Performing train/test split...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, labels, test_size=0.2, random_state=42, stratify=labels
                )
                
                # Train model
                model = self._create_model(model_type, hyperparams)
                model.fit(X_train, y_train)
                
                # Evaluate
                metrics = self._evaluate_model(model, X_test, y_test)
                
                # Cleanup test data immediately
                del X_train, X_test, y_train, y_test
                self._check_memory_and_cleanup("evaluation")
            
            # Add preprocessing stats to metrics
            metrics['preprocessing'] = preprocessing_stats
            metrics['processing_method'] = 'standard'
            
            # Save model and artifacts
            print("💾 Saving model artifacts...")
            self._save_model_artifacts(run_id, model, vectorizer, metrics, preprocessor)
            
            # Update run status
            training_time = time.time() - start_time
            self.db_service.update_run_status(run_id, "completed", training_time)
            self.db_service.update_run_metrics(run_id, metrics)
            
            # Final cleanup
            del X, labels, model, vectorizer
            self.memory_monitor.force_cleanup()
            
            print(f"✅ Standard training completed in {training_time:.2f} seconds")
            return metrics
            
        except Exception as e:
            print(f"❌ Standard training failed: {str(e)}")
            # Cleanup on error
            self.memory_monitor.force_cleanup()
            raise e

    def _save_model_artifacts(
        self, run_id: str, model, vectorizer, metrics: Dict[str, Any], 
        preprocessor: Optional[DataPreprocessor] = None
    ):
        """Save model artifacts with memory monitoring."""
        try:
            print("💾 Saving model artifacts...")
            
            # Save model
            model_path = f"data/models/{run_id}_model.pkl" 
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self._check_memory_and_cleanup("model_save")
            
            # Save vectorizer
            vectorizer_path = f"data/models/{run_id}_vectorizer.pkl"
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            
            self._check_memory_and_cleanup("vectorizer_save")
            
            # Save preprocessor if available
            if preprocessor:
                preprocessor_path = f"data/models/{run_id}_preprocessor.pkl"
                with open(preprocessor_path, 'wb') as f:
                    pickle.dump(preprocessor, f)
                
                self._check_memory_and_cleanup("preprocessor_save")
            
            # Save metrics
            metrics_path = f"data/models/{run_id}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            print(f"   ✅ Artifacts saved to data/models/{run_id}_*")
            
        except Exception as e:
            print(f"❌ Failed to save artifacts: {str(e)}")
            raise e

    def _aggregate_cv_metrics(self, fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate cross-validation metrics efficiently."""
        if not fold_metrics:
            return {}
        
        # Calculate mean and std for each metric
        aggregated = {}
        
        # Get all metric keys from first fold
        metric_keys = set()
        for fold_metric in fold_metrics:
            metric_keys.update(fold_metric.keys())
        
        for key in metric_keys:
            if key in ['confusion_matrix', 'classification_report']:
                # Skip complex metrics that can't be easily aggregated
                continue
                
            values = []
            for fold_metric in fold_metrics:
                if key in fold_metric and isinstance(fold_metric[key], (int, float)):
                    values.append(float(fold_metric[key]))
            
            if values:
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_values'] = values
        
        # Add fold count
        aggregated['cv_folds'] = len(fold_metrics)
        aggregated['cv_type'] = 'cross_validation'
        
        return aggregated

    def _evaluate_model(self, model, X_test, y_test, hyperparams: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate model with memory-efficient metrics calculation."""
        try:
            print("📊 Evaluating model performance...")
            
            # Make predictions with hyperparameters for KNN
            if hasattr(model, 'predict') and isinstance(model, KNNClassifier):
                # For KNN, pass hyperparameters to predict method
                knn_params = hyperparams or {}
                y_pred = model.predict(
                    X_test,
                    k=knn_params.get("k", 5),
                    metric=knn_params.get("metric", "cosine"),
                    weighting=knn_params.get("weighting", "uniform")
                )
            else:
                # For SVM and other models
                y_pred = model.predict(X_test)
            
            self._check_memory_and_cleanup("predictions")
            
            # Calculate basic metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            # Add confusion matrix
            try:
                cm = confusion_matrix(y_test, y_pred)
                metrics['confusion_matrix'] = cm.tolist() if hasattr(cm, 'tolist') else cm
            except Exception as e:
                print(f"   Warning: Could not calculate confusion matrix: {e}")
            
            # Add AUC-ROC if model supports probability prediction
            try:
                if hasattr(model, 'predict_proba'):
                    if isinstance(model, KNNClassifier):
                        # For KNN, pass hyperparameters to predict_proba
                        knn_params = hyperparams or {}
                        y_proba = model.predict_proba(
                            X_test,
                            k=knn_params.get("k", 5),
                            metric=knn_params.get("metric", "cosine"),
                            weighting=knn_params.get("weighting", "uniform")
                        )
                    else:
                        y_proba = model.predict_proba(X_test)
                    
                    if len(y_proba) > 0 and len(y_proba[0]) > 1:
                        # Convert list of lists to numpy array for roc_auc_score
                        import numpy as np
                        y_proba_array = np.array(y_proba)
                        auc_roc = roc_auc_score(y_test, y_proba_array[:, 1])
                        metrics['auc_roc'] = auc_roc
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                    auc_roc = roc_auc_score(y_test, y_scores)
                    metrics['auc_roc'] = auc_roc
            except Exception as e:
                print(f"   Warning: Could not calculate AUC-ROC: {e}")
            
            self._check_memory_and_cleanup("evaluation")
            
            # Clean up predictions
            del y_pred
            if 'y_proba' in locals():
                del y_proba
            if 'y_scores' in locals():
                del y_scores
            
            print(f"   ✅ Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"   ✅ F1-Score: {metrics.get('f1', 0):.3f}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Model evaluation failed: {str(e)}")
            self.memory_monitor.force_cleanup()
            raise e

    def _cross_validate_model_efficient(
        self, X, y, model_type: str, hyperparams: Dict[str, Any], cv_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Memory-efficient cross-validation implementation."""
        print(f"🔄 Starting memory-efficient {cv_config['type']} cross-validation")
        
        # Reduce folds for large datasets to save memory
        folds = cv_config.get("folds", 5)
        if len(y) > 10000:
            folds = min(folds, 3)  # Max 3 folds for very large datasets
            print(f"   Reducing to {folds} folds due to dataset size")
        
        fold_metrics = []
        
        # Get fold splits
        if cv_config["type"] == "stratified":
            fold_splits = stratified_k_fold_split(X, y, k=folds, shuffle=True, random_state=42)
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=folds, shuffle=True, random_state=42)
            fold_splits = list(kf.split(X))
        
        for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
            print(f"   Processing fold {fold_idx + 1}/{folds}...")
            
            # Monitor memory before fold processing
            self._check_memory_and_cleanup(f"fold_{fold_idx + 1}_start")
            
            try:
                # Create fold data efficiently (avoid copying if possible)
                if hasattr(X, 'getrows'):  # For sparse matrices
                    X_train_fold = X[train_idx]
                    X_test_fold = X[test_idx]
                else:
                    X_train_fold = X[train_idx]
                    X_test_fold = X[test_idx]
                
                y_train_fold = [y[i] for i in train_idx]
                y_test_fold = [y[i] for i in test_idx]
                
                # Train model for this fold
                fold_model = self._create_model(model_type, hyperparams)
                fold_model.fit(X_train_fold, y_train_fold)
                
                # Evaluate fold
                fold_metric = self._evaluate_model(fold_model, X_test_fold, y_test_fold, hyperparams)
                fold_metrics.append(fold_metric)
                
                print(f"     Fold {fold_idx + 1} accuracy: {fold_metric.get('accuracy', 0):.3f}")
                
            except Exception as e:
                print(f"     Error in fold {fold_idx + 1}: {str(e)}")
                # Continue with other folds
                continue
            
            finally:
                # Cleanup fold data immediately
                if 'X_train_fold' in locals():
                    del X_train_fold, X_test_fold, y_train_fold, y_test_fold, fold_model
                
                # Force cleanup after each fold
                gc.collect()
                self._check_memory_and_cleanup(f"fold_{fold_idx + 1}_complete")
        
        if not fold_metrics:
            raise ValueError("All cross-validation folds failed")
        
        # Aggregate metrics
        print("📊 Aggregating cross-validation results...")
        cv_metrics = self._aggregate_cv_metrics(fold_metrics)
        
        return cv_metrics

    def _create_model(self, model_type: str, hyperparams: Dict[str, Any]):
        """Create a model instance based on type and hyperparameters."""
        if model_type == "knn":
            # KNN doesn't take hyperparams in __init__, they go to predict() method
            return KNNClassifier()
        elif model_type == "svm":
            return SVMClassifier(
                C=hyperparams.get("C", 1.0),
                kernel=hyperparams.get("kernel", "linear"),
                gamma=hyperparams.get("gamma", 0.1),
                tol=hyperparams.get("tol", 1e-3),
                max_passes=hyperparams.get("max_passes", 5)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")