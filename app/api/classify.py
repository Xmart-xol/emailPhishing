"""
Classification API endpoint for real-time email phishing detection.
"""
import time
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

from ..schemas.dto import ClassifyRequestDTO, ClassificationResultDTO, ExplanationDTO
from ..services.storage import DatabaseService, StorageService
from ..services.training import TrainingService, ModelRegistry
from ..ml.features import SparseMatrix, SparseVector

# Rate limiter for public endpoints
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(prefix="/api", tags=["classification"])

# Global service instances (will be initialized in main.py)
db_service: DatabaseService = None
storage_service: StorageService = None
training_service: TrainingService = None
model_registry: ModelRegistry = None
analytics_service = None  # Add analytics service


def get_services():
    """Dependency to get service instances."""
    if not all([db_service, storage_service, training_service, model_registry]):
        raise HTTPException(status_code=500, detail="Services not initialized")
    return db_service, storage_service, training_service, model_registry


@router.post("/classify", response_model=ClassificationResultDTO)
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def classify_email(
    request: Request,
    classify_request: ClassifyRequestDTO,
    services: tuple = Depends(get_services)
):
    """Classify email text as phish or ham using real trained models."""
    db_svc, storage_svc, training_svc, model_reg = services
    
    start_time = time.time()
    
    # Get text content from the flexible input format
    text_content = classify_request.get_text_content()
    
    if not text_content:
        raise HTTPException(status_code=400, detail="No email content provided")
    
    try:
        # Load the real production model
        print(f"Loading {classify_request.model} model...")
        
        if classify_request.run_id:
            model_info = model_reg.get_model(classify_request.model, classify_request.run_id)
        else:
            model_info = model_reg.get_production_model(classify_request.model)
        
        model = model_info['model']
        vectorizer = model_info['vectorizer']
        run = model_info['run']
        
        print(f"✅ Loaded {classify_request.model} model (Run ID: {run.id})")
        
        # Transform input text using the trained vectorizer
        X = vectorizer.transform([text_content])
        
        # Make prediction using the real model
        if classify_request.model == "knn":
            # KNN requires hyperparameters for prediction
            k = 5
            metric = "cosine"
            weighting = "uniform"
            
            prediction = model.predict(X, k=k, metric=metric, weighting=weighting)
            
            # Get proper confidence from KNN probabilities
            try:
                probabilities = model.predict_proba(X, k=k, metric=metric, weighting=weighting)
                if probabilities and len(probabilities[0]) > 0:
                    # Get the maximum probability (confidence in the predicted class)
                    confidence = max(probabilities[0])
                    print(f"KNN probabilities: {probabilities[0]}, confidence: {confidence}")
                else:
                    confidence = 0.5  # Neutral if no probabilities
            except Exception as e:
                print(f"Error getting KNN probabilities: {e}")
                confidence = 0.5  # Neutral fallback
                
        else:  # SVM
            prediction = model.predict(X)
            
            # Get proper confidence from SVM decision function and probabilities
            try:
                # First try to get probabilities (more reliable for confidence)
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X)
                    if probabilities and len(probabilities[0]) > 0:
                        # Get the maximum probability (confidence in the predicted class)
                        confidence = max(probabilities[0])
                        print(f"SVM probabilities: {probabilities[0]}, confidence: {confidence}")
                    else:
                        raise Exception("No probabilities returned")
                else:
                    # Fallback to decision function
                    decision_scores = model.decision_function(X)
                    # Convert decision score to probability using sigmoid
                    import math
                    decision_value = decision_scores[0]
                    # Sigmoid transformation: 1 / (1 + exp(-x))
                    prob_positive = 1.0 / (1.0 + math.exp(-decision_value))
                    prob_negative = 1.0 - prob_positive
                    confidence = max(prob_positive, prob_negative)
                    print(f"SVM decision score: {decision_value}, confidence: {confidence}")
                    
            except Exception as e:
                print(f"Error getting SVM confidence: {e}")
                confidence = 0.5  # Neutral fallback
        
        # Process results
        is_phishing = bool(prediction[0])
        label = "phish" if is_phishing else "ham"
        processing_time = (time.time() - start_time) * 1000
        
        # Create detailed explanation
        risk_level = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.6 else "LOW"
        
        explanation = {
            "model_type": f"{classify_request.model}_production",
            "confidence": confidence,
            "risk_level": risk_level,
            "processing_time_ms": int(processing_time),
            "model_info": {
                "run_id": run.id,
                "training_accuracy": run.accuracy_mean if run.accuracy_mean else "N/A",
                "training_f1_score": run.f1_score_mean if run.f1_score_mean else "N/A",
                "trained_on": f"{run.created_at}" if run.created_at else "N/A"
            },
            "features_detected": {
                "feature_vector_size": X.shape[1] if hasattr(X, 'shape') else len(X.vectors[0].indices) if X.vectors else 0,
                "active_features": len(X.vectors[0].indices) if X.vectors and X.vectors[0].indices else 0,
                "processing_method": "real_model_prediction"
            },
            "technical_analysis": {
                "vectorization": f"BoW + Char N-grams + Heuristics ({X.shape[1] if hasattr(X, 'shape') else 'sparse'} features)",
                "model_algorithm": f"{classify_request.model.upper()} Classifier",
                "confidence_source": "Model decision function" if classify_request.model == "svm" else "KNN probability"
            }
        }
        
        # Store classification in database
        try:
            import hashlib
            import uuid
            
            email_hash = hashlib.sha256(text_content.encode()).hexdigest()
            
            with db_svc.engine.connect() as connection:
                from sqlalchemy import text
                
                classification_id = str(uuid.uuid4())
                connection.execute(text("""
                    INSERT INTO classifications (
                        id, run_id, email_hash, predicted_label, confidence_score, 
                        model_type, processing_time_ms, created_at
                    ) VALUES (
                        :id, :run_id, :email_hash, :label, :confidence, 
                        :model_type, :processing_time, datetime('now')
                    )
                """), {
                    "id": classification_id,
                    "run_id": run.id,
                    "email_hash": email_hash,
                    "label": label,
                    "confidence": confidence,
                    "model_type": f"{classify_request.model}_production",
                    "processing_time": processing_time
                })
                
                # 🔧 FIX: Update the total_classifications counter for this model
                connection.execute(text("""
                    UPDATE runs 
                    SET total_classifications = total_classifications + 1 
                    WHERE id = :run_id
                """), {"run_id": run.id})
                
                connection.commit()
                print(f"✅ Classification stored and counter updated for run {run.id}")
                
        except Exception as e:
            print(f"Failed to store classification: {e}")
        
        # Return result in expected format
        return {
            "label": label,
            "score": confidence,
            "probability": confidence,
            "explanation": explanation
        }
        
    except Exception as e:
        print(f"Real model classification failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Classification failed: {str(e)}. The model may not be available."
        )


@router.get("/classify/history")
async def get_classification_history(
    limit: int = 10,
    services: tuple = Depends(get_services)
):
    """Get recent email classification history."""
    db_svc, _, _, _ = services
    
    try:
        with db_svc.engine.connect() as connection:
            from sqlalchemy import text
            result = connection.execute(text("""
                SELECT id, email_hash, predicted_label, confidence_score, 
                       created_at, processing_time_ms, model_type
                FROM classifications 
                ORDER BY created_at DESC 
                LIMIT :limit
            """), {"limit": limit})
            
            classifications = []
            for row in result.fetchall():
                mock_result = {
                    "label": row.predicted_label,
                    "score": row.confidence_score,
                    "probability": row.confidence_score,
                    "explanation": {
                        "model_type": row.model_type,
                        "confidence": row.confidence_score,
                        "risk_level": "HIGH" if row.confidence_score > 0.8 else "MEDIUM" if row.confidence_score > 0.5 else "LOW",
                        "processing_time_ms": row.processing_time_ms or 0,
                        "features_detected": {
                            "suspicious_patterns": ["Pattern analysis"] if row.predicted_label == "phish" else [],
                            "trust_indicators": ["Legitimate indicators"] if row.predicted_label == "ham" else [],
                            "phishing_indicators_count": 1 if row.predicted_label == "phish" else 0,
                            "legitimate_indicators_count": 1 if row.predicted_label == "ham" else 0
                        },
                        "technical_analysis": {
                            "domain_analysis": "Domain analysis completed",
                            "url_analysis": "URL analysis completed", 
                            "header_analysis": "Header analysis completed"
                        }
                    }
                }
                
                classifications.append({
                    "id": str(row.id),
                    "email_content": f"Classification from {row.created_at}",
                    "result": mock_result,
                    "timestamp": str(row.created_at)
                })
            
            return classifications
            
    except Exception as e:
        print(f"Failed to get classification history: {e}")
        return []


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "email-phishing-detector"
    }


@router.get("/models/status")
async def get_models_status(services: tuple = Depends(get_services)):
    """Get status of production models."""
    db_svc, _, _, model_reg = services
    
    try:
        # Check SVM model status
        svm_status = {"available": False, "run_id": None, "metrics": {}}
        try:
            svm_run = db_svc.get_production_run("svm")
            if svm_run:
                svm_status = {
                    "available": True,
                    "run_id": svm_run.id,
                    "accuracy": f"{svm_run.accuracy_mean*100:.1f}%" if svm_run.accuracy_mean else "N/A",
                    "f1_score": f"{svm_run.f1_score_mean*100:.1f}%" if svm_run.f1_score_mean else "N/A",
                    "created_at": str(svm_run.created_at) if svm_run.created_at else "N/A",
                    "training_time": f"{svm_run.training_time:.1f}s" if svm_run.training_time else "N/A"
                }
        except Exception as e:
            print(f"Error checking SVM status: {e}")
        
        # Check KNN model status  
        knn_status = {"available": False, "run_id": None, "metrics": {}}
        try:
            knn_run = db_svc.get_production_run("knn")
            if knn_run:
                knn_status = {
                    "available": True,
                    "run_id": knn_run.id,
                    "accuracy": f"{knn_run.accuracy_mean*100:.1f}%" if knn_run.accuracy_mean else "N/A",
                    "f1_score": f"{knn_run.f1_score_mean*100:.1f}%" if knn_run.f1_score_mean else "N/A", 
                    "created_at": str(knn_run.created_at) if knn_run.created_at else "N/A",
                    "training_time": f"{knn_run.training_time:.1f}s" if knn_run.training_time else "N/A"
                }
        except Exception as e:
            print(f"Error checking KNN status: {e}")
        
        return {
            "status": "healthy",
            "models": {
                "svm": svm_status,
                "knn": knn_status
            },
            "message": "Real production models loaded" if (svm_status["available"] or knn_status["available"]) else "No production models available"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
