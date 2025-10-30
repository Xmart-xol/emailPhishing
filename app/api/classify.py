"""
Classification API endpoint for real-time email phishing detection.
"""
import time
import re
import urllib.parse
from typing import Dict, Any, List, Optional
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


def analyze_email_content(text_content: str) -> Dict[str, Any]:
    """Perform comprehensive technical analysis of email content."""
    
    # Domain Analysis
    domain_analysis = analyze_domains(text_content)
    
    # URL Analysis  
    url_analysis = analyze_urls(text_content)
    
    # Header Analysis
    header_analysis = analyze_headers(text_content)
    
    # Content Analysis
    content_analysis = analyze_content_patterns(text_content)
    
    return {
        "domain_analysis": domain_analysis,
        "url_analysis": url_analysis,
        "header_analysis": header_analysis,
        "content_analysis": content_analysis
    }


def analyze_domains(text_content: str) -> Dict[str, Any]:
    """Analyze domains found in the email content."""
    # Extract domains from email addresses and URLs
    email_pattern = r'\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b'
    url_pattern = r'https?://([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
    
    email_domains = re.findall(email_pattern, text_content, re.IGNORECASE)
    url_domains = re.findall(url_pattern, text_content, re.IGNORECASE)
    
    all_domains = list(set(email_domains + url_domains))
    
    # Analyze each domain for suspicious characteristics
    suspicious_indicators = []
    legitimate_indicators = []
    
    for domain in all_domains:
        domain_lower = domain.lower()
        
        # Check for suspicious patterns
        if len(domain_lower) > 30:
            suspicious_indicators.append(f"Long domain name: {domain}")
        
        if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
            suspicious_indicators.append(f"IP address instead of domain: {domain}")
            
        if domain_lower.count('-') > 3:
            suspicious_indicators.append(f"Multiple hyphens in domain: {domain}")
            
        if re.search(r'(secure|login|verify|update|confirm|account)', domain_lower):
            suspicious_indicators.append(f"Security-themed domain: {domain}")
            
        # Check for common typosquatting patterns
        common_brands = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook', 'twitter']
        for brand in common_brands:
            if brand in domain_lower and domain_lower != f"{brand}.com":
                suspicious_indicators.append(f"Possible typosquatting of {brand}: {domain}")
        
        # Check for legitimate indicators
        well_known_domains = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'icloud.com',
            'amazon.com', 'google.com', 'microsoft.com', 'apple.com', 'paypal.com'
        ]
        if domain_lower in well_known_domains:
            legitimate_indicators.append(f"Well-known legitimate domain: {domain}")
    
    return {
        "total_domains": len(all_domains),
        "domains_found": all_domains[:10],  # Limit to first 10 for display
        "suspicious_indicators": suspicious_indicators,
        "legitimate_indicators": legitimate_indicators,
        "risk_score": min(len(suspicious_indicators) * 20, 100),
        "analysis_summary": f"Found {len(all_domains)} unique domains with {len(suspicious_indicators)} suspicious indicators"
    }


def analyze_urls(text_content: str) -> Dict[str, Any]:
    """Analyze URLs found in the email content."""
    # Extract all URLs
    url_pattern = r'https?://[^\s<>"\'`|(){}[\]]+|www\.[^\s<>"\'`|(){}[\]]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/[^\s<>"\'`|(){}[\]]*'
    urls = re.findall(url_pattern, text_content, re.IGNORECASE)
    
    suspicious_indicators = []
    legitimate_indicators = []
    
    for url in urls:
        url_lower = url.lower()
        
        # Parse URL components
        try:
            parsed = urllib.parse.urlparse(url if url.startswith('http') else f'http://{url}')
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
            
            # Check for suspicious URL patterns
            if len(url) > 100:
                suspicious_indicators.append(f"Very long URL: {url[:50]}...")
                
            if query and len(query) > 50:
                suspicious_indicators.append(f"Long query string in URL: {domain}")
                
            if path.count('/') > 5:
                suspicious_indicators.append(f"Deep path structure: {domain}{path}")
                
            # Check for URL shorteners
            shorteners = ['bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'ow.ly', 'short.link']
            if any(shortener in domain for shortener in shorteners):
                suspicious_indicators.append(f"URL shortener detected: {domain}")
                
            # Check for suspicious parameters
            suspicious_params = ['redirect', 'url', 'link', 'goto', 'target']
            if any(param in query.lower() for param in suspicious_params):
                suspicious_indicators.append(f"Redirect parameter detected: {domain}")
                
            # Check for HTTPS
            if url.startswith('http://'):
                suspicious_indicators.append(f"Non-HTTPS URL: {domain}")
            elif url.startswith('https://'):
                legitimate_indicators.append(f"HTTPS URL: {domain}")
                
        except Exception as e:
            suspicious_indicators.append(f"Malformed URL: {url[:30]}...")
    
    return {
        "total_urls": len(urls),
        "urls_found": urls[:5],  # Limit to first 5 for display
        "suspicious_indicators": suspicious_indicators,
        "legitimate_indicators": legitimate_indicators,
        "risk_score": min(len(suspicious_indicators) * 15, 100),
        "analysis_summary": f"Found {len(urls)} URLs with {len(suspicious_indicators)} suspicious indicators"
    }


def analyze_headers(text_content: str) -> Dict[str, Any]:
    """Analyze email header information if present in content."""
    suspicious_indicators = []
    legitimate_indicators = []
    
    # Look for header-like patterns in the text
    header_patterns = {
        'from': r'from:\s*(.+)',
        'subject': r'subject:\s*(.+)',
        'reply-to': r'reply-to:\s*(.+)',
        'return-path': r'return-path:\s*(.+)',
        'message-id': r'message-id:\s*(.+)',
        'received': r'received:\s*(.+)'
    }
    
    found_headers = {}
    for header_name, pattern in header_patterns.items():
        matches = re.findall(pattern, text_content, re.IGNORECASE | re.MULTILINE)
        if matches:
            found_headers[header_name] = matches[:3]  # Limit to first 3 matches
    
    # Analyze From field
    if 'from' in found_headers:
        from_addresses = found_headers['from']
        for from_addr in from_addresses:
            # Check for display name spoofing
            if '<' in from_addr and '>' in from_addr:
                display_name = from_addr.split('<')[0].strip()
                email_addr = from_addr.split('<')[1].split('>')[0]
                
                if display_name and email_addr:
                    # Check if display name suggests one domain but email is from another
                    display_lower = display_name.lower()
                    email_lower = email_addr.lower()
                    
                    brands = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'bank']
                    for brand in brands:
                        if brand in display_lower and brand not in email_lower:
                            suspicious_indicators.append(f"Display name spoofing: '{display_name}' vs {email_addr}")
    
    # Analyze Subject
    if 'subject' in found_headers:
        subjects = found_headers['subject']
        for subject in subjects:
            subject_lower = subject.lower()
            
            # Check for urgent/phishing keywords
            urgent_keywords = ['urgent', 'immediate', 'action required', 'verify', 'suspended', 'expires']
            if any(keyword in subject_lower for keyword in urgent_keywords):
                suspicious_indicators.append(f"Urgent language in subject: {subject[:50]}...")
                
            # Check for excessive punctuation
            if subject.count('!') > 2 or subject.count('?') > 2:
                suspicious_indicators.append(f"Excessive punctuation in subject: {subject[:50]}...")
    
    # Check for Reply-To mismatches
    if 'from' in found_headers and 'reply-to' in found_headers:
        from_domains = [addr.split('@')[-1].split('>')[0] for addr in found_headers['from'] if '@' in addr]
        reply_domains = [addr.split('@')[-1].split('>')[0] for addr in found_headers['reply-to'] if '@' in addr]
        
        for from_domain, reply_domain in zip(from_domains, reply_domains):
            if from_domain != reply_domain:
                suspicious_indicators.append(f"Reply-To mismatch: From {from_domain}, Reply-To {reply_domain}")
    
    # If no obvious headers found, analyze the text for header-like information
    if not found_headers:
        # Look for sender information in the text
        sender_patterns = [
            r'from[:\s]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'sent by[:\s]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            r'sender[:\s]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
        ]
        
        for pattern in sender_patterns:
            matches = re.findall(pattern, text_content, re.IGNORECASE)
            if matches:
                found_headers['inferred_sender'] = matches[:2]
                break
    
    return {
        "headers_found": found_headers,
        "suspicious_indicators": suspicious_indicators,
        "legitimate_indicators": legitimate_indicators,
        "risk_score": min(len(suspicious_indicators) * 25, 100),
        "analysis_summary": f"Analyzed email headers, found {len(suspicious_indicators)} suspicious indicators"
    }


def analyze_content_patterns(text_content: str) -> Dict[str, Any]:
    """Analyze content patterns for phishing indicators."""
    suspicious_indicators = []
    legitimate_indicators = []
    
    text_lower = text_content.lower()
    
    # Phishing keywords and phrases
    phishing_keywords = [
        'verify your account', 'account suspended', 'click here', 'act now',
        'limited time', 'expires today', 'confirm identity', 'update payment',
        'security alert', 'unusual activity', 'login attempt', 'winner',
        'congratulations', 'claim now', 'free money', 'urgent action'
    ]
    
    urgency_keywords = [
        'immediate', 'urgent', 'asap', 'expires', 'deadline', 'limited time',
        'act now', 'hurry', 'don\'t delay', 'final notice'
    ]
    
    financial_keywords = [
        'bank account', 'credit card', 'payment', 'refund', 'money',
        'transaction', 'billing', 'invoice', 'charge', 'fee'
    ]
    
    # Count keyword occurrences
    phishing_count = sum(1 for keyword in phishing_keywords if keyword in text_lower)
    urgency_count = sum(1 for keyword in urgency_keywords if keyword in text_lower)
    financial_count = sum(1 for keyword in financial_keywords if keyword in text_lower)
    
    if phishing_count > 0:
        suspicious_indicators.append(f"Contains {phishing_count} phishing-related keywords")
    
    if urgency_count > 2:
        suspicious_indicators.append(f"High urgency language detected ({urgency_count} indicators)")
    
    if financial_count > 3:
        suspicious_indicators.append(f"Multiple financial terms detected ({financial_count} terms)")
    
    # Check for suspicious patterns
    if re.search(r'\$\d+', text_content):
        suspicious_indicators.append("Contains monetary amounts")
    
    if len(re.findall(r'[A-Z]{3,}', text_content)) > 5:
        suspicious_indicators.append("Excessive use of capital letters")
    
    # Check for legitimate indicators
    if any(phrase in text_lower for phrase in ['unsubscribe', 'privacy policy', 'terms of service']):
        legitimate_indicators.append("Contains standard email footer elements")
    
    if re.search(r'\d{4}-\d{4}-\d{4}-\d{4}', text_content):
        suspicious_indicators.append("Contains credit card-like number pattern")
    
    return {
        "phishing_keywords_count": phishing_count,
        "urgency_keywords_count": urgency_count,
        "financial_keywords_count": financial_count,
        "suspicious_indicators": suspicious_indicators,
        "legitimate_indicators": legitimate_indicators,
        "risk_score": min((phishing_count * 10) + (urgency_count * 5) + (financial_count * 3), 100),
        "analysis_summary": f"Content analysis detected {len(suspicious_indicators)} suspicious patterns"
    }


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
        # Perform comprehensive email analysis
        email_analysis = analyze_email_content(text_content)
        
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
        
        # Create detailed explanation with real technical analysis
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
                "domain_analysis": email_analysis["domain_analysis"],
                "url_analysis": email_analysis["url_analysis"],
                "header_analysis": email_analysis["header_analysis"],
                "content_analysis": email_analysis["content_analysis"],
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
