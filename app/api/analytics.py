"""
Analytics API endpoints for the phishing detection system.
Provides comprehensive statistics, user feedback, and system metrics.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.services.analytics import AnalyticsService
from app.services.auth import AuthenticationService
from app.services.storage import DatabaseService
from ..schemas.dto import FeedbackRequest, AnalyticsRequest

# Initialize router
router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Global service instances (will be set in main.py)
db_service: DatabaseService = None
analytics_service: AnalyticsService = None
auth_service: AuthenticationService = None


@router.get("/dashboard/stats")
async def get_enhanced_dashboard_stats(days: int = 30) -> Dict[str, Any]:
    """Get comprehensive dashboard statistics with enhanced analytics."""
    try:
        if not analytics_service:
            raise HTTPException(status_code=500, detail="Analytics service not initialized")
        
        stats = analytics_service.get_dashboard_stats(days=days)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard stats: {str(e)}")


@router.get("/trends")
async def get_performance_trends(days: int = 30) -> Dict[str, List]:
    """Get model performance trends over time."""
    try:
        if not analytics_service:
            raise HTTPException(status_code=500, detail="Analytics service not initialized")
        
        trends = analytics_service.get_model_performance_trends(days=days)
        return trends
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trends: {str(e)}")


@router.get("/user-analytics")
async def get_user_analytics(days: int = 30) -> Dict[str, Any]:
    """Get user behavior analytics."""
    try:
        if not analytics_service:
            raise HTTPException(status_code=500, detail="Analytics service not initialized")
        
        user_analytics = analytics_service.get_user_analytics(days=days)
        return user_analytics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user analytics: {str(e)}")


@router.get("/threat-intel")
async def get_threat_intelligence(days: int = 7) -> Dict[str, Any]:
    """Get threat intelligence and patterns."""
    try:
        if not analytics_service:
            raise HTTPException(status_code=500, detail="Analytics service not initialized")
        
        threat_intel = analytics_service.get_threat_intelligence(days=days)
        return threat_intel
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get threat intelligence: {str(e)}")


@router.get("/pending-reviews")
async def get_pending_reviews(limit: int = 50) -> List[Dict[str, Any]]:
    """Get pending feedback reports for admin review."""
    try:
        if not analytics_service:
            raise HTTPException(status_code=500, detail="Analytics service not initialized")
        
        pending_reviews = analytics_service.get_pending_reviews(limit=limit)
        return pending_reviews
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pending reviews: {str(e)}")


@router.post("/feedback")
async def submit_feedback(
    classification_id: str,
    feedback_type: str,
    reported_label: Optional[str] = None,
    confidence_level: Optional[int] = None,
    comments: Optional[str] = None,
    evidence_url: Optional[str] = None
) -> Dict[str, Any]:
    """Submit user feedback about a classification."""
    try:
        if not analytics_service:
            raise HTTPException(status_code=500, detail="Analytics service not initialized")
        
        # For demo purposes, use a default user ID
        # In production, you would get this from authentication
        user_id = "demo-user-001"
        
        feedback_id = analytics_service.submit_feedback(
            classification_id=classification_id,
            user_id=user_id,
            feedback_type=feedback_type,
            reported_label=reported_label,
            confidence_level=confidence_level,
            comments=comments,
            evidence_url=evidence_url
        )
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "message": "Feedback submitted successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.post("/review-feedback")
async def review_feedback(
    feedback_id: str,
    decision: str,
    admin_notes: Optional[str] = None
) -> Dict[str, Any]:
    """Admin review of user feedback (admin only)."""
    try:
        if not analytics_service:
            raise HTTPException(status_code=500, detail="Analytics service not initialized")
        
        # For demo purposes, use a default admin user ID
        # In production, you would verify admin authentication
        reviewer_id = "admin"
        
        success = analytics_service.review_feedback(
            feedback_id=feedback_id,
            reviewer_id=reviewer_id,
            decision=decision,
            admin_notes=admin_notes
        )
        
        if success:
            return {
                "success": True,
                "message": f"Feedback {decision} successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Feedback not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to review feedback: {str(e)}")


@router.get("/system-health")
async def get_system_health() -> Dict[str, Any]:
    """Get system health status."""
    try:
        # Check database connectivity
        db_healthy = True
        try:
            if db_service:
                with db_service.get_session() as session:
                    session.execute("SELECT 1")
        except:
            db_healthy = False
        
        return {
            "api_server": "online",
            "database": "connected" if db_healthy else "disconnected",
            "analytics": "active" if analytics_service else "inactive",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "api_server": "online",
            "database": "unknown",
            "analytics": "unknown",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }