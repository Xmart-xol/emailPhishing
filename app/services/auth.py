"""
User authentication and authorization service for the phishing detection system.
Handles user management, role-based access control, and session management.
"""
import hashlib
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from .storage import DatabaseService
from ..schemas.models_enhanced import User, UserSession, UserRole


class AuthenticationService:
    """Service for user authentication and authorization."""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
        self.session_duration_hours = 24  # Session expires after 24 hours
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        full_name: str = None,
        role: str = UserRole.USER
    ) -> Dict[str, Any]:
        """Create a new user account."""
        with self.db_service.get_session() as session:
            # Check if username or email already exists
            existing_user = session.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first()
            
            if existing_user:
                return {
                    "success": False,
                    "error": "Username or email already exists"
                }
            
            # Hash password
            hashed_password = self._hash_password(password)
            
            # Create user
            user = User(
                id=self._generate_id(),
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                role=role,
                is_active=True
            )
            
            session.add(user)
            session.commit()
            
            return {
                "success": True,
                "user_id": user.id,
                "username": user.username,
                "role": user.role
            }
    
    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str = None,
        user_agent: str = None
    ) -> Dict[str, Any]:
        """Authenticate user and create session."""
        with self.db_service.get_session() as session:
            user = session.query(User).filter(
                User.username == username,
                User.is_active == True
            ).first()
            
            if not user or not self._verify_password(password, user.hashed_password):
                return {
                    "success": False,
                    "error": "Invalid username or password"
                }
            
            # Update login stats
            user.last_login = datetime.utcnow()
            user.login_count = (user.login_count or 0) + 1
            
            # Create session
            session_token = self._generate_session_token()
            expires_at = datetime.utcnow() + timedelta(hours=self.session_duration_hours)
            
            user_session = UserSession(
                id=self._generate_id(),
                user_id=user.id,
                session_token=session_token,
                ip_address=ip_address,
                user_agent=user_agent,
                expires_at=expires_at
            )
            
            session.add(user_session)
            session.commit()
            
            return {
                "success": True,
                "user_id": user.id,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role,
                "session_token": session_token,
                "expires_at": expires_at.isoformat()
            }
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token and return user info."""
        with self.db_service.get_session() as session:
            user_session = session.query(UserSession).filter(
                UserSession.session_token == session_token,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            ).first()
            
            if not user_session:
                return None
            
            user = session.query(User).filter(
                User.id == user_session.user_id,
                User.is_active == True
            ).first()
            
            if not user:
                return None
            
            return {
                "user_id": user.id,
                "username": user.username,
                "full_name": user.full_name,
                "role": user.role,
                "session_id": user_session.id
            }
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user by deactivating session."""
        with self.db_service.get_session() as session:
            user_session = session.query(UserSession).filter(
                UserSession.session_token == session_token
            ).first()
            
            if user_session:
                user_session.is_active = False
                session.commit()
                return True
            
            return False
    
    def has_permission(self, user_role: str, required_role: str) -> bool:
        """Check if user has required permission level."""
        role_hierarchy = {
            UserRole.USER: 1,
            UserRole.ANALYST: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile information."""
        with self.db_service.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return None
            
            return {
                "user_id": user.id,
                "username": user.username,
                "email": user.email,
                "full_name": user.full_name,
                "role": user.role,
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None,
                "login_count": user.login_count or 0
            }
    
    def update_user_profile(
        self,
        user_id: str,
        full_name: str = None,
        email: str = None,
        current_password: str = None,
        new_password: str = None
    ) -> Dict[str, Any]:
        """Update user profile information."""
        with self.db_service.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            
            if not user:
                return {"success": False, "error": "User not found"}
            
            # Verify current password if changing password
            if new_password:
                if not current_password or not self._verify_password(current_password, user.hashed_password):
                    return {"success": False, "error": "Current password is incorrect"}
                user.hashed_password = self._hash_password(new_password)
            
            # Update other fields
            if full_name is not None:
                user.full_name = full_name
            
            if email is not None:
                # Check if email is already taken by another user
                existing = session.query(User).filter(
                    User.email == email,
                    User.id != user_id
                ).first()
                
                if existing:
                    return {"success": False, "error": "Email already in use"}
                
                user.email = email
            
            session.commit()
            
            return {"success": True, "message": "Profile updated successfully"}
    
    def list_users(self, role_filter: str = None) -> List[Dict[str, Any]]:
        """List all users (admin only)."""
        with self.db_service.get_session() as session:
            query = session.query(User)
            
            if role_filter:
                query = query.filter(User.role == role_filter)
            
            users = query.order_by(User.created_at.desc()).all()
            
            return [
                {
                    "user_id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role,
                    "is_active": user.is_active,
                    "created_at": user.created_at.isoformat(),
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "login_count": user.login_count or 0
                }
                for user in users
            ]
    
    def toggle_user_status(self, admin_user_id: str, target_user_id: str) -> Dict[str, Any]:
        """Activate/deactivate a user (admin only)."""
        with self.db_service.get_session() as session:
            # Verify admin user
            admin = session.query(User).filter(
                User.id == admin_user_id,
                User.role == UserRole.ADMIN
            ).first()
            
            if not admin:
                return {"success": False, "error": "Unauthorized"}
            
            target_user = session.query(User).filter(User.id == target_user_id).first()
            
            if not target_user:
                return {"success": False, "error": "User not found"}
            
            # Cannot deactivate yourself
            if admin_user_id == target_user_id:
                return {"success": False, "error": "Cannot modify your own status"}
            
            # Toggle status
            target_user.is_active = not target_user.is_active
            
            # Deactivate all sessions if user is being deactivated
            if not target_user.is_active:
                sessions = session.query(UserSession).filter(
                    UserSession.user_id == target_user_id
                ).all()
                
                for sess in sessions:
                    sess.is_active = False
            
            session.commit()
            
            return {
                "success": True,
                "message": f"User {'activated' if target_user.is_active else 'deactivated'} successfully",
                "is_active": target_user.is_active
            }
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA256 with salt."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}${password_hash}"
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        try:
            salt, password_hash = hashed_password.split('$')
            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return computed_hash == password_hash
        except:
            return False
    
    def _generate_session_token(self) -> str:
        """Generate secure session token."""
        return secrets.token_urlsafe(32)
    
    def _generate_id(self) -> str:
        """Generate unique ID."""
        import uuid
        return str(uuid.uuid4())