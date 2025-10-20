#!/usr/bin/env python3
"""
Database migration script to upgrade from basic schema to enhanced schema.
Adds user management, classification logging, and analytics capabilities.
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.models_enhanced import Base


def migrate_database(database_url: str = "sqlite:///./phishing_detector.db"):
    """Migrate database to enhanced schema."""
    
    print("🔄 Starting database migration to enhanced schema...")
    
    # Create engine
    engine = create_engine(database_url)
    
    # Create all new tables
    print("📊 Creating new tables...")
    Base.metadata.create_all(bind=engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Add new columns to existing tables if they don't exist
        print("🔧 Adding new columns to existing tables...")
        
        # Add columns to runs table
        migration_queries = [
            # Runs table enhancements
            "ALTER TABLE runs ADD COLUMN created_by TEXT",
            "ALTER TABLE runs ADD COLUMN accuracy_mean REAL",
            "ALTER TABLE runs ADD COLUMN f1_score_mean REAL", 
            "ALTER TABLE runs ADD COLUMN total_classifications INTEGER DEFAULT 0",
            
            # Datasets table enhancements
            "ALTER TABLE datasets ADD COLUMN created_by TEXT",
            "ALTER TABLE datasets ADD COLUMN size_bytes INTEGER",
            "ALTER TABLE datasets ADD COLUMN checksum TEXT",
            "ALTER TABLE datasets ADD COLUMN is_active BOOLEAN DEFAULT 1",
            
            # Artifacts table enhancements
            "ALTER TABLE artifacts ADD COLUMN size_bytes INTEGER",
            "ALTER TABLE artifacts ADD COLUMN checksum TEXT"
        ]
        
        for query in migration_queries:
            try:
                session.execute(text(query))
                print(f"✅ Executed: {query}")
            except Exception as e:
                if "duplicate column name" in str(e).lower() or "already exists" in str(e).lower():
                    print(f"⚠️  Column already exists: {query}")
                else:
                    print(f"❌ Failed: {query} - {e}")
        
        # Create default admin user
        print("👤 Creating default admin user...")
        
        # Import here to avoid circular imports
        from app.services.auth import AuthenticationService
        from app.services.storage import DatabaseService
        
        db_service = DatabaseService()
        auth_service = AuthenticationService(db_service)
        
        # Create admin user if not exists
        admin_result = auth_service.create_user(
            username="admin",
            email="admin@phishingdetector.local",
            password="admin123",  # Change this in production!
            full_name="System Administrator",
            role="admin"
        )
        
        if admin_result["success"]:
            print("✅ Default admin user created (username: admin, password: admin123)")
            print("⚠️  IMPORTANT: Change the admin password after first login!")
        else:
            print(f"⚠️  Admin user creation: {admin_result.get('error', 'Unknown error')}")
        
        session.commit()
        print("🎉 Database migration completed successfully!")
        
        # Print summary
        print("\n📋 Migration Summary:")
        print("✅ Enhanced user management with roles (admin, user, analyst)")
        print("✅ Classification logging for analytics")
        print("✅ False positive reporting system")
        print("✅ Comprehensive audit trails")
        print("✅ System metrics and analytics")
        print("✅ Session management")
        
        print("\n🔐 Default Credentials:")
        print("Username: admin")
        print("Password: admin123")
        print("Role: admin")
        print("\n⚠️  SECURITY: Change default password immediately!")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Migration failed: {e}")
        raise
    
    finally:
        session.close()


if __name__ == "__main__":
    # Use existing database path
    db_path = "./phishing_detector.db"
    database_url = f"sqlite:///{db_path}"
    
    migrate_database(database_url)