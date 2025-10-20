"""
Admin API endpoints for dataset management and model training.
"""
import asyncio
import json
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Form
from fastapi.responses import JSONResponse

from ..schemas.dto import (
    TrainRequestDTO, DatasetSummaryDTO, DatasetUploadResponseDTO, 
    RunDTO, ErrorResponseDTO, TrainingProgressDTO
)
from ..schemas.models import Dataset, Run, Artifact
from ..services.storage import DatabaseService, StorageService, validate_dataset_file
from ..services.training import TrainingService, ModelRegistry

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Global service instances (will be initialized in main.py)
db_service: DatabaseService = None
storage_service: StorageService = None
training_service: TrainingService = None
model_registry: ModelRegistry = None

logger = logging.getLogger(__name__)

def get_services():
    """Dependency to get service instances."""
    if not all([db_service, storage_service, training_service, model_registry]):
        raise HTTPException(status_code=500, detail="Services not initialized")
    return db_service, storage_service, training_service, model_registry


@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    notes: str = Form(None),
    services: tuple = Depends(get_services)
):
    """Upload and validate a new dataset."""
    db_svc, storage_svc, _, _ = services
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Save file temporarily for validation
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Validate the dataset file using the temp file path
            validation_result = validate_dataset_file(tmp_file_path)
            
            # Create dataset record
            dataset_id = db_svc.create_dataset(
                name=name,
                n_rows=validation_result["n_rows"],
                n_phish=validation_result["n_phish"],
                n_ham=validation_result["n_ham"],
                notes=notes
            )
            
            # Save the dataset file permanently
            file_path = storage_svc.save_dataset(dataset_id, file_content, file.filename)
            
            # Update dataset record with file path
            with db_svc.get_session() as session:
                dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
                if dataset:
                    dataset.file_path = file_path
                    session.commit()
            
            return {
                "success": True,
                "dataset_id": dataset_id,
                "message": f"Dataset '{name}' uploaded successfully",
                "stats": validation_result
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
        
    except Exception as e:
        logger.error(f"Dataset upload failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/datasets")
async def upload_dataset_alias(
    file: UploadFile = File(...),
    name: str = Form(...),
    notes: str = Form(None),
    services: tuple = Depends(get_services)
):
    """Upload and validate a new dataset (alias endpoint)."""
    # Call the main upload function
    return await upload_dataset(file, name, notes, services)


@router.get("/datasets", response_model=List[DatasetSummaryDTO])
async def get_datasets(services: tuple = Depends(get_services)):
    """Get all datasets."""
    db_svc, _, _, _ = services
    
    try:
        datasets = db_svc.get_all_datasets()
        return [DatasetSummaryDTO.from_orm(dataset) for dataset in datasets]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets/{dataset_id}/summary", response_model=DatasetSummaryDTO)
async def get_dataset_summary(
    dataset_id: str,
    services: tuple = Depends(get_services)
):
    """Get dataset summary."""
    db_svc, storage_svc, _, _ = services
    
    try:
        dataset = db_svc.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Add detailed statistics if needed
        if dataset.file_path:
            try:
                df = storage_svc.load_dataset(dataset.file_path)
                # Could add more detailed stats here
            except Exception:
                pass  # Use basic stats from database
        
        return DatasetSummaryDTO.from_orm(dataset)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def start_training(
    request: TrainRequestDTO,
    background_tasks: BackgroundTasks,
    services: tuple = Depends(get_services)
):
    """Start model training with preprocessing pipeline."""
    db_svc, storage_svc, training_svc, _ = services
    
    try:
        # Validate dataset exists
        dataset = db_svc.get_dataset(request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Create training run
        features_config = request.features.dict()
        hyperparams = request.hyperparams.dict()
        cv_config = request.cv.dict()
        
        # Extract preprocessing configurations from request
        dataset_format = getattr(request, 'dataset_format', 'simple')
        cleaning_config = getattr(request, 'cleaning_config', {})
        transform_config = getattr(request, 'transform_config', {})
        
        # Convert to dict if they're objects
        if hasattr(cleaning_config, 'dict'):
            cleaning_config = cleaning_config.dict()
        if hasattr(transform_config, 'dict'):
            transform_config = transform_config.dict()
        
        run_id = db_svc.create_run(
            dataset_id=request.dataset_id,
            model_type=request.model_type,
            features_config=features_config,
            hyperparams=hyperparams
        )
        
        # Start training in background with preprocessing parameters
        background_tasks.add_task(
            _train_model_background,
            run_id,
            dataset.file_path,
            features_config,
            request.model_type,
            hyperparams,
            cv_config,
            training_svc,
            dataset_format,
            cleaning_config,
            transform_config
        )
        
        return {"run_id": run_id, "status": "training_started"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


async def _train_model_background(
    run_id: str,
    dataset_path: str,
    features_config: Dict[str, Any],
    model_type: str,
    hyperparams: Dict[str, Any],
    cv_config: Dict[str, Any],
    training_service: TrainingService,
    dataset_format: str = "simple",
    cleaning_config: Dict[str, Any] = None,
    transform_config: Dict[str, Any] = None
):
    """Background task for model training with preprocessing."""
    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            training_service.train_model,
            run_id,
            dataset_path,
            features_config,
            model_type,
            hyperparams,
            cv_config,
            dataset_format,
            cleaning_config,
            transform_config
        )
    except Exception as e:
        print(f"Training failed for run {run_id}: {e}")


@router.get("/runs", response_model=List[RunDTO])
async def get_runs(services: tuple = Depends(get_services)):
    """Get all training runs."""
    db_svc, _, _, _ = services
    
    try:
        runs = db_svc.get_all_runs()
        run_dtos = []
        
        for run in runs:
            # Parse JSON fields safely
            try:
                features = json.loads(run.features_json) if run.features_json else {}
            except (json.JSONDecodeError, TypeError):
                features = {}
                
            try:
                hyperparams = json.loads(run.hyperparams_json) if run.hyperparams_json else {}
            except (json.JSONDecodeError, TypeError):
                hyperparams = {}
                
            try:
                raw_metrics = json.loads(run.metrics_json) if run.metrics_json else None
                # Transform metrics to match MetricsDTO schema
                metrics = _transform_metrics(raw_metrics) if raw_metrics else None
            except (json.JSONDecodeError, TypeError):
                metrics = None
            
            # Convert to DTO format and include total_classifications
            run_dto = {
                "id": run.id,
                "dataset_id": run.dataset_id,
                "model_type": run.model_type,
                "features": features,
                "hyperparams": hyperparams,
                "metrics": metrics,
                "created_at": run.created_at,
                "is_production": run.is_production,
                "status": run.status,
                "training_time": run.training_time,
                "total_classifications": getattr(run, 'total_classifications', 0) or 0
            }
            
            run_dtos.append(RunDTO(**run_dto))
        
        return run_dtos
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}", response_model=RunDTO)
async def get_run(run_id: str, services: tuple = Depends(get_services)):
    """Get a specific training run."""
    db_svc, _, _, _ = services
    
    try:
        run = db_svc.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Parse JSON fields safely
        try:
            features = json.loads(run.features_json) if run.features_json else {}
        except (json.JSONDecodeError, TypeError):
            features = {}
            
        try:
            hyperparams = json.loads(run.hyperparams_json) if run.hyperparams_json else {}
        except (json.JSONDecodeError, TypeError):
            hyperparams = {}
            
        try:
            raw_metrics = json.loads(run.metrics_json) if run.metrics_json else None
            # Transform metrics to match MetricsDTO schema
            metrics = _transform_metrics(raw_metrics) if raw_metrics else None
        except (json.JSONDecodeError, TypeError):
            metrics = None
        
        run_dto = {
            "id": run.id,
            "dataset_id": run.dataset_id,
            "model_type": run.model_type,
            "features": features,
            "hyperparams": hyperparams,
            "metrics": metrics,
            "created_at": run.created_at,
            "is_production": run.is_production,
            "status": run.status,
            "training_time": run.training_time,
            "total_classifications": getattr(run, 'total_classifications', 0) or 0
        }
        
        return RunDTO(**run_dto)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _transform_metrics(raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Transform raw metrics data to match MetricsDTO schema."""
    if not raw_metrics:
        return None
    
    # Extract mean values and transform field names
    transformed = {
        "accuracy": raw_metrics.get("accuracy_mean", 0.0),
        "precision": raw_metrics.get("precision_mean", 0.0),
        "recall": raw_metrics.get("recall_mean", 0.0),
        "f1_score": raw_metrics.get("f1_mean", 0.0),
        "roc_auc": None,  # Not available in current metrics
        "confusion_matrix": {},  # Not available in current metrics format
        "class_metrics": {
            "phish": {
                "precision": raw_metrics.get("precision_mean", 0.0),
                "recall": raw_metrics.get("recall_mean", 0.0),
                "f1_score": raw_metrics.get("f1_mean", 0.0)
            },
            "ham": {
                "precision": raw_metrics.get("precision_mean", 0.0),
                "recall": raw_metrics.get("recall_mean", 0.0),
                "f1_score": raw_metrics.get("f1_mean", 0.0)
            }
        }
    }
    
    return transformed


@router.post("/runs/{run_id}/promote")
async def promote_run(run_id: str, services: tuple = Depends(get_services)):
    """Promote a run to production."""
    db_svc, _, _, model_reg = services
    
    try:
        run = db_svc.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        if run.status != "completed":
            raise HTTPException(status_code=400, detail="Can only promote completed runs")
        
        # Promote run and clear model cache
        model_reg.promote_model_to_production(run_id)
        
        return {"message": f"Run {run_id} promoted to production"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}/status", response_model=TrainingProgressDTO)
async def get_training_status(run_id: str, services: tuple = Depends(get_services)):
    """Get training status and progress."""
    db_svc, _, _, _ = services
    
    try:
        run = db_svc.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Calculate progress based on status
        progress_map = {
            "pending": 0.0,
            "training": 0.5,
            "completed": 1.0,
            "failed": 0.0
        }
        
        # Calculate elapsed time
        from datetime import datetime
        elapsed_time = (datetime.utcnow() - run.created_at).total_seconds()
        
        # Estimate remaining time (simple heuristic)
        estimated_remaining = None
        if run.status == "training":
            # Rough estimate based on typical training times
            estimated_remaining = max(0, 120 - elapsed_time)  # Assume 2 minutes max
        
        current_step_map = {
            "pending": "Initializing",
            "training": "Training model",
            "completed": "Completed",
            "failed": "Failed"
        }
        
        return TrainingProgressDTO(
            run_id=run_id,
            status=run.status,
            progress=progress_map.get(run.status, 0.0),
            current_step=current_step_map.get(run.status, "Unknown"),
            elapsed_time=elapsed_time,
            estimated_remaining=estimated_remaining
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str, services: tuple = Depends(get_services)):
    """Delete a training run and its artifacts."""
    db_svc, storage_svc, _, model_reg = services
    
    try:
        run = db_svc.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        if run.is_production:
            raise HTTPException(status_code=400, detail="Cannot delete production run")
        
        # Delete artifacts from storage
        artifacts = db_svc.get_artifacts_by_run(run_id)
        for artifact in artifacts:
            try:
                import os
                if os.path.exists(artifact.path):
                    os.remove(artifact.path)
            except Exception:
                pass  # Continue even if artifact deletion fails
        
        # Delete from database
        with db_svc.get_session() as session:
            # Delete artifacts
            session.query(Artifact).filter(Artifact.run_id == run_id).delete()
            # Delete run
            session.query(Run).filter(Run.id == run_id).delete()
            session.commit()
        
        # Clear model cache
        model_reg.clear_cache()
        
        return {"message": f"Run {run_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))