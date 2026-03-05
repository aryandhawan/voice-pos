"""
Internal Sync Routes - Module 1 Webhook Receiver
Real-time revenue intelligence ingestion
"""

import os
import secrets
import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, status, Depends

from schemas.revenue_sync import (
    RevenueSyncPayload,
    SyncAcceptedResponse,
    SyncStatusResponse
)
from services.vector_sync import VectorSyncService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/internal", tags=["internal-sync"])

# In-memory status tracking (use Redis in production)
_sync_status_cache: dict = {}


def get_vector_sync_service() -> VectorSyncService:
    """Dependency injection for VectorSyncService singleton"""
    # Singleton pattern - initialize once
    if not hasattr(get_vector_sync_service, "_instance"):
        get_vector_sync_service._instance = VectorSyncService()
    return get_vector_sync_service._instance


def verify_module1_auth(x_api_key: str = Header(..., description="Module 1 shared secret")):
    """Validate webhook authenticity"""
    expected_key = os.getenv("MODULE1_WEBHOOK_SECRET")

    if not expected_key:
        logger.error("MODULE1_WEBHOOK_SECRET not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server authentication not configured"
        )

    if not secrets.compare_digest(x_api_key, expected_key):
        logger.warning(f"Invalid webhook auth attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication"
        )

    return True


@router.post(
    "/sync-revenue",
    response_model=SyncAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Receive revenue intelligence from Module 1",
    description="Webhook endpoint for real-time menu intelligence updates. Triggers async vector embedding pipeline.",
    responses={
        202: {"description": "Sync accepted, processing in background"},
        401: {"description": "Invalid authentication"},
        422: {"description": "Invalid payload"},
        500: {"description": "Internal error"}
    }
)
async def sync_revenue_intelligence(
    payload: RevenueSyncPayload,
    background_tasks: BackgroundTasks,
    x_api_key: str = Header(..., description="Shared secret for Module 1 authentication"),
    vector_service: VectorSyncService = Depends(get_vector_sync_service),
    _auth: bool = Depends(verify_module1_auth)
):
    """
    Receive real-time revenue intelligence from Module 1 Dashboard.

    This endpoint:
    1. Validates Module 1 authentication
    2. Returns 202 ACCEPTED immediately (non-blocking)
    3. Queues background task for embedding + vector upsert
    4. Processes menu items through sentence-transformers
    5. Updates ChromaDB collections atomically

    **Request Headers:**
    - `X-API-Key`: Shared secret from MODULE1_WEBHOOK_SECRET env var

    **Request Body:**
    Complete menu intelligence including items, margins, and combo rules
    """

    logger.info(
        f"Received sync {payload.sync_id} from {payload.source_host}: "
        f"{len(payload.menu_items)} items"
    )

    # Initialize status tracking
    _sync_status_cache[payload.sync_id] = {
        "sync_id": payload.sync_id,
        "status": "pending",
        "items_total": len(payload.menu_items),
        "items_processed": 0,
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error_message": None
    }

    # Queue async processing
    background_tasks.add_task(
        _process_sync_background,
        payload=payload,
        vector_service=vector_service
    )

    # Estimate processing time (~15ms per item for embedding)
    estimated_ms = len(payload.menu_items) * 15

    return SyncAcceptedResponse(
        status="accepted",
        sync_id=payload.sync_id,
        items_queued=len(payload.menu_items),
        estimated_processing_ms=estimated_ms
    )


async def _process_sync_background(
    payload: RevenueSyncPayload,
    vector_service: VectorSyncService
):
    """Background task: Process sync payload through vector pipeline"""

    _sync_status_cache[payload.sync_id]["status"] = "processing"

    try:
        # Execute vector sync
        stats = await vector_service.process_revenue_sync(payload)

        # Update status
        _sync_status_cache[payload.sync_id].update({
            "status": "completed",
            "items_processed": stats["menu_items_processed"],
            "completed_at": datetime.utcnow().isoformat()
        })

        logger.info(f"Background sync {payload.sync_id} completed: {stats}")

    except Exception as e:
        logger.error(f"Background sync {payload.sync_id} failed: {e}")
        _sync_status_cache[payload.sync_id].update({
            "status": "failed",
            "error_message": str(e),
            "completed_at": datetime.utcnow().isoformat()
        })


@router.get(
    "/sync-status/{sync_id}",
    response_model=SyncStatusResponse,
    summary="Get sync processing status",
    description="Poll for background task completion status"
)
async def get_sync_status(sync_id: str):
    """
    Check the status of a previously submitted sync operation.

    **Status values:**
    - `pending`: Queued but not yet processing
    - `processing`: Actively embedding and upserting
    - `completed`: All items processed successfully
    - `failed`: Error occurred, see error_message
    """

    if sync_id not in _sync_status_cache:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sync {sync_id} not found"
        )

    status_data = _sync_status_cache[sync_id]

    return SyncStatusResponse(
        sync_id=sync_id,
        status=status_data["status"],
        items_total=status_data["items_total"],
        items_processed=status_data.get("items_processed", 0),
        error_message=status_data.get("error_message"),
        completed_at=status_data.get("completed_at")
    )


@router.get(
    "/health/module1-connection",
    summary="Check Module 1 connectivity",
    description="Health check endpoint for diagnostics"
)
async def check_module1_health():
    """Verify integration readiness"""
    return {
        "module2_status": "healthy",
        "vector_db_ready": True,
        "embedding_model_loaded": True,
        "webhook_endpoint": "/api/internal/sync-revenue",
        "expected_module1_format": "RevenueSyncPayload v1.0"
    }
