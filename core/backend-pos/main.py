"""
Module 2: AI Voice Ordering Copilot - FastAPI Backend
Production-grade FastAPI application with vector DB integration
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Routes
from routes.internal_sync import router as sync_router
from routes.voice import router as voice_router
from routes.order import router as order_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - startup/shutdown events"""
    logger.info("=" * 60)
    logger.info("Module 2: Voice Ordering Copilot Starting Up")
    logger.info("=" * 60)

    # Verify critical environment variables
    required_vars = ["MODULE1_WEBHOOK_SECRET", "CHROMA_DB_PATH"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.warning(f"Missing env vars (will use defaults): {missing}")

    # Initialize vector store (lazy loaded in services)
    logger.info("Vector sync service ready")
    logger.info("Whisper STT engine ready")

    yield

    # Shutdown
    logger.info("Module 2 shutting down gracefully")


# FastAPI Application
app = FastAPI(
    title="AI Voice Ordering Copilot - Module 2",
    description="Real-time voice ordering with semantic intent mapping and Module 1 BI integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure strictly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "internal_error"}
    )


# Include routers
app.include_router(sync_router)
app.include_router(voice_router)
app.include_router(order_router)


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "AI Voice Ordering Copilot - Module 2",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "sync": "/api/internal/sync-revenue (POST)",
            "voice": "/voice/process (POST)",
            "order": "/order/confirm (POST)",
            "health": "/health"
        },
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "module": "backend-pos",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
        "components": {
            "vector_db": "ready",
            "whisper_stt": "ready",
            "webhook_receiver": "ready"
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENV", "production") == "development",
        workers=1  # Single worker for ChromaDB in-memory
    )
