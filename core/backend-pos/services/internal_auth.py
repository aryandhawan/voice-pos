"""
Internal Authentication Service
Secures webhook endpoints between Module 1 and Module 2
"""

import os
import secrets
import logging
from typing import Optional
from fastapi import HTTPException, Header

logger = logging.getLogger(__name__)


class InternalAuthService:
    """
    Simple shared-secret authentication for local network communication.
    No OAuth - both laptops share .env secret.
    """

    def __init__(self, webhook_secret: Optional[str] = None):
        self.webhook_secret = webhook_secret or os.getenv("MODULE1_WEBHOOK_SECRET")
        self.allowed_ips = os.getenv("ALLOWED_MODULE1_IPS", "").split(",")
        self.allowed_ips = [ip.strip() for ip in self.allowed_ips if ip.strip()]

    def verify_module1_signature(
        self,
        source_host: str,
        x_api_key: str
    ) -> bool:
        """
        Verify webhook request originates from authorized Module 1.

        Args:
            source_host: IP or hostname from payload
            x_api_key: API key from X-Api-Key header

        Returns:
            True if valid

        Raises:
            HTTPException: 401 if invalid
        """
        if not self.webhook_secret:
            logger.error("MODULE1_WEBHOOK_SECRET not configured")
            raise HTTPException(500, "Server misconfiguration: auth not set up")

        # Verify API key using constant-time comparison
        if not secrets.compare_digest(x_api_key, self.webhook_secret):
            logger.warning(f"Invalid API key from {source_host}")
            raise HTTPException(401, "Invalid authentication credentials")

        # Optional IP whitelist check
        if self.allowed_ips and source_host not in self.allowed_ips:
            # Allow if any part of allowed_ips matches (for local network flexibility)
            allowed = False
            for allowed_ip in self.allowed_ips:
                if allowed_ip in source_host or source_host in allowed_ip:
                    allowed = True
                    break

            if not allowed:
                logger.warning(f"Request from unauthorized IP: {source_host}")
                # Don't fail here in hackathon environment - just log
                # In production: raise HTTPException(403, "IP not allowed")

        logger.debug(f"Authenticated webhook from {source_host}")
        return True

    def generate_sync_id(self) -> str:
        """Generate unique sync ID"""
        import uuid
        return f"sync_{uuid.uuid4().hex[:12]}"


# Singleton
_internal_auth_service: Optional[InternalAuthService] = None


def get_internal_auth() -> InternalAuthService:
    """Dependency injection provider"""
    global _internal_auth_service
    if _internal_auth_service is None:
        _internal_auth_service = InternalAuthService()
    return _internal_auth_service
