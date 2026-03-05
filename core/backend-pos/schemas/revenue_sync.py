"""
Revenue Sync Schemas - Module 1 to Module 2 Integration
Handles incoming webhook payloads from Revenue Dashboard
"""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class ComboRulePayload(BaseModel):
    """Association rules for upsell recommendations from Module 1 Combo Engine"""
    trigger_item_id: str = Field(..., description="ID of item that triggers recommendation")
    recommendation_id: str = Field(..., description="ID of item to recommend")
    lift_margin: float = Field(..., ge=0, description="Expected margin lift from this combo")
    confidence_score: float = Field(..., ge=0, le=1, description="Association rule confidence")

    class Config:
        json_schema_extra = {
            "example": {
                "trigger_item_id": "pizza_margherita_001",
                "recommendation_id": "garlic_bread_cheese_001",
                "lift_margin": 45.0,
                "confidence_score": 0.85
            }
        }


class MenuItemPayload(BaseModel):
    """Single menu item with business intelligence from Module 1"""
    item_id: str = Field(..., min_length=1, description="Unique SKU or UUID")
    name: str = Field(..., min_length=1, description="Display name")
    description: str = Field(default="", description="Menu description for embedding")
    category: str = Field(..., description="Menu category: pizza, beverage, dessert, etc.")
    contribution_margin: float = Field(..., ge=0, description="Absolute margin in local currency")
    margin_percent: float = Field(..., ge=0, le=100, description="Margin percentage")
    risk_flag: bool = Field(default=False, description="Low-margin high-volume risk flag")
    hidden_star: bool = Field(default=False, description="High-margin under-promoted flag")
    popularity_score: float = Field(default=0.5, ge=0, le=1, description="Normalized popularity")
    avg_order_frequency: float = Field(default=0.0, ge=0, description="Daily order frequency")
    seasonal_tag: Optional[str] = Field(default=None, description="e.g., 'summer_special', 'winter_warm'")
    upsell_tags: List[str] = Field(default_factory=list, description="Tags: ['premium', 'shareable', 'bestseller']")
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @validator('name')
    def name_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty or whitespace')
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": "pizza_truffle_001",
                "name": "Truffle Mushroom Pizza",
                "description": "Wood-fired with truffle oil, mozzarella, and wild mushrooms",
                "category": "pizza",
                "contribution_margin": 180.50,
                "margin_percent": 68.5,
                "risk_flag": False,
                "hidden_star": True,
                "popularity_score": 0.35,
                "avg_order_frequency": 5.2,
                "seasonal_tag": "monsoon_special",
                "upsell_tags": ["premium", "shareable", "vegetarian"],
                "last_updated": "2024-03-05T14:30:00Z"
            }
        }


class RevenueSyncPayload(BaseModel):
    """Complete sync payload from Module 1 Revenue Dashboard"""
    sync_id: str = Field(..., description="UUID for this sync batch")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_host: str = Field(..., description="Module 1 IP/hostname")
    menu_items: List[MenuItemPayload]
    combo_rules: Optional[List[ComboRulePayload]] = Field(default=None)
    deleted_items: Optional[List[str]] = Field(default=None, description="Item IDs to remove")

    @validator('menu_items')
    def validate_non_empty(cls, v):
        if len(v) == 0:
            raise ValueError('Menu items list cannot be empty')
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "sync_id": "sync_20240305_143000_abc123",
                "timestamp": "2024-03-05T14:30:00Z",
                "source_host": "192.168.1.45",
                "menu_items": [],
                "combo_rules": [],
                "deleted_items": []
            }
        }


class SyncStatusResponse(BaseModel):
    """Response for sync status polling"""
    sync_id: str
    status: str = Field(..., regex="^(pending|processing|completed|failed)$")
    items_processed: int
    items_total: int
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None


class SyncAcceptedResponse(BaseModel):
    """Immediate 202 response for async processing"""
    status: str = "accepted"
    sync_id: str
    items_queued: int
    estimated_processing_ms: int
