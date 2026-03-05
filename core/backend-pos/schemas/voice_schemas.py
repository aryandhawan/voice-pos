"""
Voice Processing Schemas
Handles audio processing and semantic intent mapping
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TranscriptSegment(BaseModel):
    """Whisper transcription segment with timing"""
    text: str
    start: float
    end: float
    confidence: float = Field(ge=0, le=1)


class IntentMatch(BaseModel):
    """Single semantic match from vector search"""
    item_id: str
    name: str
    confidence: float = Field(ge=0, le=1)
    matched_text: str = Field(..., description="What text in transcript matched")


class OrderItem(BaseModel):
    """Parsed order item from voice transcript"""
    item_id: str
    name: str
    quantity: int = Field(ge=1)
    modifiers: List[str] = Field(default_factory=list, description="e.g., ['extra_cheese', 'no_onion']")
    unit_price: float = Field(ge=0)
    intent_confidence: float = Field(ge=0, le=1, description="ML confidence in intent match")
    match_reason: str = Field(..., description="Why this item was selected")


class UpsellSuggestion(BaseModel):
    """Generated upsell from Hidden Stars or Combo Engine"""
    item_id: str
    name: str
    description: str
    reason: str = Field(..., description="Why suggest this: 'Pairs with pizza', 'High margin alternative'")
    margin_lift: float = Field(ge=0, description="Expected additional margin")
    confidence: float = Field(ge=0, le=1)
    original_trigger: str = Field(..., description="Which cart item triggered this")


class VoiceProcessRequest(BaseModel):
    """Request for /voice/process"""
    # Audio file uploaded as multipart/form-data
    session_id: Optional[str] = Field(default=None, description="For conversation continuity")
    preferred_language: str = Field(default="en", description="en, hi, hinglish")
    enable_upsell: bool = Field(default=True)


class VoiceProcessResponse(BaseModel):
    """Response from /voice/process with intent + upsell"""
    session_id: str
    transcript: str
    segments: List[TranscriptSegment]
    detected_items: List[OrderItem]
    ambiguity_flag: bool = Field(default=False, description="True if multiple similar items found")
    ambiguity_options: Optional[List[IntentMatch]] = None
    upsell_suggestions: List[UpsellSuggestion] = Field(default_factory=list)
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "sess_abc123",
                "transcript": "I want a large pepperoni pizza with extra cheese",
                "detected_items": [{
                    "item_id": "pizza_pepperoni_lg",
                    "name": "Large Pepperoni Pizza",
                    "quantity": 1,
                    "modifiers": ["extra_cheese"],
                    "unit_price": 450.0,
                    "intent_confidence": 0.94
                }],
                "upsell_suggestions": [{
                    "item_id": "garlic_bread_001",
                    "name": "Cheesy Garlic Bread",
                    "reason": "Perfect pairing with pizza - Hidden Star item",
                    "margin_lift": 85.0
                }]
            }
        }


class CartItem(BaseModel):
    """Item in finalized cart for confirmation"""
    item_id: str
    name: str
    quantity: int
    modifiers: List[str]
    unit_price: float
    line_total: float
    original_confidence: float


class OrderConfirmRequest(BaseModel):
    """Request for POST /order/confirm - Finalized cart"""
    session_id: str
    items: List[CartItem]
    accepted_upsells: List[str] = Field(default_factory=list, description="Item IDs of accepted upsells")
    table_number: Optional[str] = Field(default=None)
    customer_phone: Optional[str] = Field(default=None, description="For order tracking")
    special_instructions: Optional[str] = Field(default=None)
    order_type: str = Field(default="dine_in", regex="^(dine_in|takeaway|delivery)$")


class PetpoojaItem(BaseModel):
    """Petpooja-compatible KOT item structure"""
    ItemID: str = Field(..., alias="item_id")
    ItemName: str = Field(..., alias="name")
    Quantity: int = Field(..., alias="quantity")
    Rate: float = Field(..., alias="unit_price")
    Amount: float = Field(..., alias="line_total")
    ItemDiscount: float = Field(default=0.0)
    CommentOnItem: str = Field(default="", alias="modifiers_text")
    IsUpsellItem: bool = Field(default=False)


class PetpoojaPayload(BaseModel):
    """Strict Petpooja KOT/Order JSON structure"""
    RestaurantID: str = Field(..., alias="restaurant_id")
    OrderID: str = Field(..., alias="order_id")
    OrderDate: str = Field(..., alias="order_date")  # ISO format
    OrderType: str = Field(..., alias="order_type")
    TableNumber: str = Field(default="", alias="table_number")
    CustomerPhone: str = Field(default="", alias="customer_phone")
    Items: List[PetpoojaItem]

    # Financials
    SubTotal: float = Field(..., alias="subtotal")
    Discount: float = Field(default=0.0, alias="discount")
    TaxAmount: float = Field(default=0.0, alias="tax_amount")
    TotalAmount: float = Field(..., alias="total_amount")

    # Business intelligence from Module 1
    EstimatedMargin: float = Field(..., alias="estimated_margin")
    MarginRiskFlag: bool = Field(default=False, alias="margin_risk_flag")
    HiddenStarItems: List[str] = Field(default_factory=list, alias="hidden_star_items")

    # Metadata
    Source: str = Field(default="voice_copilot", alias="source")
    SessionID: str = Field(..., alias="session_id")
    SpecialInstructions: str = Field(default="", alias="special_instructions")
    CreatedAt: str = Field(..., alias="created_at")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "RestaurantID": "REST001",
                "OrderID": "KOT240305001",
                "OrderDate": "2024-03-05T15:30:00+05:30",
                "OrderType": "Dine In",
                "TableNumber": "T12",
                "Items": [],
                "SubTotal": 895.0,
                "TaxAmount": 44.75,
                "TotalAmount": 939.75,
                "EstimatedMargin": 320.50,
                "Source": "voice_copilot"
            }
        }


class OrderConfirmResponse(BaseModel):
    """Response from POST /order/confirm"""
    success: bool
    order_id: str
    kot_number: str = Field(..., description="Kitchen Order Ticket number")
    pos_payload: PetpoojaPayload = Field(..., description="Full Petpooja-compatible payload")
    estimated_preparation_time: int = Field(..., ge=0, description="Minutes")
    margin_summary: Dict[str, Any] = Field(..., description="Margin breakdown from Module 1 logic")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
