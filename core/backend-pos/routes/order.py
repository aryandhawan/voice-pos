"""
Order Confirmation Route - Final Cart → PoS Push
POST /order/confirm - Returns Petpooja-compatible KOT JSON
"""

import uuid
import time
import logging
from typing import List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field

from schemas.voice_schemas import (
    OrderConfirmRequest,
    OrderConfirmResponse,
    CartItem,
    PetpoojaPayload,
    PetpoojaItem
)
from services.vector_sync import VectorSyncService, get_vector_sync_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/order", tags=["order-confirmation"])


class PetpoojaPayloadStrict(BaseModel):
    """
    Strict Petpooja KOT/Order JSON structure.
    Follows Petpooja API v2.0 specification for Kitchen Order Ticket.
    """

    # Identification
    RestaurantID: str = Field(..., min_length=1)
    OrderID: str = Field(..., min_length=1, description="KOT number")
    BillNo: str = Field(default="", description="Optional bill number")

    # Timestamps
    OrderDate: str = Field(..., description="ISO 8601 with timezone")
    OrderDateTime: int = Field(..., description="Unix timestamp milliseconds")

    # Order Type
    OrderType: str = Field(..., regex="^(Dine In|Take Away|Delivery|Room Service)$")

    # Location
    TableNumber: str = Field(default="")
    SectionName: str = Field(default="Main Dining")
    RoomNumber: str = Field(default="")

    # Customer
    CustomerName: str = Field(default="Voice Customer")
    CustomerMobile: str = Field(default="", description="Customer phone")
    CustomerID: str = Field(default="", description="Customer loyalty ID")

    # Staff
    StewardName: str = Field(default="Voice Copilot")
    StewardCode: str = Field(default="VC001")

    # Items Array (Core)
    Items: List[Dict[str, Any]] = Field(..., min_items=1)

    # Financial Summary
    SubTotal: float = Field(..., ge=0, description="Before tax/discount")
    Discount: float = Field(default=0.0, ge=0)
    DiscountPercentage: float = Field(default=0.0, ge=0, le=100)
    TaxAmount: float = Field(default=0.0, ge=0)
    TaxPercentage: float = Field(default=0.0, ge=0)
    ServiceCharge: float = Field(default=0.0, ge=0)
    PackingCharge: float = Field(default=0.0, ge=0)
    DeliveryCharge: float = Field(default=0.0, ge=0)
    RoundOff: float = Field(default=0.0)
    TotalAmount: float = Field(..., ge=0, description="Final payable")

    # Payment
    PaymentMode: str = Field(default="Pending", description="Cash/Card/UPI/Pending")

    # Status
    OrderStatus: str = Field(default="KOT", description="KOT/Settlement/Settled")
    KOTStatus: str = Field(default="Printed", description="Printed/Not Printed")

    # Special Instructions
    SpecialDiscountRemarks: str = Field(default="")
    SpecialInstructions: str = Field(default="", alias="Instructions")

    # Source Tracking
    Source: str = Field(default="Voice Copilot", description="Originating system")
    OrderSource: str = Field(default="Voice Copilot", description="Order channel")

    # Business Intelligence (Module 1 Integration)
    EstimatedMargin: float = Field(default=0.0, ge=0)
    MarginRiskFlag: bool = Field(default=False, description="Low margin warning")
    HiddenStarItems: List[str] = Field(default_factory=list, description="High-margin items included")
    ContributionMargin: float = Field(default=0.0, ge=0, description="Total order margin")

    # Metadata
    SessionID: str = Field(default="")
    CreatedAt: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


@router.post(
    "/confirm",
    response_model=OrderConfirmResponse,
    summary="Confirm order and generate PoS payload",
    description="Finalizes cart, validates margins, returns Petpooja-compatible KOT JSON"
)
async def confirm_order(
    request: OrderConfirmRequest,
    vector_service: VectorSyncService = Depends(get_vector_sync_service)
):
    """
    Finalize order and generate Petpooja-compatible KOT payload.

    **Process:**
    1. Validates cart items against current menu (from ChromaDB)
    2. Calculates totals, taxes, margins
    3. Checks margin risk flags from Module 1
    4. Generates strict Petpooja JSON

    **Returns:**
    - Petpooja-ready KOT payload
    - Margin summary from Module 1
    - Preparation time estimate
    """

    start_time = time.time()
    order_id = f"KOT{datetime.now().strftime('%y%m%d%H%M%S')}"
    kot_number = f"VC{int(time.time()) % 10000}"

    logger.info(f"Confirming order {order_id}: {len(request.items)} items")

    try:
        # === STEP 1: Validate Items and Calculate ===
        validated_items, pos_items = await _validate_and_build_items(
            cart_items=request.items,
            accepted_upsells=request.accepted_upsells,
            vector_service=vector_service
        )

        if not validated_items:
            raise HTTPException(400, "No valid items in cart")

        # === STEP 2: Calculate Financials ===
        financials = _calculate_financials(
            items=validated_items,
            accepted_upsells=request.accepted_upsells
        )

        # === STEP 3: Calculate Margin (Module 1 Intelligence) ===
        margin_summary = _calculate_margin_summary(
            items=validated_items,
            vector_service=vector_service
        )

        # === STEP 4: Build Petpooja Payload ===
        now = datetime.utcnow()

        payload = PetpoojaPayloadStrict(
            RestaurantID=os.getenv("RESTAURANT_ID", "REST001"),
            OrderID=kot_number,
            OrderDate=now.isoformat(),
            OrderDateTime=int(now.timestamp() * 1000),
            OrderType=_map_order_type(request.order_type),
            TableNumber=request.table_number or "",
            CustomerName="Voice Customer",
            CustomerMobile=request.customer_phone or "",
            StewardName="Voice Copilot",
            StewardCode="VC001",
            Items=pos_items,
            SubTotal=financials["subtotal"],
            Discount=financials["discount"],
            TaxAmount=financials["tax"],
            TotalAmount=financials["total"],
            PaymentMode="Pending",
            OrderStatus="KOT",
            KOTStatus="Printed",
            SpecialInstructions=request.special_instructions or "",
            Source="Voice Copilot",
            OrderSource="Voice Copilot",
            EstimatedMargin=margin_summary["estimated_margin"],
            MarginRiskFlag=margin_summary["risk_flag"],
            HiddenStarItems=margin_summary["hidden_stars_included"],
            ContributionMargin=margin_summary["contribution_margin"],
            SessionID=request.session_id,
            CreatedAt=now.isoformat()
        )

        # === STEP 5: Preparation Time Estimate ===
        prep_time = _estimate_prep_time(validated_items)

        logger.info(f"Order {order_id} confirmed: ₹{payload.TotalAmount:.2f}, margin: ₹{payload.EstimatedMargin:.2f}")

        return OrderConfirmResponse(
            success=True,
            order_id=order_id,
            kot_number=kot_number,
            pos_payload=payload.dict(),  # Convert Pydantic to dict
            estimated_preparation_time=prep_time,
            margin_summary=margin_summary,
            timestamp=datetime.utcnow()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Order confirmation error: {e}", exc_info=True)
        raise HTTPException(500, "Order processing failed")


async def _validate_and_build_items(
    cart_items: List[CartItem],
    accepted_upsells: List[str],
    vector_service: VectorSyncService
) -> tuple[List[Dict], List[Dict]]:
    """
    Validate items against current menu and build POS format
    Returns: (validated_items, pos_formatted_items)
    """

    validated = []
    pos_formatted = []

    for item in cart_items:
        # Get current pricing from vector DB
        try:
            results = vector_service.menu_collection.get(
                ids=[item.item_id],
                include=["metadatas"]
            )

            if not results["ids"]:
                logger.warning(f"Item not found in menu: {item.item_id}")
                continue

            metadata = results["metadatas"][0]

            # Use actual price from menu
            current_price = metadata.get("contribution_margin", item.unit_price) * 2.5  # Price = margin * markup
            line_total = current_price * item.quantity

            # Build POS item format (Petpooja compatible)
            modifiers_text = ", ".join(item.modifiers) if item.modifiers else ""
            is_upsell = item.item_id in accepted_upsells

            pos_item = {
                "ItemID": item.item_id,
                "ItemName": item.name,
                "Quantity": item.quantity,
                "Rate": round(current_price, 2),
                "Amount": round(line_total, 2),
                "ItemDiscount": 0.0,
                "DiscountPercentage": 0.0,
                "TaxPercentage": 5.0,  # GST
                "TaxAmount": round(line_total * 0.05, 2),
                "CommentOnItem": modifiers_text,
                "IsUpsellItem": is_upsell,
                "CategoryName": metadata.get("category", "General"),
                "IsComplimentary": False,
                "ItemStatus": "Confirmed"
            }

            validated.append({
                "item_id": item.item_id,
                "name": item.name,
                "quantity": item.quantity,
                "unit_price": current_price,
                "line_total": line_total,
                "modifiers": item.modifiers,
                "is_upsell": is_upsell,
                "metadata": metadata
            })

            pos_formatted.append(pos_item)

        except Exception as e:
            logger.error(f"Error validating item {item.item_id}: {e}")
            continue

    return validated, pos_formatted


def _calculate_financials(items: List[Dict], accepted_upsells: List[str]) -> Dict[str, float]:
    """Calculate order financials"""

    subtotal = sum(item["line_total"] for item in items)

    # Apply small discount for upsell acceptance (incentive)
    discount = 0.0
    if accepted_upsells:
        discount = subtotal * 0.05  # 5% for accepting upsell

    taxable = subtotal - discount
    tax = taxable * 0.05  # 5% GST

    total = taxable + tax
    round_off = round(total) - total
    total = round(total)

    return {
        "subtotal": round(subtotal, 2),
        "discount": round(discount, 2),
        "tax": round(tax, 2),
        "total": round(total, 2),
        "round_off": round(round_off, 2)
    }


def _calculate_margin_summary(
    items: List[Dict],
    vector_service: VectorSyncService
) -> Dict[str, Any]:
    """Calculate margin intelligence from Module 1 data"""

    total_margin = 0.0
    risk_flag = False
    hidden_stars = []

    for item in items:
        metadata = item.get("metadata", {})
        margin = metadata.get("contribution_margin", 0) * item["quantity"]
        total_margin += margin

        if metadata.get("risk_flag"):
            risk_flag = True

        if metadata.get("hidden_star"):
            hidden_stars.append(item["name"])

    # Margin risk: if total margin < 15% of subtotal
    subtotal = sum(item["line_total"] for item in items)
    margin_percent = (total_margin / subtotal * 100) if subtotal > 0 else 0

    if margin_percent < 15:
        risk_flag = True

    return {
        "contribution_margin": round(total_margin, 2),
        "margin_percentage": round(margin_percent, 2),
        "estimated_margin": round(total_margin, 2),  # Same for now
        "risk_flag": risk_flag,
        "risk_reason": "Low margin percentage" if risk_flag else None,
        "hidden_stars_included": hidden_stars,
        "total_items": len(items),
        "hidden_star_count": len(hidden_stars)
    }


def _map_order_type(order_type: str) -> str:
    """Map internal order type to Petpooja format"""
    mapping = {
        "dine_in": "Dine In",
        "takeaway": "Take Away",
        "delivery": "Delivery"
    }
    return mapping.get(order_type.lower(), "Dine In")


def _estimate_prep_time(items: List[Dict]) -> int:
    """Estimate preparation time in minutes"""

    base_time = 15  # Base kitchen time

    for item in items:
        category = item.get("metadata", {}).get("category", "").lower()

        if "pizza" in category:
            base_time = max(base_time, 20)
        elif "burger" in category or "sandwich" in category:
            base_time = max(base_time, 12)
        elif "beverage" in category or "drink" in category:
            base_time = max(base_time, 5)
        elif "dessert" in category:
            base_time = max(base_time, 10)

    # Add time for quantity
    total_qty = sum(item["quantity"] for item in items)
    additional_time = (total_qty // 3) * 5  # +5 min per 3 items

    return min(base_time + additional_time, 45)  # Cap at 45 min


@router.get("/kot/{kot_number}")
async def get_kot(kot_number: str):
    """Retrieve generated KOT (mock - in production, query DB)"""
    return {
        "kot_number": kot_number,
        "status": "sent_to_kitchen",
        "message": "KOT retrieved from queue"
    }


import os
