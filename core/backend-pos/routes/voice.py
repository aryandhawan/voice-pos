"""
Voice Execution Route - Audio to Intent + Upsell
POST /voice/process - Main endpoint for voice ordering
"""

import os
import time
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status, Depends

from schemas.voice_schemas import (
    VoiceProcessRequest,
    VoiceProcessResponse,
    TranscriptSegment,
    OrderItem,
    UpsellSuggestion,
    IntentMatch
)
from schemas.revenue_sync import MenuItemPayload
from stt.whisper_engine import get_whisper_engine, TranscriptionResult
from formats.converter import get_converter, AudioValidationError
from services.vector_sync import VectorSyncService, get_vector_sync_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/voice", tags=["voice-processing"])


@router.post(
    "/process",
    response_model=VoiceProcessResponse,
    summary="Process voice audio - Capture → Intent → Upsell",
    description="Accepts audio file, transcribes with Whisper, does semantic intent mapping, returns upsells"
)
async def process_voice(
    audio: UploadFile = File(..., description="Audio file (any format supported)"),
    session_id: Optional[str] = Form(None, description="Session for conversation continuity"),
    preferred_language: str = Form("en", description="en, hi, hinglish"),
    enable_upsell: bool = Form(True),
    vector_service: VectorSyncService = Depends(get_vector_sync_service)
):
    """
    Main voice processing pipeline: Capture → Intent → Upsell

    **Process:**
    1. Receives audio (any format)
    2. Converts to 16kHz mono WAV
    3. Transcribes with local Whisper
    4. Does semantic vector search for intent mapping
    5. Generates upsell suggestions from Hidden Stars

    **Returns:**
    - Parsed order items with confidence
    - Ambiguity flag if multiple similar items
    - Real-time upsell suggestions

    **Target latency:** <2 seconds total
    """

    start_time = time.time()
    request_id = f"req_{int(start_time * 1000)}"

    logger.info(f"[{request_id}] Voice request: {audio.filename}, size={audio.size if hasattr(audio, 'size') else 'unknown'}")

    try:
        # === STEP 1: Audio Conversion ===
        audio_bytes = await audio.read()

        if len(audio_bytes) < 100:
            raise HTTPException(400, "Audio file is empty")

        converter = get_converter(max_size_mb=25.0)
        wav_bytes = converter.validate_and_convert(audio_bytes, audio.filename)

        conversion_time = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] Audio conversion: {conversion_time:.1f}ms")

        # === STEP 2: Speech-to-Text ===
        whisper = get_whisper_engine()
        transcription = await whisper.transcribe(
            wav_bytes,
            language=preferred_language,
            task="transcribe"
        )

        if not transcription.text:
            raise HTTPException(400, "Could not transcribe audio - please try again")

        stt_time = (time.time() - start_time) * 1000
        logger.info(f"[{request_id}] STT completed: '{transcription.text[:50]}...' in {transcription.processing_time_ms}ms")

        # === STEP 3: Semantic Intent Mapping ===
        detected_items, ambiguity_flag, ambiguity_options = await _map_intent(
            transcript=transcription.text,
            vector_service=vector_service
        )

        intent_time = (time.time() - start_time) * 1000

        # === STEP 4: Upsell Generation ===
        upsell_suggestions = []
        if enable_upsell and detected_items:
            upsell_suggestions = await _generate_upsells(
                detected_item_ids=[item.item_id for item in detected_items],
                vector_service=vector_service
            )

        # === RESPONSE ===
        total_time = int((time.time() - start_time) * 1000)

        return VoiceProcessResponse(
            session_id=session_id or request_id,
            transcript=transcription.text,
            segments=[
                TranscriptSegment(
                    text=seg.get("text", ""),
                    start=seg.get("start", 0),
                    end=seg.get("end", 0),
                    confidence=10 ** seg.get("avg_logprob", -1)  # Convert logprob to prob
                )
                for seg in transcription.segments
            ],
            detected_items=detected_items,
            ambiguity_flag=ambiguity_flag,
            ambiguity_options=ambiguity_options if ambiguity_flag else None,
            upsell_suggestions=upsell_suggestions,
            processing_time_ms=total_time,
            timestamp=datetime.utcnow()
        )

    except AudioValidationError as e:
        logger.error(f"[{request_id}] Audio validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"[{request_id}] Processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")


async def _map_intent(
    transcript: str,
    vector_service: VectorSyncService
) -> tuple[List[OrderItem], bool, List[IntentMatch]]:
    """
    Map transcript to menu items via semantic search.
    Extract items, quantities, modifiers from natural language.
    """

    detected_items: List[OrderItem] = []
    ambiguity_flag = False
    ambiguity_options: List[IntentMatch] = []

    try:
        # Semantic search for menu items matching transcript
        # Query for top matches
        results = await vector_service.semantic_search(
            query_text=transcript,
            n_results=5
        )

        if not results:
            return [], False, []

        # Extract entities from transcript (simple NLP)
        quantities = _extract_quantity(transcript)
        modifiers = _extract_modifiers(transcript)

        # Take top match as primary item
        if results:
            top_match = results[0]
            metadata = top_match["metadata"]

            # Check for ambiguity (multiple similar items)
            if len(results) >= 2 and results[1]["confidence"] > 0.7:
                if results[0]["confidence"] - results[1]["confidence"] < 0.15:
                    ambiguity_flag = True
                    ambiguity_options = [
                        IntentMatch(
                            item_id=r["item_id"],
                            name=r["metadata"]["name"],
                            confidence=r["confidence"],
                            matched_text=r["document"]
                        )
                        for r in results[:3]
                    ]

            # Create order item
            order_item = OrderItem(
                item_id=top_match["item_id"],
                name=metadata["name"],
                quantity=quantities[0] if quantities else 1,
                modifiers=modifiers,
                unit_price=metadata.get("contribution_margin", 0) * 2,  # Estimated price
                intent_confidence=top_match["confidence"],
                match_reason=f"Semantic match: {top_match['document'][:50]}..."
            )

            detected_items.append(order_item)

    except Exception as e:
        logger.error(f"Intent mapping error: {e}")

    return detected_items, ambiguity_flag, ambiguity_options


async def _generate_upsells(
    detected_item_ids: List[str],
    vector_service: VectorSyncService
) -> List[UpsellSuggestion]:
    """Generate upsell recommendations from Hidden Stars"""

    upsells: List[UpsellSuggestion] = []

    try:
        recommendations = await vector_service.get_upsell_recommendations(
            item_ids=detected_item_ids,
            n_per_item=1
        )

        for rec in recommendations:
            metadata = rec.get("upsell_metadata", {})

            upsell = UpsellSuggestion(
                item_id=rec.get("upsell_item_id", ""),
                name=metadata.get("name", ""),
                description=metadata.get("document", "")[:100],
                reason=_generate_upsell_reason(metadata),
                margin_lift=metadata.get("contribution_margin", 0),
                confidence=rec.get("confidence", 0.5),
                original_trigger=rec.get("trigger_item", "")
            )

            upsells.append(upsell)

    except Exception as e:
        logger.error(f"Upsell generation error: {e}")

    return upsells[:2]  # Max 2 upsells


def _extract_quantity(transcript: str) -> List[int]:
    """Extract quantity numbers from transcript"""
    import re

    # Match number words and digits
    numbers = re.findall(r'\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b', transcript.lower())

    word_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }

    quantities = []
    for n in numbers:
        if n.isdigit():
            quantities.append(int(n))
        elif n in word_map:
            quantities.append(word_map[n])

    return quantities if quantities else [1]


def _extract_modifiers(transcript: str) -> List[str]:
    """Extract modifiers from transcript"""
    modifiers = []
    transcript_lower = transcript.lower()

    modifier_keywords = {
        "extra_cheese": ["extra cheese", "more cheese", "double cheese"],
        "no_onion": ["no onion", "without onion", "skip onion"],
        "no_garlic": ["no garlic", "without garlic"],
        "spicy": ["spicy", "extra spicy", "hot", "make it hot"],
        "less_spicy": ["less spicy", "mild", "not spicy"],
        "large": ["large", "big", "bigger"],
        "small": ["small", "regular"],
        "well_done": ["well done", "crispy", "extra crispy"],
    }

    for mod, keywords in modifier_keywords.items():
        if any(kw in transcript_lower for kw in keywords):
            modifiers.append(mod)

    return modifiers


def _generate_upsell_reason(metadata: dict) -> str:
    """Generate human-readable upsell reason"""

    if metadata.get("hidden_star"):
        return "Customer favorite that's perfect with your order - Hidden Star item"

    margin = metadata.get("contribution_margin", 0)
    if margin > 150:
        return "Premium add-on that complements your selection"

    return "Recommended pairing based on popular combinations"


@router.get("/health")
async def voice_health_check():
    """Voice pipeline health check"""
    return {
        "status": "healthy",
        "whisper_model": get_whisper_engine().model_size,
        "device": get_whisper_engine().device,
        "supported_formats": "webm,mp3,m4a,wav,ogg,flac,3gp,amr,aac,opus",
        "target_latency_ms": 2000
    }
