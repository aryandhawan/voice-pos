# Module 2 Backend - AI Voice Ordering Copilot

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Start server
python main.py
```

## API Endpoints

### 1. Module 1 Integration
- `POST /api/internal/sync-revenue` - Receive revenue intelligence webhook
- `GET /api/internal/sync-status/{sync_id}` - Check sync status
- `GET /api/internal/health/module1-connection` - Integration health

### 2. Voice Processing
- `POST /voice/process` - Transcribe audio + intent mapping + upsell
  - Accepts: multipart/form-data with audio file
  - Returns: transcript, detected items, upsell suggestions

### 3. Order Confirmation
- `POST /order/confirm` - Finalize order, return Petpooja KOT JSON

### 4. Health
- `GET /health` - Service health check
- `GET /` - API info

## Architecture

```
Audio (any format)
    ↓
[Converter] → 16kHz mono WAV
    ↓
[Whisper] → Transcript
    ↓
[Vector Search] → Semantic intent match
    ↓
[Upsell Engine] → Hidden Star recommendations
    ↓
[Order Confirm] → Petpooja KOT JSON
```

## Module 1 Webhook Integration

Module 1 should POST to `/api/internal/sync-revenue` with:
- `X-API-Key` header matching MODULE1_WEBHOOK_SECRET
- Body: RevenueSyncPayload JSON

## Technology Stack

- FastAPI - Web framework
- ChromaDB - Vector database (local)
- sentence-transformers - Text embeddings (all-MiniLM-L6-v2)
- Whisper - Speech recognition (local)
- pydub - Audio conversion

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MODULE1_WEBHOOK_SECRET` | Shared secret for Module 1 auth |
| `CHROMA_DB_PATH` | Path to ChromaDB storage |
| `WHISPER_MODEL` | tiny/base/small |
| `MAX_AUDIO_SIZE_MB` | Audio upload limit |
| `RESTAURANT_ID` | Restaurant identifier |
