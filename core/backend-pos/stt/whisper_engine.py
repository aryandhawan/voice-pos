"""
Whisper STT Engine - Local Speech-to-Text
Optimized for low-latency voice ordering
"""

import os
import io
import logging
import tempfile
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Structured Whisper output"""
    text: str
    segments: List[Dict[str, Any]]
    language: str
    confidence: float
    processing_time_ms: int


class WhisperEngine:
    """
    Local Whisper inference with caching and async support.
    Optimized for ~500ms transcription on CPU.
    """

    DEFAULT_MODEL = "base"  # Options: tiny, base, small, medium
    SUPPORTED_LANGUAGES = ["en", "hi", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "zh"]

    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None):
        self.model_size = model_size or os.getenv("WHISPER_MODEL", self.DEFAULT_MODEL)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self._load_model()

    def _load_model(self):
        """Lazy load Whisper model"""
        try:
            import whisper

            logger.info(f"Loading Whisper model: {self.model_size} on {self.device}")
            self.model = whisper.load_model(self.model_size).to(self.device)

            # Warm-up inference (JIT compilation)
            dummy = np.zeros(16000, dtype=np.float32)
            self.model.transcribe(dummy, language="en", fp16=(self.device == "cuda"))

            logger.info(f"Whisper model loaded and warmed up")

        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str] = "en",
        task: str = "transcribe",
        prompt: Optional[str] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio to text (async wrapper for blocking Whisper).

        Args:
            audio_bytes: 16kHz mono WAV audio
            language: ISO 639-1 code (en, hi, etc.)
            task: 'transcribe' or 'translate'
            prompt: Optional context prompt

        Returns:
            TranscriptionResult with text and timing
        """
        import time
        start_time = time.time()

        # Run in threadpool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._sync_transcribe,
            audio_bytes,
            language,
            task,
            prompt
        )

        processing_time = int((time.time() - start_time) * 1000)

        return TranscriptionResult(
            text=result["text"].strip(),
            segments=result.get("segments", []),
            language=result.get("language", language or "en"),
            confidence=self._calculate_confidence(result),
            processing_time_ms=processing_time
        )

    def _sync_transcribe(
        self,
        audio_bytes: bytes,
        language: Optional[str],
        task: str,
        prompt: Optional[str]
    ) -> Dict:
        """Synchronous Whisper inference"""

        # Convert bytes to numpy array
        audio_np = self._bytes_to_numpy(audio_bytes)

        # Decode options
        decode_options = {
            "task": task,
            "fp16": (self.device == "cuda"),
            "language": language,
            "without_timestamps": False,
        }

        if prompt:
            decode_options["initial_prompt"] = prompt

        # Run inference
        result = self.model.transcribe(audio_np, **decode_options)

        return result

    def _bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        """Convert WAV bytes to numpy float32 array [-1.0, 1.0]"""
        import wave

        with io.BytesIO(audio_bytes) as wav_io:
            with wave.open(wav_io, 'rb') as wav_file:
                # Read raw PCM data
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                n_frames = wav_file.getnframes()

                raw_data = wav_file.readframes(n_frames)

                # Convert to numpy based on bit depth
                if sample_width == 2:  # 16-bit
                    audio_np = np.frombuffer(raw_data, dtype=np.int16)
                elif sample_width == 4:  # 32-bit
                    audio_np = np.frombuffer(raw_data, dtype=np.int32)
                else:
                    audio_np = np.frombuffer(raw_data, dtype=np.int16)

                # Normalize to float32 [-1.0, 1.0]
                audio_np = audio_np.astype(np.float32) / 32768.0

                # Ensure mono (average channels)
                if n_channels > 1:
                    audio_np = audio_np.reshape(-1, n_channels).mean(axis=1)

                return audio_np

    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate average confidence from segment-level probabilities"""
        segments = result.get("segments", [])
        if not segments:
            return 0.0

        # Average of avg_log_probs (higher is better, Whisper uses logprobs)
        avg_probs = [seg.get("avg_logprob", -1.0) for seg in segments]
        mean_prob = sum(avg_probs) / len(avg_probs)

        # Convert logprob to confidence [0, 1]
        # -1.0 is good, -2.0 is borderline
        confidence = min(1.0, max(0.0, (mean_prob + 2.0) / 2.0))

        return round(confidence, 3)

    def detect_language(self, audio_bytes: bytes) -> str:
        """Detect spoken language (first 30s)"""
        audio_np = self._bytes_to_numpy(audio_bytes)

        # Truncate to 30s for speed
        max_samples = 30 * 16000
        audio_np = audio_np[:max_samples]

        _, probs = self.model.detect_language(audio_np)
        detected = max(probs, key=probs.get)

        return detected


# Singleton instance
_whisper_instance: Optional[WhisperEngine] = None


def get_whisper_engine() -> WhisperEngine:
    """Get or create WhisperEngine singleton"""
    global _whisper_instance
    if _whisper_instance is None:
        _whisper_instance = WhisperEngine()
    return _whisper_instance
