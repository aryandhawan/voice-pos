"""
Audio Format Converter - Universal Audio Pipeline
All formats → 16kHz mono WAV for Whisper consumption
"""

import io
import magic
import logging
from typing import Optional, Dict, List
from enum import Enum

import ffmpeg
from pydub import AudioSegment
from pydub.effects import normalize

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats"""
    WEBM = "webm"
    WEBM_OPUS = "webm"
    OPUS = "opus"
    OGG = "ogg"
    OGG_OPUS = "ogg"
    M4A = "m4a"
    MP4 = "mp4"
    AAC = "aac"
    MP3 = "mp3"
    MPEG = "mp3"
    WAV = "wav"
    WAVE = "wav"
    FLAC = "flac"
    AIFF = "aiff"
    CAF = "caf"
    THREE_GP = "3gp"
    AMR = "amr"
    WMA = "wma"
    PCM = "pcm"
    RAW = "raw"


class AudioValidationError(Exception):
    """Raised when audio validation fails"""
    pass


class AudioConverter:
    """
    Universal audio converter with validation.
    Outputs Whisper-ready 16kHz mono WAV.
    """

    # MIME type to format mapping
    MIME_MAP: Dict[str, str] = {
        # Web / Browser
        'audio/webm': 'webm',
        'audio/webm;codecs=opus': 'webm',
        'audio/webm;codecs=vorbis': 'webm',
        'audio/webm;codecs=pcm': 'webm',

        # Container
        'audio/mp4': 'm4a',
        'audio/mp4;codecs=mp4a.40.2': 'm4a',
        'audio/x-m4a': 'm4a',
        'video/mp4': 'mp4',

        # Lossy
        'audio/mpeg': 'mp3',
        'audio/mp3': 'mp3',
        'audio/mpg': 'mp3',
        'audio/x-mp3': 'mp3',
        'audio/mpeg3': 'mp3',

        # Lossless
        'audio/wav': 'wav',
        'audio/wave': 'wav',
        'audio/x-wav': 'wav',
        'audio/x-pn-wav': 'wav',
        'audio/flac': 'flac',
        'audio/x-flac': 'flac',
        'audio/aiff': 'aiff',
        'audio/x-aiff': 'aiff',

        # Compressed
        'audio/ogg': 'ogg',
        'audio/ogg;codecs=opus': 'ogg',
        'audio/ogg;codecs=vorbis': 'ogg',
        'application/ogg': 'ogg',
        'audio/opus': 'opus',
        'audio/x-opus': 'opus',

        # Legacy/Mobile
        'audio/3gpp': '3gp',
        'audio/3gpp2': '3gp',
        'audio/amr': 'amr',
        'audio/x-amr': 'amr',
        'audio/x-ms-wma': 'wma',

        # Apple
        'audio/x-caf': 'caf',
        'audio/x-gsm': 'gsm',

        # AAC
        'audio/aac': 'aac',
        'audio/x-aac': 'aac',
    }

    # Magic bytes detection (fallback)
    MAGIC_MAP: Dict[bytes, str] = {
        b'\\x1a\\x45\\xdf\\xa3': 'webm',  # Matroska/WebM
        b'RIFF': 'wav',  # RIFF/WAVE
        b'ID3': 'mp3',   # MP3 with ID3
        b'\\xff\\xfb': 'mp3',  # MP3 frame sync
        b'\\xff\\xf3': 'mp3',
        b'\\xff\\xf2': 'mp3',
        b'fLaC': 'flac',
        b'FORM': 'aiff',  # AIFF
        b'OggS': 'ogg',
        b'ftyp': 'm4a',   # ISO base media
    }

    def __init__(self, max_size_mb: float = 25.0):
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def validate_and_convert(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None
    ) -> bytes:
        """
        Main entry: Validate and convert any audio to 16kHz mono WAV.

        Args:
            audio_bytes: Raw audio file bytes
            filename: Optional filename for extension hint

        Returns:
            16kHz mono WAV bytes ready for Whisper
        """
        # Step 1: Validate size
        self._validate_size(audio_bytes)

        # Step 2: Detect format
        source_format = self._detect_format(audio_bytes, filename)

        if not source_format:
            raise AudioValidationError("Unable to detect audio format")

        logger.info(f"Detected format: {source_format}, size: {len(audio_bytes)/1024:.1f}KB")

        # Step 3: Convert to normalized WAV
        return self._convert_to_whisper_format(audio_bytes, source_format)

    def _validate_size(self, audio_bytes: bytes):
        """Check file size limits"""
        if len(audio_bytes) > self.max_size_bytes:
            raise AudioValidationError(
                f"Audio file too large: {len(audio_bytes)/1024/1024:.1f}MB "
                f"(max {self.max_size_bytes/1024/1024:.1f}MB)"
            )

        if len(audio_bytes) < 100:
            raise AudioValidationError("Audio file too small (likely empty)")

    def _detect_format(
        self,
        audio_bytes: bytes,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        Multi-layer format detection:
        1. MIME type from python-magic
        2. Extension hint from filename
        3. Magic bytes fallback
        """

        # Layer 1: MIME detection (most reliable)
        try:
            mime = magic.from_buffer(audio_bytes, mime=True)
            if mime in self.MIME_MAP:
                logger.debug(f"Format detected via MIME: {mime}")
                return self.MIME_MAP[mime]
        except Exception as e:
            logger.warning(f"MIME detection failed: {e}")

        # Layer 2: Filename extension
        if filename:
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            if ext in [f.value for f in AudioFormat]:
                logger.debug(f"Format detected via extension: {ext}")
                return ext

        # Layer 3: Magic bytes
        header = audio_bytes[:12]
        for magic_bytes, fmt in self.MAGIC_MAP.items():
            if header.startswith(magic_bytes):
                logger.debug(f"Format detected via magic bytes: {fmt}")
                return fmt

        # Default: attempt as WAV
        logger.warning("Format unknown, attempting as WAV")
        return "wav"

    def _convert_to_whisper_format(self, audio_bytes: bytes, source_format: str) -> bytes:
        """
        Convert to 16kHz mono WAV with normalization.
        Uses pydub (ffmpeg wrapper) for maximum format support.
        """
        try:
            # Load via pydub (handles most formats via ffmpeg)
            audio = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format=source_format
            )

        except Exception as pydub_err:
            # Fallback: try ffmpeg-python directly
            logger.warning(f"Pydub failed, trying ffmpeg directly: {pydub_err}")
            try:
                audio = self._ffmpeg_fallback_convert(audio_bytes, source_format)
            except Exception as ffmpeg_err:
                raise AudioValidationError(
                    f"Cannot convert audio: pydub={pydub_err}, ffmpeg={ffmpeg_err}"
                )

        # Validate we got audio
        if audio is None or len(audio) == 0:
            raise AudioValidationError("Loaded audio is empty")

        # Normalize: mono, 16kHz, consistent volume
        audio = self._normalize_audio(audio)

        # Export to WAV
        output = io.BytesIO()
        audio.export(
            output,
            format="wav",
            codec="pcm_s16le",  # 16-bit PCM for Whisper
            parameters=["-ar", "16000", "-ac", "1"]
        )

        wav_bytes = output.getvalue()
        logger.info(f"Converted to WAV: {len(wav_bytes)/1024:.1f}KB")

        return wav_bytes

    def _normalize_audio(self, audio: AudioSegment) -> AudioSegment:
        """
        Normalize audio for optimal Whisper processing:
        - Mono channel
        - 16kHz sample rate
        - Normalized volume (-1dBFS peak)
        - Silence trimmed (optional)
        """
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Resample to 16kHz (Whisper requirement)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)

        # Normalize volume to -1dBFS (prevent clipping, boost quiet)
        audio = normalize(audio, headroom=1.0)

        return audio

    def _ffmpeg_fallback_convert(self, audio_bytes: bytes, source_format: str) -> AudioSegment:
        """Direct ffmpeg fallback for problematic formats"""
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=f'.{source_format}', delete=False) as inp:
            inp.write(audio_bytes)
            inp_path = inp.name

        out_path = inp_path + '.wav'

        try:
            # ffmpeg direct conversion
            subprocess.run([
                'ffmpeg', '-y', '-i', inp_path,
                '-ar', '16000', '-ac', '1',
                '-f', 'wav', out_path
            ], check=True, capture_output=True)

            audio = AudioSegment.from_wav(out_path)
            return audio

        finally:
            import os
            os.unlink(inp_path)
            if os.path.exists(out_path):
                os.unlink(out_path)

    def get_audio_info(self, audio_bytes: bytes) -> dict:
        """Get audio metadata for debugging"""
        try:
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
            return {
                "channels": audio.channels,
                "sample_rate": audio.frame_rate,
                "bit_depth": audio.sample_width * 8,
                "duration_sec": len(audio) / 1000,
                "bytes": len(audio_bytes)
            }
        except Exception as e:
            return {"error": str(e)}


def get_converter(max_size_mb: float = 25.0) -> AudioConverter:
    """Factory for AudioConverter singleton"""
    return AudioConverter(max_size_mb)
