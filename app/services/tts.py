"""
Text-to-Speech (TTS) Service
Enhanced implementation using gTTS for text-to-speech conversion.
"""

import logging
import os
import io
import asyncio
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from gtts import gTTS

# Configure logging
logger = logging.getLogger(__name__)

class TTSService:
    """TTS service using gTTS for text-to-speech conversion."""
    
    def __init__(self):
        """Initialize the TTS service."""
        try:
            # Create output directory
            os.makedirs("audio_output", exist_ok=True)
            
            # Thread pool for async operations
            self.executor = ThreadPoolExecutor(max_workers=2)
            
            # Default settings
            self.default_language = 'en'
            logger.info("TTSService initialized.")

        except Exception as e:
            logger.error(f"Error initializing TTS service: {str(e)}")
            raise

    async def synthesize_speech(self, text: str, language: str = 'en', slow: bool = False) -> Optional[bytes]:
        """
        Convert text to speech using gTTS and return audio as bytes (in-memory).
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._synthesize_gtts, text, language, slow)

    def _synthesize_gtts(self, text: str, language: str = 'en', slow: bool = False) -> Optional[bytes]:
        """Generate speech using gTTS and return as bytes."""
        try:
            tts = gTTS(text=text, lang=language, slow=slow)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            buf.seek(0)
            return buf.read()
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            return None

    async def synthesize_speech_with_metadata(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Convert text to speech and return metadata.
        
        Args:
            text (str): Text to convert to speech
            **kwargs: Additional parameters (language, slow, etc.)
            
        Returns:
            Dict[str, Any]: Audio file path and metadata
        """
        try:
            audio_bytes = await self.synthesize_speech(text, **kwargs)
            
            if not audio_bytes:
                return {"success": False, "error": "Failed to generate audio"}
            
            return {
                "success": True,
                "audio_bytes": audio_bytes,
                "text_length": len(text),
                "language": kwargs.get('language', self.default_language)
            }
            
        except Exception as e:
            logger.error(f"Error in synthesize_speech_with_metadata: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_supported_languages(self) -> list:
        """Get list of supported languages."""
        return [
            "en",  # English
            "es",  # Spanish
            "fr",  # French
        ]

    def __del__(self):
        """Cleanup when service is destroyed."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass