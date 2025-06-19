"""
Text-to-Speech (TTS) Service
Enhanced implementation using gTTS for text-to-speech conversion.
"""

import logging
import os
import hashlib
import io
import asyncio
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np

try:
    import torch
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
    from transformers import AutoTokenizer
    import soundfile as sf
    HF_TTS_AVAILABLE = True
except ImportError:
    HF_TTS_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class TTSService:
    """TTS service using HuggingFace SpeechT5 (microsoft/speecht5_tts)."""
    
    def __init__(self):
        """Initialize the TTS service."""
        try:
            # Create output directory
            os.makedirs("audio_output", exist_ok=True)
            
            # Thread pool for async operations
            self.executor = ThreadPoolExecutor(max_workers=2)
            
            # Default settings
            self.default_language = 'en'
            self.model = None
            self.processor = None
            self.speaker_embeddings = None
            if HF_TTS_AVAILABLE:
                try:
                    self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
                    self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
                    # Use a default speaker embedding from the repo
                    import requests
                    speaker_url = "https://huggingface.co/microsoft/speecht5_tts/resolve/main/speaker_embeddings/spk1.npy"
                    emb_data = requests.get(speaker_url).content
                    self.speaker_embeddings = np.load(io.BytesIO(emb_data))
                    logger.info("Loaded SpeechT5 model and default speaker embedding.")
                except Exception as e:
                    logger.error(f"Error loading SpeechT5: {e}")
                    self.model = None
            else:
                logger.warning("HuggingFace SpeechT5 not available. TTS will not work.")
            
        except Exception as e:
            logger.error(f"Error initializing TTS service: {str(e)}")
            raise

    async def synthesize_speech(self, text: str, language: str = 'en', slow: bool = False) -> Optional[bytes]:
        """
        Convert text to speech using gTTS and return audio as bytes (in-memory).
        """
        if not HF_TTS_AVAILABLE or self.model is None or self.processor is None or self.speaker_embeddings is None:
            logger.error("SpeechT5 not available. Returning None.")
            return None
        if not text or not text.strip():
            logger.warning("Empty text provided for TTS")
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self._synthesize_sync, text)

    def _synthesize_sync(self, text: str) -> Optional[bytes]:
        """Generate speech and return as bytes."""
        try:
            inputs = self.processor(text=text, return_tensors="pt")
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings=torch.tensor(self.speaker_embeddings).unsqueeze(0)
                )
            buf = io.BytesIO()
            sf.write(buf, speech.cpu().numpy(), 16000, format='WAV')
            buf.seek(0)
            return buf.read()
        except Exception as e:
            logger.error(f"SpeechT5 synthesis failed: {e}")
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
            "de",  # German
            "it",  # Italian
            "pt",  # Portuguese
            "ru",  # Russian
            "ja",  # Japanese
            "ko",  # Korean
            "zh",  # Chinese
            "hi",  # Hindi
            "ar",  # Arabic
            "nl",  # Dutch
            "pl",  # Polish
            "tr"   # Turkish
        ]

    def __del__(self):
        """Cleanup when service is destroyed."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass 