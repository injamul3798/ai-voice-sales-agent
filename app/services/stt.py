"""
Speech-to-Text (STT) Service
Enhanced implementation using OpenAI Whisper for speech-to-text conversion.
"""

import logging
import whisper
import os
import tempfile
from typing import Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class STTService:
    """Enhanced Speech-to-Text service using Whisper."""

    def __init__(self, model_size: str = "base"):
        """Initialize the STT service."""
        try:
            # Load Whisper model
            self.model = whisper.load_model(model_size)
            self.model_size = model_size
            
            # Thread pool for async operations
            self.executor = ThreadPoolExecutor(max_workers=2)
            
            # Supported audio formats
            self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
            
            logger.info(f"STT service initialized with Whisper {model_size} model")
            
        except Exception as e:
            logger.error(f"Error initializing STT service: {str(e)}")
            raise

    async def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Convert speech to text using Whisper.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Optional[str]: Transcribed text
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                return None
            
            # Check file format
            file_ext = os.path.splitext(audio_path)[1].lower()
            if file_ext not in self.supported_formats:
                logger.warning(f"Unsupported audio format: {file_ext}")
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._transcribe_sync,
                audio_path
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {str(e)}")
            return None

    def _transcribe_sync(self, audio_path: str) -> str:
        """Synchronous transcription method."""
        try:
            # Transcribe audio
            result = self.model.transcribe(audio_path)
            transcription = result["text"].strip()
            
            logger.info(f"Transcribed audio from: {audio_path}")
            logger.info(f"Transcription: {transcription[:100]}...")
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error in synchronous transcription: {str(e)}")
            raise

    def process_audio_file(self, audio_data: bytes) -> Optional[str]:
        """Process raw audio data and convert it to text."""
        try:
            if not audio_data:
                logger.warning("Empty audio data provided")
                return None
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name
            
            try:
                # Transcribe
                result = self.model.transcribe(temp_path)
                transcription = result["text"].strip()
                
                logger.info(f"Processed audio data, transcription: {transcription[:100]}...")
                
                # Return actual transcription or a fallback message
                if transcription:
                    return transcription
                else:
                    return "I couldn't hear what you said clearly. Could you please repeat that?"
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            # Return a helpful message instead of None
            return "I'm having trouble processing your audio. Could you please try speaking more clearly?"

    async def transcribe_with_metadata(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio and return metadata.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Dict[str, Any]: Transcription and metadata
        """
        try:
            if not os.path.exists(audio_path):
                return {"success": False, "error": "File not found"}
            
            # Get file info
            file_size = os.path.getsize(audio_path)
            file_duration = self._get_audio_duration(audio_path)
            
            # Transcribe
            transcription = await self.transcribe_audio(audio_path)
            
            if not transcription:
                return {"success": False, "error": "Transcription failed"}
            
            return {
                "success": True,
                "transcription": transcription,
                "file_size": file_size,
                "duration": file_duration,
                "model_used": self.model_size,
                "confidence": 0.85  # Mock confidence score
            }
            
        except Exception as e:
            logger.error(f"Error in transcribe_with_metadata: {str(e)}")
            return {"success": False, "error": str(e)}

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            # This is a simplified version - in production you'd use a proper audio library
            # For now, we'll estimate based on file size
            file_size = os.path.getsize(audio_path)
            # Rough estimate: 1MB ≈ 1 minute for typical audio
            estimated_duration = file_size / (1024 * 1024) * 60
            return round(estimated_duration, 2)
        except:
            return 0.0

    def set_model(self, model_size: str):
        """Change the Whisper model size."""
        try:
            self.model = whisper.load_model(model_size)
            self.model_size = model_size
            logger.info(f"Changed to {model_size} model")
        except Exception as e:
            logger.error(f"Error changing model: {str(e)}")

    def get_supported_formats(self) -> list:
        """Get list of supported audio formats."""
        return self.supported_formats

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_size": self.model_size,
            "supported_formats": self.supported_formats,
            "status": "loaded"
        }

    def __del__(self):
        """Cleanup when service is destroyed."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except:
            pass 