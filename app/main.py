"""
AI Voice Sales Agent - FastAPI Application
Core implementation for AI Voice Sales Agent with voice conversation capabilities.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime
import uuid
import sys
import os
import base64

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pydantic import BaseModel
from services.llm import LLMService
from services.tts import TTSService
from services.stt import STTService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class CallRequest(BaseModel):
    phone_number: str
    customer_name: str

class MessageRequest(BaseModel):
    message: str

class StartCallResponse(BaseModel):
    call_id: str
    message: str
    first_message: str
    audio_base64: Optional[str] = None

class RespondResponse(BaseModel):
    reply: str
    should_end_call: bool
    audio_base64: Optional[str] = None

class ConversationResponse(BaseModel):
    call_id: str
    history: List[Dict]

class VoiceResponse(BaseModel):
    text: str
    audio_base64: str
    duration: float

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Sales Agent",
    description="AI voice sales agent for course pitching with voice conversation capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
llm_service = LLMService()
tts_service = TTSService()
stt_service = STTService()

# Ensure audio output directory exists
os.makedirs("audio_output", exist_ok=True)

# Store conversations (in production, use a database)
conversations: Dict[str, Dict] = {}

@app.post("/start-call", response_model=StartCallResponse)
async def start_call(request: CallRequest):
    """Start a new sales call."""
    try:
        call_id = str(uuid.uuid4())
        
        # Create conversation
        conversations[call_id] = {
            "call_id": call_id,
            "customer_name": request.customer_name,
            "phone_number": request.phone_number,
            "state": "introduction",
            "messages": [],
            "qualification_answers": {},
            "customer_info": {},
            "start_time": datetime.now().isoformat()
        }
        
        # Generate initial greeting
        greeting = llm_service.generate_introduction(request.customer_name)
        
        # Add to conversation history
        conversations[call_id]["messages"].append({
            "role": "agent",
            "content": greeting,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate audio using TTS
        audio_bytes = await tts_service.synthesize_speech(greeting)
        audio_base64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
        
        logger.info(f"Call started: {call_id}")
        logger.info(f"Generated audio: {audio_bytes}")
        
        return StartCallResponse(
            call_id=call_id,
            message=greeting,
            first_message=greeting,
            audio_base64=audio_base64
        )
        
    except Exception as e:
        logger.error(f"Error starting call: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/respond/{call_id}", response_model=RespondResponse)
async def respond_to_customer(call_id: str, request: MessageRequest):
    """Process customer message and generate response."""
    try:
        if call_id not in conversations:
            raise HTTPException(status_code=404, detail="Call not found")
        
        conversation = conversations[call_id]
        
        # Add customer message
        conversation["messages"].append({
            "role": "customer",
            "content": request.message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Analyze message
        analysis = llm_service.analyze_message(request.message)
        
        # Generate response
        customer_info = conversation.get("customer_info", {})
        customer_info.update(analysis)
        
        # Use new LLMService signature
        reply, should_end_call = llm_service.generate_response(request.message, conversation["messages"], customer_info)
        
        # Update conversation with any changes to customer_info
        conversation["customer_info"] = customer_info
        
        # Add agent response
        conversation["messages"].append({
            "role": "agent",
            "content": reply,
            "timestamp": datetime.now().isoformat()
        })
        
        # Truncate reply for TTS
        tts_text = reply[:300]
        logger.info(f"TTS text: {tts_text} (len={len(tts_text)})")
        audio_bytes = await tts_service.synthesize_speech(tts_text)
        if not audio_bytes:
            fallback = "Sorry, the agent's voice is temporarily unavailable."
            audio_bytes = await tts_service.synthesize_speech(fallback)
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8") if audio_bytes else None
        
        # Check if call should end
        should_end = len(conversation["messages"]) >= 20
        
        if should_end:
            conversation["end_time"] = datetime.now().isoformat()
            logger.info(f"Call ended: {call_id}")
        
        logger.info(f"Generated response audio: {audio_bytes}")
        
        return RespondResponse(
            reply=reply,
            should_end_call=should_end_call,
            audio_base64=audio_base64
        )
        
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/transcribe/{call_id}")
async def transcribe_voice(call_id: str, audio_file: UploadFile = File(...)):
    """Transcribe voice input and process response."""
    try:
        if call_id not in conversations:
            raise HTTPException(status_code=404, detail="Call not found")
        
        # Read audio file
        audio_data = await audio_file.read()
        
        if not audio_data:
            raise HTTPException(status_code=400, detail="No audio data provided")
        
        # Transcribe audio using STT
        transcription = stt_service.process_audio_file(audio_data)
        
        if not transcription:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")
        
        # Process the transcribed message
        conversation = conversations[call_id]
        
        # Add customer message
        conversation["messages"].append({
            "role": "customer",
            "content": transcription,
            "timestamp": datetime.now().isoformat()
        })
        
        # Analyze message
        analysis = llm_service.analyze_message(transcription)
        
        # Generate response
        customer_info = conversation.get("customer_info", {})
        customer_info.update({
            "name": conversation["customer_name"],
            "phone_number": conversation["phone_number"],
            "qualification_answers": conversation["qualification_answers"]
        })
        
        response = llm_service.generate_response(
            message=transcription,
            history=conversation["messages"],
            customer_info=customer_info
        )
        
        # Update conversation with any changes to customer_info
        conversation["customer_info"] = customer_info
        
        # Add agent response
        conversation["messages"].append({
            "role": "agent",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update qualification answers if in qualification stage
        if "qualification_questions" in customer_info:
            questions_asked = len(customer_info["qualification_questions"])
            if questions_asked < 3:  # We ask 3 qualification questions
                if "background" not in conversation["qualification_answers"]:
                    conversation["qualification_answers"]["background"] = transcription
                elif "goals" not in conversation["qualification_answers"]:
                    conversation["qualification_answers"]["goals"] = transcription
                elif "timeline" not in conversation["qualification_answers"]:
                    conversation["qualification_answers"]["timeline"] = transcription
        
        # Generate audio response using TTS
        audio_bytes = await tts_service.synthesize_speech(response)
        audio_base64 = base64.b64encode(audio_bytes).decode() if audio_bytes else None
        
        # Check if call should end
        should_end = len(conversation["messages"]) >= 20
        
        if should_end:
            conversation["end_time"] = datetime.now().isoformat()
            logger.info(f"Call ended: {call_id}")
        
        return {
            "text": response,
            "audio_base64": audio_base64,
            "duration": 0.0
        }
        
    except Exception as e:
        logger.error(f"Error processing voice input: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{call_id}", response_model=ConversationResponse)
async def get_conversation(call_id: str):
    """Get conversation details and history."""
    try:
        if call_id not in conversations:
            raise HTTPException(status_code=404, detail="Call not found")
        
        conversation = conversations[call_id]
        
        return ConversationResponse(
            call_id=call_id,
            history=conversation["messages"]
        )
        
    except Exception as e:
        logger.error(f"Error getting conversation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "llm": "operational",
            "tts": "operational", 
            "stt": "operational"
        },
        "active_conversations": len(conversations),
        "voice_capabilities": {
            "tts": "gTTS (Google Text-to-Speech)",
            "stt": "OpenAI Whisper",
            "supported_formats": ["mp3", "wav", "m4a", "flac", "ogg"]
        }
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AI Voice Sales Agent API",
        "version": "1.0.0",
        "api_requirements": {
            "POST /start-call": {
                "input": ["phone_number", "customer_name"],
                "output": ["call_id", "message", "first_message"]
            },
            "POST /respond/{call_id}": {
                "input": ["message"],
                "output": ["reply", "should_end_call"]
            },
            "GET /conversation/{call_id}": {
                "output": ["call_id", "history"]
            }
        },
        "voice_features": {
            "text_to_speech": "Available via /voice/synthesize",
            "speech_to_text": "Available via /voice/transcribe/{call_id}",
            "real_time_conversation": "Available via /respond/{call_id}"
        },
        "endpoints": {
            "POST /start-call": "Start a new sales call",
            "POST /respond/{call_id}": "Process customer message",
            "POST /voice/transcribe/{call_id}": "Process voice input",
            "GET /conversation/{call_id}": "Get conversation history",
            "GET /health": "Health check"
        },
        "course_info": {
            "name": "AI Mastery Bootcamp",
            "duration": "12 weeks",
            "price": 499,
            "discount_price": 299,
            "features": [
                "Learn LLMs, Computer Vision, and MLOps",
                "Hands-on projects", 
                "Job placement assistance",
                "Certificate upon completion"
            ]
        },
        "testing": {
            "voice_test_interface": "Visit /test for interactive voice testing",
            "quick_tts_test": "Visit /voice/test-tts?text=Hello%20World",
            "voice_test": "POST audio file to /voice/test",
            "health_check": "Visit /health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting AI Voice Sales Agent...")
    print("📞 Course: AI Mastery Bootcamp")
    print("💰 Special Offer: $299 (Regular: $499)")
    print("⏰ Duration: 12 weeks")
    print("\n🌐 Web Interface: http://localhost:8000/static/test_voice.html")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("\n🎤 Voice Features:")
    print("   - Text-to-Speech: GET /voice/synthesize?text=your_text")
    print("   - Speech-to-Text: POST /voice/transcribe/{call_id}")
    print("   - Real-time voice conversation available")
    print("\n🧪 Testing:")
    # print("   - Voice Test Interface: http://localhost:8000/test")
    print("   - TTS Test: http://localhost:8000/voice/synthesize?text=Hello%20World")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 