# 🤖 AI Voice Sales Agent

A FastAPI-based voice sales agent for online course sales. Handles dynamic conversations, lead qualification, and objection handling using open-source AI models.

---

## Features

- Real-time voice chat (TTS & STT)
- Sales flow: introduction, qualification, pitch, objection handling, closing
- LLM-powered (LangChain + HuggingFace DialoGPT-medium)
- All audio and data handled in-memory

## Tech Stack

- Python, FastAPI
- LangChain (LLM orchestration)
- HuggingFace DialoGPT-medium (LLM)
- gTTS (TTS)
- OpenAI Whisper (STT)
- Minimal HTML/JS frontend

## Quick Start

```bash
# Clone and enter repo
$ git clone https://github.com/injamul3798/ai-voice-sales-agent
$ cd ai-voice-sales-agent

# Create and activate virtualenv
$ python -m venv venv
$ venv\Scripts\activate  # or source venv/bin/activate

# Install dependencies
$ pip install -r requirements.txt

# Run the server
$ uvicorn app.main:app --reload
```

Open [http://localhost:8000/static/test_voice.html](http://localhost:8000/static/test_voice.html) in your browser.

## API Endpoints

- `POST /start-call` — Start a new call
- `POST /respond/{call_id}` — Send message, get agent reply
- `GET /conversation/{call_id}` — Get conversation transcript
- `POST /voice/transcribe/{call_id}` — Transcribe audio

---

All models and frameworks are free/open-source. No paid API keys required.


