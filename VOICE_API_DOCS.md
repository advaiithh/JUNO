# ROBO BUDDY - INTEGRATED TTS/STT SERVER

## Overview
Server has been upgraded with full voice capabilities:
- **STT**: Speech-to-Text using faster_whisper
- **TTS**: Text-to-Speech using Piper
- **Voice Chat**: Complete voice interaction pipeline

## Installation & Setup

### 1. Install Dependencies
```bash
pip install fastapi uvicorn faster_whisper requests
```

### 2. Verify Setup
```bash
python check_status.py
```

## Running the Server

### Start Server
```bash
uvicorn server:app --reload
```

Server runs at: `http://localhost:8000`

### Check Status
```bash
python test_server.py
```

---

## API Endpoints

### 1. **TEXT CHAT** - `/chat`
**Method:** POST
**Parameters:** `prompt` (string)

```bash
curl -X POST "http://localhost:8000/chat?prompt=What%20is%20the%20weather"
```

**Response:**
```json
{
  "reply": "The weather is..."
}
```

---

### 2. **SPEECH-TO-TEXT** - `/stt`
**Method:** POST
**Body:** Audio file (.wav)

```bash
curl -X POST -F "file=@audio.wav" http://localhost:8000/stt
```

**Response:**
```json
{
  "text": "Transcribed text",
  "error": null
}
```

---

### 3. **TEXT-TO-SPEECH** - `/tts`
**Method:** POST
**Parameters:** `text` (string)

```bash
curl -X POST "http://localhost:8000/tts?text=Hello%20there" -o response.wav
```

**Response:**
```json
{
  "audio_file": "response.wav",
  "error": null
}
```

---

### 4. **VOICE CHAT** - `/voice_chat` ⭐
**Method:** POST (Complete Pipeline)
**Body:** Audio file (.wav)

Automatically performs:
1. **STT** - Transcribes audio
2. **Intent Classification** - Understands command
3. **LLM** - Generates response
4. **TTS** - Converts to speech

```bash
curl -X POST -F "file=@voice_command.wav" http://localhost:8000/voice_chat
```

**Response:**
```json
{
  "text": "What is the weather",
  "reply": "Current weather is...",
  "audio_file": "response.wav",
  "tts_error": null
}
```

---

### 5. **SYSTEM STATUS** - `/status`
**Method:** GET

```bash
curl http://localhost:8000/status
```

**Response:**
```json
{
  "status": "running",
  "stt_available": true,
  "tts_available": true,
  "llm_url": "http://localhost:11434/api/generate",
  "conversation_history_size": 5
}
```

---

## Testing

### Run Full Test Suite
```bash
python test_server.py
```

### Manual Testing with Python
```python
import requests

# Text chat
response = requests.post("http://localhost:8000/chat", params={"prompt": "Hello"})
print(response.json())

# Text to speech
response = requests.post("http://localhost:8000/tts", params={"text": "Hello world"})
print(response.json())

# Upload audio for transcription
with open("audio.wav", "rb") as f:
    response = requests.post("http://localhost:8000/stt", files={"file": f})
    print(response.json())

# Complete voice interaction
with open("command.wav", "rb") as f:
    response = requests.post("http://localhost:8000/voice_chat", files={"file": f})
    print(response.json())
```

---

## System Architecture

```
VOICE INPUT (Microphone)
    ↓
[STT] Speech-to-Text (faster_whisper)
    ↓
[INTENT] Classification (LLM)
    ↓
[CONTEXT] Conversation History
    ↓
[LLM] Generate Response (Ollama)
    ↓
[TTS] Text-to-Speech (Piper)
    ↓
VOICE OUTPUT (Speaker)
```

---

## Configuration

### LLM Model
Edit `server.py`:
```python
"model": "llama3.1:8b"  # Change to your model
```

### TTS Voice
Edit `server.py`:
```python
PIPER_VOICE = "piper/voices/en_US-lessac-medium.onnx"  # Change voice
```

### STT Model
Edit `server.py`:
```python
stt_model = WhisperModel("medium", compute_type="int8")  # Change size
```

### Conversation History
Edit `server.py`:
```python
MAX_HISTORY = 10  # Change limit
```

---

## Features Added

✅ **Full Voice Support**
- Microphone input processing
- Real-time speech recognition
- Conversational AI response
- Natural speech output

✅ **Modular Design**
- Separate STT, TTS, LLM endpoints
- Graceful error handling
- Auto-detection of available features

✅ **Voice Pipeline**
- Single endpoint for complete workflow
- Maintains conversation context
- Intent-aware responses

✅ **Status Monitoring**
- Check system health
- Verify component availability
- Monitor conversation history

---

## Troubleshooting

### Server won't start
- Ensure Ollama is running: `ollama serve`
- Check port 8000 is available

### STT not working
- Install: `pip install faster-whisper`
- Check microphone input

### TTS generates but no audio
- Verify Piper files exist:
  - `piper/piper.exe`
  - `piper/voices/en_US-lessac-medium.onnx`

### LLM errors
- Ensure Ollama server is running
- Check `OLLAMA_URL` in server.py

---

## Next Steps

1. **Integrate with Recognition System**
   - Authenticate user before voice chat
   - Filter voice commands by owner

2. **Add More Commands**
   - Device control
   - Calendar integration
   - Web automation

3. **Optimize Models**
   - Quantize models further
   - Use smaller STT/TTS models
   - Cache responses

---

**Robo Buddy is now ready for full voice interaction! 🚀**
