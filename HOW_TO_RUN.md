# How to Run JUNO

## Quick Start

### 1. Activate Virtual Environment
```bash
cd /home/jinto/Desktop/JUNO
source venv/bin/activate
```

### 2. Start Ollama (Required for LLM)
Make sure Ollama is running with the llama3.1:8b model:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it:
ollama serve

# In another terminal, pull the model if needed:
ollama pull llama3.1:8b
```

### 3. Start the Server

**Option A: Direct Python (Recommended)**
```bash
python server.py
```

**Option B: Using Uvicorn**
```bash
uvicorn server:app --reload --port 8000
```

**Option C: Using the startup script**
```bash
bash start_juno.sh
```

### 4. Access the Web UI

Open your browser and go to:
```
http://localhost:8000/ui/index.html
```

## What You'll See

When the server starts, you should see:
```
============================================================
 JUNO - AI Voice Assistant Server with Face Authentication
============================================================
 Face Auth: [OK] Available (or [WARNING] Not available)
 STT: [OK] Available (or [ERROR] Not available)
 TTS: [OK] Available (or [ERROR] Not available)
 LLM: http://localhost:11434/api/generate
============================================================
 Server URLs:
   • API:    http://localhost:8000
   • Web UI: http://localhost:8000/ui/index.html
============================================================
```

## Troubleshooting

### Server won't start
- Check if port 8000 is already in use: `lsof -i :8000`
- Make sure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`

### "Cannot connect to Ollama" error
- Make sure Ollama is running: `ollama serve`
- Check if model is installed: `ollama list`
- Install model: `ollama pull llama3.1:8b`

### "STT not available" error
- Install faster_whisper: `pip install faster-whisper`

### "TTS not available" error
- Check if piper directory exists and has the voice model
- See `setup_voice.py` or `DOWNLOAD_MODEL.md` for voice setup

### UI not loading
- Check browser console (F12) for errors
- Verify server is running: `curl http://localhost:8000/status`
- Make sure you're accessing: `http://localhost:8000/ui/index.html`

## Testing the Server

Test if the server is responding:
```bash
curl http://localhost:8000/status
```

You should get a JSON response with server status.

## Stopping the Server

Press `Ctrl+C` in the terminal where the server is running.
