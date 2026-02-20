# 🎭 JUNO AI Voice Assistant with Face Authentication

JUNO is an advanced AI voice assistant with face recognition authentication, ensuring only authorized users can access it.

## 🌟 Features

- **Face Authentication**: Secure access with face recognition
- **Voice Interaction**: Natural voice conversations with STT & TTS
- **Web UI**: Beautiful, responsive web interface
- **Memory**: Remembers conversation context
- **Commands**: Time, date, and custom commands

## 📋 Prerequisites (One-Time Setup)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install & Start Ollama

1. Download Ollama from [ollama.ai](https://ollama.ai)
2. Pull the model:
   ```bash
   ollama pull llama3.1:8b
   ```
3. Start Ollama:
   ```bash
   ollama serve
   ```

### 3. Set Up Piper TTS

- Ensure Piper TTS is installed with voice files in `piper/voices/`

### 4. **Register Your Face (IMPORTANT!)**

Before using JUNO, you must register your face:

```bash
python recognition_advanced.py
```

**Registration Steps:**
1. Select option `1` (Register new person)
2. Enter your name (e.g., "Owner")
3. Position your face in the frame
4. Follow on-screen instructions
5. Capture 12 face samples from different angles
6. Complete registration

**Tips for best results:**
- Use good lighting
- Look directly at camera
- Rotate head slightly for variety
- Avoid glasses/hats if possible
- Ensure clear, frontal face view

## 🚀 Running JUNO

### Method 1: Web UI with Face Authentication (Recommended)

1. **Start the server:**
   ```bash
   python server.py
   ```

2. **Open your browser:**
   ```
   http://localhost:8000/ui/auth.html
   ```

3. **Face Authentication Flow:**
   - Click "Start Camera"
   - Position your face in the frame
   - Click "Verify Face"
   - Wait for authentication
   - Once verified, click "Continue to JUNO"

4. **Use JUNO:**
   - Click the microphone button
   - Speak your query
   - JUNO will respond with voice

### Method 2: Direct Access (Without Face Auth)

If face authentication is not available, access directly:
```
http://localhost:8000/ui/index.html
```

## 🔐 Security Features

### Face Authentication
- Verifies user identity before allowing access
- Uses deep learning face recognition
- Supports multiple recognition methods:
  - **face_recognition** (dlib) - Best accuracy
  - **InsightFace** - Excellent accuracy
  - **ArcFace** - Excellent accuracy
  - **OpenCV** - Fallback method

### Session Management
- Creates secure session tokens
- Tokens expire after period of inactivity
- Each session is tracked and logged

## 🛠️ Server API Endpoints

### Authentication Endpoints

- `POST /auth/verify_frame` - Verify face from uploaded frame
- `POST /auth/quick_verify` - Quick camera capture and verify
- `GET /auth/check_session?token={token}` - Check session validity

### Voice Assistant Endpoints

- `POST /chat` - Text chat endpoint
- `POST /voice_chat` - Complete voice interaction (STT→LLM→TTS)
- `POST /stt` - Speech-to-text only
- `POST /tts` - Text-to-speech only
- `GET /status` - Check server status
- `GET /memory` - View conversation history
- `POST /memory/clear` - Clear conversation memory

## 📝 Usage Examples

### 1. First-Time Setup
```bash
# Register face
python recognition_advanced.py

# Start server
python server.py

# Open browser
# Navigate to: http://localhost:8000/ui/auth.html
```

### 2. Verifying Face
- Camera starts automatically on auth page
- Position face in center
- Click "Verify Face" button
- System will analyze and authenticate

### 3. Using Voice Assistant
- Click microphone button or press Space
- Speak clearly
- Wait for response
- Audio will play automatically

### 4. Voice Commands
- "What time is it?"
- "What's the date?"
- "Clear memory"
- General conversation

## 🔧 Troubleshooting

### Camera Not Working
- Check camera permissions in browser
- Ensure no other app is using camera
- Try refreshing the page

### Face Not Recognized
- Ensure you registered your face first
- Improve lighting conditions
- Position face directly in frame
- Re-register with better samples

### Authentication Disabled
- If face authentication module is not available, JUNO will allow direct access
- Install required packages: `pip install face_recognition opencv-python`

### Ollama Connection Error
- Ensure Ollama is running: `ollama serve`
- Check if model is downloaded: `ollama pull llama3.1:8b`
- Verify connection at: http://localhost:11434

### Audio Not Playing
- Check browser audio permissions
- Verify Piper TTS is installed
- Check voice files in `piper/voices/`

## 🎯 Tips for Best Experience

### Face Recognition
- Register in good lighting
- Capture various angles
- Update registration if appearance changes (glasses, beard, etc.)
- Re-register every few months for best accuracy

### Voice Interaction
- Speak clearly and at normal pace
- Minimize background noise
- Use good microphone
- Wait for processing to complete

### Performance
- First LLM request may be slow (model loading)
- Subsequent requests are faster
- Close other camera-using apps
- Use Chrome/Edge for best WebRTC support

## 📚 Advanced Features

### Multiple Users
You can register multiple people:
1. Run `python recognition_advanced.py`
2. Select "Register new person"
3. Register each person separately
4. System tracks individual recognition

### Recognition Memory
- Tracks all authentication attempts
- Logs confidence scores
- Maintains session history
- View stats in `memory/recognition_memory.json`

### Custom Commands
Add custom command processing in `server.py`:
```python
def process_command(text):
    text_lower = text.lower().strip()
    
    # Add your custom command
    if "custom" in text_lower:
        return True, "Custom response"
    
    return False, None
```

## 🔍 System Status

Check system status at any time:
```
http://localhost:8000/status
```

Returns:
- Server status
- STT availability
- TTS availability
- Face auth availability
- Active sessions count

## 📦 File Structure

```
JUNO/
├── server.py                    # Main FastAPI server
├── face_auth.py                 # Face authentication module
├── recognition_advanced.py      # Full face recognition system
├── ui/
│   ├── auth.html               # Authentication page
│   └── index.html              # Main voice assistant UI
├── piper/                      # TTS engine and voices
├── memory/                     # Conversation and recognition memory
├── registered_faces_advanced.pkl  # Registered face data
└── requirements.txt            # Python dependencies
```

## 🚨 Important Notes

1. **Privacy**: Face data is stored locally only
2. **Security**: No data is sent to external servers (except Ollama locally)
3. **Registration Required**: Must register face before first use
4. **Camera Access**: Browser will request camera permissions
5. **Lighting**: Good lighting is crucial for face recognition

## 🆘 Support

If you encounter issues:
1. Check Prerequisites are installed
2. Verify Ollama is running
3. Ensure face is registered
4. Check server logs for errors
5. Try re-registering your face

## 🎉 Enjoy JUNO!

Your secure, voice-activated AI assistant with face authentication is ready to use!

For the best experience:
- ✓ Register face in good lighting
- ✓ Keep Ollama running
- ✓ Use clear audio
- ✓ Speak naturally

---
Made with ❤️ for secure and intelligent voice interaction
