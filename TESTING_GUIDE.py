"""
Robo Buddy - Complete Testing Guide
Step-by-step instructions to test all features
"""

GUIDE = """
╔══════════════════════════════════════════════════════════════════════╗
║          ROBO BUDDY - COMPLETE TESTING GUIDE                         ║
╚══════════════════════════════════════════════════════════════════════╝

SETUP CHECKLIST
═══════════════════════════════════════════════════════════════════════

Before testing, ensure:

  ☐ Virtual environment activated
    Command: .\\venv\\Scripts\\Activate.ps1

  ☐ Ollama server is running
    Command (in another terminal): ollama serve
    Check: http://localhost:11434/api/generate

  ☐ All dependencies installed
    Command: pip list | grep -E "fastapi|requests|faster|onnxruntime"

  ☐ FFmpeg installed (for audio processing)
    Command: ffmpeg -version

═══════════════════════════════════════════════════════════════════════


STEP 1: START THE SERVER
═══════════════════════════════════════════════════════════════════════

In a PowerShell terminal:

  cd C:\\Users\\Lenovo\\Desktop\\RoboBuddy
  uvicorn server:app --reload

Expected output:
  ✓ Uvicorn running on http://127.0.0.1:8000
  ✓ Application startup complete

═══════════════════════════════════════════════════════════════════════


STEP 2: CHECK SYSTEM STATUS (In a new terminal)
═══════════════════════════════════════════════════════════════════════

Command:
  python client.py status

Expected output:
  ✓ Status: running
  ✓ STT Available: ✓
  ✓ TTS Available: ✓
  ✓ Conversation History: 0 messages

═══════════════════════════════════════════════════════════════════════


STEP 3: TEST TEXT CHAT
═══════════════════════════════════════════════════════════════════════

Command:
  python client.py chat "Hello, what is your name?"

Expected output:
  📝 Chat: Hello, what is your name?
  🤖 Response: [AI response about being Robo Buddy]

═══════════════════════════════════════════════════════════════════════


STEP 4: TEST TEXT-TO-SPEECH
═══════════════════════════════════════════════════════════════════════

Command:
  python client.py tts "Hello, I am Robo Buddy, your personal assistant"

Expected output:
  🔊 Converting to speech: Hello, I am Robo Buddy...
  ✓ Audio generated: response.wav

Verify: Listen to response.wav (should be audio file ~100KB)

═══════════════════════════════════════════════════════════════════════


STEP 5: TEST SPEECH-TO-TEXT
═══════════════════════════════════════════════════════════════════════

First, let's use the test audio we created:

Command:
  python client.py stt test_audio.wav

Expected output:
  🎤 Transcribing: test_audio.wav
  📝 Transcribed: [some text from audio]

If you have a real audio file with voice:

Command:
  python client.py stt sample.wav

═══════════════════════════════════════════════════════════════════════


STEP 6: TEST COMPLETE VOICE CHAT
═══════════════════════════════════════════════════════════════════════

This tests the complete pipeline:
  1. Transcribe your audio
  2. Send to Ollama LLM
  3. Generate response
  4. Convert to speech

Command:
  python client.py voice sample.wav

Expected output:
  🎤 Voice Chat: sample.wav
  📝 You: [transcribed text from audio]
  🤖 Robo Buddy: [AI response]
  🔊 Audio saved: response.wav

═══════════════════════════════════════════════════════════════════════


STEP 7: INTEGRATION WITH FACE RECOGNITION
═══════════════════════════════════════════════════════════════════════

Next: Integrate face recognition to authenticate user before voice chat:

Command:
  python recognition.py

Features:
  1. Register your face (front, left, right angles)
  2. System recognizes only the owner
  3. Other users get rejected

═══════════════════════════════════════════════════════════════════════


TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════

Problem: "File not found: audio.wav"
Solution: 
  - Use existing files: sample.wav or test_audio.wav
  - Or record audio with: python record_audio.py
  - Or generate with: python generate_audio.py

Problem: "Server not responding"
Solution:
  - Check if uvicorn is running: python client.py status
  - Start server: uvicorn server:app --reload
  - Verify Ollama is running: ollama serve

Problem: "STT not available"
Solution:
  - Install: pip install faster-whisper
  - Restart server: Kill previous uvicorn and restart

Problem: "TTS error"
Solution:
  - Verify Piper files exist:
    Test-Path piper\\piper.exe
    Test-Path piper\\voices\\en_US-lessac-medium.onnx
  - If missing, run: python download_voice_model.py

═══════════════════════════════════════════════════════════════════════


QUICK REFERENCE
═══════════════════════════════════════════════════════════════════════

Files in RoboBuddy:
  server.py              ← FastAPI server with all endpoints
  client.py              ← Command-line client for testing
  recognition.py         ← Face detection & authentication
  stt.py, tts.py         ← Individual modules
  test_server.py         ← Extended test client
  check_status.py        ← Quick system check
  generate_audio.py      ← Generate test audio
  test_audio.wav         ← Generated test audio file
  sample.wav             ← Your existing audio file

Available Commands:
  python client.py status          ← Check server status
  python client.py chat "hello"    ← Text chat
  python client.py tts "hello"     ← Text to speech
  python client.py stt audio.wav   ← Speech to text
  python client.py voice audio.wav ← Complete voice interaction

═══════════════════════════════════════════════════════════════════════


TESTING WORKFLOW
═══════════════════════════════════════════════════════════════════════

Terminal 1 (Server):
  cd RoboBuddy
  .\\venv\\Scripts\\Activate.ps1
  uvicorn server:app --reload

Terminal 2 (Testing):
  cd RoboBuddy
  .\\venv\\Scripts\\Activate.ps1
  python client.py status              ← Verify ready
  python client.py chat "test"         ← Test text
  python client.py tts "hello"         ← Test TTS
  python client.py stt test_audio.wav  ← Test STT
  python client.py voice sample.wav    ← Test full pipeline

═══════════════════════════════════════════════════════════════════════

Ready to test? Follow these steps:

  1. Activate venv: .\\venv\\Scripts\\Activate.ps1
  2. Start server: uvicorn server:app --reload  (in another terminal)
  3. Check status: python client.py status
  4. Test chat: python client.py chat "Hello"
  5. Test voice: python client.py voice sample.wav

═══════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(GUIDE)
