"""
Robo Buddy - Live Voice Chat Setup & Usage Guide
ChatGPT-like voice interaction with real-time recording
"""

GUIDE = """

╔══════════════════════════════════════════════════════════════════════╗
║       ROBO BUDDY - LIVE VOICE CHAT (ChatGPT-like Interface)          ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT'S NEW
═══════════════════════════════════════════════════════════════════════

✓ Live Audio Recording
  - Record from microphone with real-time speech detection
  - Automatic silence detection (stops recording when you finish speaking)
  - Similar to ChatGPT's voice mode

✓ Energy-Based Voice Activity Detection (VAD)
  - No external dependencies needed
  - Uses audio energy levels to detect speech vs silence
  - Adjustable sensitivity

✓ Seamless Voice Chat
  - Record → Transcribe → Process → Respond → Playback
  - All in one interface
  - Natural conversation flow


QUICK START (3 STEPS)
═══════════════════════════════════════════════════════════════════════

Step 1: Start the Server (Terminal 1)
────────────────────────────────────────────────────────────────────
cd C:\\Users\\Lenovo\\Desktop\\RoboBuddy
.\\venv\\Scripts\\Activate.ps1
uvicorn server:app --reload


Step 2: Start Live Voice Chat (Terminal 2)
────────────────────────────────────────────────────────────────────
cd C:\\Users\\Lenovo\\Desktop\\RoboBuddy
.\\venv\\Scripts\\Activate.ps1
python live_voice_client.py


Step 3: Choose Mode & Start Talking!
────────────────────────────────────────────────────────────────────
When prompted, select:
  1 = Interactive mode (continuous conversation)
  2 = Single query mode (record once)
  3 = Process file (use existing audio)


MODES EXPLAINED
═══════════════════════════════════════════════════════════════════════

1. INTERACTIVE MODE (Recommended)
   ├─ Continuously listens for voice input
   ├─ Automatically detects when you stop speaking
   ├─ Sends to AI, gets response
   ├─ Plays audio response automatically
   └─ Ready for next question immediately
  
  Usage: python live_voice_client.py → Select option 1


2. SINGLE QUERY MODE
   ├─ Records one question after you select this mode
   ├─ Processes it and returns answer
   ├─ Can ask to replay audio
   └─ Good for testing
  
  Usage: python live_voice_client.py → Select option 2


3. BATCH MODE
   ├─ Process pre-recorded audio files
   ├─ Uses audio you already have
   └─ Good for debugging
  
  Usage: python live_voice_client.py voice sample.wav


HOW IT WORKS
═══════════════════════════════════════════════════════════════════════

USER SPEAKS
    ↓
[PYAUDIO] Records audio from microphone
    ↓
[ENERGY DETECTION] Detects when user stops (silence)
    ↓
[SAVE AUDIO] Saves WAV file of user input
    ↓
[HTTP POST] Sends to server /voice_chat endpoint
    ↓
[STT] faster_whisper transcribes audio to text
    ↓
[INTENT] LLM classifies intent + context
    ↓
[LLM] Ollama generates response
    ↓
[TTS] Piper converts response to audio
    ↓
[HTTP RESPONSE] Returns transcription + reply + audio
    ↓
[PLAYBACK] winsound plays audio response
    ↓
READY FOR NEXT QUESTION


RECORDING BEHAVIOR
═══════════════════════════════════════════════════════════════════════

Display Format:
  🔴 123 frames | Energy: 456 🔊     ← User is speaking (high energy)
  ⚪ 124 frames | Silence: 2/8         ← User stopped (counting)
  
Recording stops when:
  └─ 8+ frames of silence detected AFTER speech started
  └─ OR timeout (30 seconds) reached
  └─ OR Ctrl+C pressed


AUDIO QUALITY
═══════════════════════════════════════════════════════════════════════

Settings:
  Sample Rate: 16000 Hz (16 kHz)
  Channels: 1 (Mono)
  Bit Depth: 16-bit
  Chunk Duration: 30ms
  Format: PCM WAV

Files Generated:
  recorded_audio.wav  ← Your input (100-500 KB depending on duration)
  response.wav        ← AI's audio response


TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════

Problem: "No microphone detected"
Solution:
  ├─ Check microphone connection
  ├─ Verify Windows sound settings
  └─ Try: python audio_recorder.py

Problem: "Cuts off during speaking"
Solution:
  └─ Energy threshold too high
  └─ Edit audio_recorder.py:
      self.energy_threshold = 200  (lower = more sensitive)

Problem: "Records too much silence"
Solution:
  └─ Energy threshold too low
  └─ Edit audio_recorder.py:
      self.energy_threshold = 500  (higher = less sensitive)

Problem: "No audio response playback"
Solution:
  ├─ Check TTS is enabled: python client.py tts "hello"
  ├─ Verify response.wav exists after chat
  └─ Try playing manually: python -c "import winsound; winsound.PlaySound('response.wav', winsound.SND_FILENAME)"

Problem: "Server connection failed"
Solution:
  ├─ Verify Ollama is running: ollama serve
  ├─ Start FastAPI server: uvicorn server:app --reload
  └─ Check port 8000 is available


TESTING THE PIPELINE
═══════════════════════════════════════════════════════════════════════

Test 1: Check Server Status
  python client.py status
  → Should show: STT: ✓  TTS: ✓

Test 2: Test Text Chat
  python client.py chat "Hello"
  → Should get AI response

Test 3: Test TTS
  python client.py tts "Hello world"
  → Should create response.wav

Test 4: Test STT
  python client.py stt sample.wav
  → Should transcribe audio

Test 5: Test Voice Chat (Complete Pipeline)
  python client.py voice sample.wav
  → Should: transcribe + respond + create audio

Test 6: Live Recording
  python audio_recorder.py
  → Record audio interactively


PERFORMANCE TIPS
═══════════════════════════════════════════════════════════════════════

For Faster Response:
  ├─ Use smaller LLM model
  ├─ Reduce Whisper model size
  └─ Enable GPU acceleration if available

For Better Recognition:
  ├─ Speak clearly and at normal pace
  ├─ Reduce background noise
  ├─ Use quality microphone
  └─ Increase energy_threshold if false positives


KEYBOARD SHORTCUTS
═══════════════════════════════════════════════════════════════════════

Recording:
  Ctrl+C  ← Stop recording immediately
  
Interactive Mode:
  Ctrl+C  ← Exit to main menu


FILES CREATED
═══════════════════════════════════════════════════════════════════════

Audio Files:
  recorded_audio.wav  ← Your voice input (auto-created)
  response.wav        ← AI's voice response (auto-created)
  test_audio.wav      ← Test file (for debugging)
  sample.wav          ← Your existing audio file

Python Modules:
  audio_recorder.py       ← Recording with VAD
  live_voice_client.py    ← Interactive voice interface
  server.py               ← FastAPI server
  client.py               ← Non-interactive client


NEXT STEPS
═══════════════════════════════════════════════════════════════════════

1. ✓ Run live_voice_client.py
2. ☐ Integrate with face recognition
3. ☐ Add command-based actions
4. ☐ Improve TTS voice quality
5. ☐ Add context persistence
6. ☐ Deploy to Raspberry Pi


EXAMPLE CONVERSATION
═══════════════════════════════════════════════════════════════════════

User:     "What is the weather today?"
System:   [Records audio → Transcribes → Sends to AI]
AI:       "I don't have access to current weather, but I can help
           you check a weather website. Would you like me to guide
           you to weather.com?"
System:   [Converts to speech → Plays back automatically]
          🔊 [Audio plays: "I don't have access..."]


ADVANCED: ADJUST SENSITIVITY
═══════════════════════════════════════════════════════════════════════

Edit live_voice_client.py or audio_recorder.py:

To make it MORE sensitive (catches quieter speech):
  self.energy_threshold = 200    # Lower = more sensitive

To make it LESS sensitive (ignores background noise):
  self.energy_threshold = 500    # Higher = less sensitive

To make it stop talking SLOWER:
  self.silence_threshold = 15    # More frames of silence needed

To make it stop talking FASTER:
  self.silence_threshold = 5     # Fewer frames of silence needed


═══════════════════════════════════════════════════════════════════════

READY TO START?

Run this command in PowerShell:

python live_voice_client.py

═══════════════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(GUIDE)
    input("\nPress Enter to continue...")
