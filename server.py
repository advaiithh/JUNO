# -*- coding: utf-8 -*-
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import sys
import json
import subprocess
import tempfile
import platform
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

# Ensure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Import face authentication module
try:
    from face_auth import verify_owner, capture_and_verify
    FACE_AUTH_AVAILABLE = True
    print("[OK] Face authentication module loaded")
except Exception as e:
    FACE_AUTH_AVAILABLE = False
    print(f"[WARNING] Face authentication not available: {e}")

try:
    from faster_whisper import WhisperModel
    STT_AVAILABLE = True
    # Use BASE model for better accuracy with good speed (2-3s for typical audio)
    stt_model = WhisperModel("base", device="cpu", compute_type="int8", num_workers=4)
    print("[OK] Whisper model loaded: BASE (balanced accuracy & speed)")
except ImportError:
    STT_AVAILABLE = False
    print("[WARNING] faster_whisper not installed. STT features disabled.")
except Exception as e:
    STT_AVAILABLE = False
    print(f"[WARNING] Could not load Whisper model: {e}")

app = FastAPI()

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web UI
ui_path = Path(__file__).parent / "ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")
    print(f"[OK] Web UI available at: http://localhost:8000/ui/index.html")

# Authentication state
authenticated_users = {}  # Store authenticated session tokens


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"  # Fast 1B model (~2-3s) — change to llama3.1:8b for higher quality
PIPER_EXE = "piper/piper.exe"

# Load active voice from config
VOICE_CONFIG_FILE = "piper_voice_config.json"
DEFAULT_VOICE = "piper/voices/en_US-lessac-medium.onnx"

def get_active_voice():
    """Get the active voice model from config"""
    if os.path.exists(VOICE_CONFIG_FILE):
        try:
            with open(VOICE_CONFIG_FILE, 'r') as f:
                config = json.load(f)
                voice_file = config.get("active_voice")
                if voice_file:
                    return f"piper/voices/{voice_file}"
        except:
            pass
    return DEFAULT_VOICE

PIPER_VOICE = get_active_voice()

TTS_AVAILABLE = os.path.exists(PIPER_EXE) and os.path.exists(PIPER_VOICE)

if TTS_AVAILABLE:
    print(f"[OK] TTS Voice loaded: {PIPER_VOICE}")
else:
    print(f"[WARNING] TTS not available - Voice file missing: {PIPER_VOICE}")

# Conversation memory persistence
MEMORY_FILE = "conversation_memory.json"
conversation_history = []
MAX_HISTORY = 6  # Keep short for speed (fewer tokens = faster LLM)

def load_conversation_memory():
    """Load conversation history from file"""
    global conversation_history
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                conversation_history = json.load(f)
            print(f"[OK] Loaded {len(conversation_history)} messages from memory")
        except Exception as e:
            print(f"[WARNING] Could not load memory: {e}")
            conversation_history = []
    else:
        conversation_history = []

def save_conversation_memory():
    """Save conversation history to file"""
    try:
        with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARNING] Could not save memory: {e}")

# Load existing memory on startup
load_conversation_memory()

# ── Multi-command processor ──────────────────────────────────────────────────
def extract_all_commands(text):
    """Scan text for ALL commands. Returns list of (type, value) tuples.
    Empty list means no commands found → fall through to LLM."""
    text_lower = text.lower().strip()
    commands = []

    # Time
    if any(w in text_lower for w in ["time", "clock", "what time"]):
        commands.append(("reply", f"The current time is {datetime.now().strftime('%I:%M %p')}"))

    # Date
    if any(w in text_lower for w in ["date", "today", "what day"]):
        commands.append(("reply", f"Today is {datetime.now().strftime('%B %d, %Y')}"))

    # Clear memory
    if "clear memory" in text_lower or "forget everything" in text_lower:
        commands.append(("special", "CLEAR_MEMORY"))

    # Music / song detection (English + Malayalam)
    music_kw_en = ["song", "music", "singing", "sing", "play", "what song", "what music",
                   "identify song", "name that song", "shazam", "recognize song", "find song",
                   "search song", "what is this song", "which song", "song name"]
    music_kw_ml = ["പാട്ട്", "ഗാനം", "സംഗീതം", "എന്ത് പാട്ട്", "പാട്ട് പേര്", "ഗാനം പേര്",
                   "ഈ പാട്ട്", "ഏത് പാട്ട്", "പാട്ട് കണ്ടെത്തുക", "ഗാനം കണ്ടെത്തുക"]
    if any(k in text_lower for k in music_kw_en) or any(k in text for k in music_kw_ml):
        commands.append(("special", "MUSIC_DETECTION_REQUESTED"))

    # Wake word
    if "talk to me" in text_lower:
        commands.append(("reply", "I'm listening! How can I help you?"))

    # App launch commands
    app_map = {
        "notepad":       ["notepad", "open notepad", "launch notepad", "start notepad"],
        "calculator":    ["calculator", "calc", "open calculator", "launch calculator"],
        "file explorer": ["file explorer", "explorer", "open file explorer", "open files"],
        "browser":       ["browser", "open browser", "chrome", "firefox", "open chrome"],
        "terminal":      ["terminal", "cmd", "command prompt", "open terminal"],
    }
    for app, keywords in app_map.items():
        if any(k in text_lower for k in keywords):
            commands.append(("app", app))

    return commands  # empty → no commands detected


def handle_all_commands(commands, user_text=""):
    """Execute every detected command and return a single combined response."""
    global conversation_history
    parts = []
    for cmd_type, cmd_value in commands:
        if cmd_type == "reply":
            parts.append(cmd_value)
        elif cmd_type == "special":
            if cmd_value == "CLEAR_MEMORY":
                conversation_history = []
                save_conversation_memory()
                parts.append("Memory cleared!")
            elif cmd_value == "MUSIC_DETECTION_REQUESTED":
                music_reply = call_llm(
                    f"user: {user_text}\nassistant:",
                    tokens=80,
                    system_prompt=(
                        "The user is asking about music or a song. "
                        "Suggest Shazam or Google Sound Search if they want to identify a song. "
                        "Be brief (1-2 sentences)."
                    )
                )
                parts.append(music_reply)
        elif cmd_type == "app":
            parts.append(launch_application(cmd_value, user_text))
    return "  ".join(parts)

def launch_application(app_name, user_text=""):
    """Launch applications based on command"""
    system = platform.system().lower()
    
    app_commands = {
        "notepad": {
            "windows": "notepad.exe",
            "linux": "gedit",
            "darwin": "open -a TextEdit"
        },
        "calculator": {
            "windows": "calc.exe",
            "linux": "gnome-calculator",
            "darwin": "open -a Calculator"
        },
        "file explorer": {
            "windows": "explorer.exe",
            "linux": "nautilus",
            "darwin": "open ."
        },
        "browser": {
            "windows": "start chrome",
            "linux": "xdg-open https://www.google.com",
            "darwin": "open -a Safari"
        },
        "terminal": {
            "windows": "cmd.exe",
            "linux": "gnome-terminal",
            "darwin": "open -a Terminal"
        }
    }
    
    if app_name not in app_commands:
        return f"I don't know how to open {app_name}. Available apps: notepad, calculator, file explorer, browser, terminal."
    
    try:
        command = app_commands[app_name].get(system)
        if not command:
            # Fallback for unsupported OS
            return f"Opening {app_name} is not supported on this operating system."
        
        if system == "windows":
            subprocess.Popen(command, shell=True)
        elif system == "linux":
            # For Linux, try multiple methods
            try:
                # Try direct command first
                subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                # Fallback to xdg-open for some apps
                if app_name == "browser":
                    subprocess.Popen(["xdg-open", "https://www.google.com"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # macOS
            subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"[INFO] Launched application: {app_name}")
        return f"Opening {app_name} for you!"
    except Exception as e:
        print(f"[ERROR] Failed to launch {app_name}: {e}")
        return f"I tried to open {app_name}, but encountered an error. Please try opening it manually."

def call_llm(prompt, tokens=100, system_prompt=None):
    """Call LLM with optimized settings for speed
    Supports multilingual conversations including Malayalam
    """
    try:
        # Enhanced system prompt for multilingual support
        if system_prompt is None:
            system_prompt = (
                "You are JUNO, a voice assistant. Be concise (1-2 sentences max). "
                "Respond in the same language the user uses."
            )
        
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False,
                "keep_alive": "10m",   # Keep model in RAM for 10 minutes
                "options": {
                    "num_predict": tokens,
                    "temperature": 0.5,
                    "top_k": 20,
                    "top_p": 0.85,
                    "num_ctx": 512,   # Small context = fast
                    "num_thread": 8   # Use all CPU threads
                }
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"LLM Error: {response.status_code}. Make sure Ollama is running."
    except requests.exceptions.Timeout:
        return "[WARNING]️ LLM timeout. First request may be slow while loading model. Try again."
    except requests.exceptions.ConnectionError:
        return "[WARNING]️ Cannot connect to Ollama. Please start it with: 'ollama serve'"
    except Exception as e:
        return f"[WARNING]️ LLM Error: {str(e)}"

def transcribe_audio(audio_path, language=None):
    """
    Convert audio file to text using faster_whisper with optimizations
    Supports multiple languages including Malayalam
    """
    if not STT_AVAILABLE:
        return None, "STT not available. Install faster_whisper."
    
    try:
        # Language codes: None = auto-detect, "ml" = Malayalam, "en" = English
        # Whisper supports: en, ml, hi, ta, te, kn, and many more
        transcribe_params = {
            "beam_size": 1,  # Greedy - fastest transcription
            "vad_filter": True,  # Filter out non-speech
            "vad_parameters": dict(
                min_silence_duration_ms=300,  # Faster VAD
                threshold=0.5  # More sensitive voice detection
            ),
            "condition_on_previous_text": True,  # Use context for better accuracy
            "temperature": 0.0  # Deterministic output (most accurate)
        }
        
        # Add language if specified, otherwise auto-detect
        if language:
            transcribe_params["language"] = language
            print(f"[INFO] Transcribing with language: {language}")
        else:
            print("[INFO] Auto-detecting language (supports English, Malayalam, and more)")
        
        segments, info = stt_model.transcribe(audio_path, **transcribe_params)
        
        # Log detected language
        detected_lang = info.language if hasattr(info, 'language') else 'unknown'
        print(f"[INFO] Detected language: {detected_lang}")
        
        text = ""
        for segment in segments:
            text += segment.text + " "
        
        result = text.strip()
        print(f"[OK] Transcribed: '{result[:50]}...' (length: {len(result)}, lang: {detected_lang})")
        return result, None
    except Exception as e:
        print(f"[ERROR] Transcription error: {str(e)}")
        return None, f"Transcription error: {str(e)}"


def synthesize_speech(text, output_file="response.wav"):
    """
    Convert text to speech using Piper TTS
    """
    if not TTS_AVAILABLE:
        return None, "TTS not available. Install Piper."
    
    try:
        # Use absolute path for file creation
        abs_output_file = os.path.abspath(output_file)
        
        # Delete old file if it exists to prevent playback overlap
        if os.path.exists(abs_output_file):
            try:
                os.remove(abs_output_file)
            except:
                pass  # Continue even if delete fails
        
        result = subprocess.run([
            PIPER_EXE,
            "--model", PIPER_VOICE,
            "--output_file", abs_output_file
        ], input=text.encode(), capture_output=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(abs_output_file):
            # Return just the filename, not full path
            return output_file, None
        else:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            return None, f"TTS error: {error_msg}"
    except subprocess.TimeoutExpired:
        return None, "TTS synthesis timed out"
    except Exception as e:
        return None, f"TTS error: {str(e)}"

# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/auth/verify_frame")
async def verify_frame(file: UploadFile = File(...)):
    """
    Verify owner from uploaded frame image
    """
    if not FACE_AUTH_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"authenticated": False, "message": "Face authentication not available"}
        )
    
    try:
        # Read uploaded image
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"authenticated": False, "confidence": 0, "message": "Invalid image"}
        
        # Verify owner
        is_auth, confidence, message = verify_owner(frame)
        
        # Generate session token if authenticated
        session_token = None
        if is_auth:
            import uuid
            session_token = str(uuid.uuid4())
            authenticated_users[session_token] = {
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence
            }
        
        return {
            "authenticated": is_auth,
            "confidence": round(confidence, 1),
            "message": message,
            "session_token": session_token
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"authenticated": False, "message": f"Error: {str(e)}"}
        )


@app.post("/auth/quick_verify")
async def quick_verify():
    """
    Quick camera capture and verify (for testing)
    """
    if not FACE_AUTH_AVAILABLE:
        return JSONResponse(
            status_code=503,
            content={"authenticated": False, "message": "Face authentication not available"}
        )
    
    try:
        is_auth, confidence, message, frame = capture_and_verify(timeout=3)
        
        # Generate session token if authenticated
        session_token = None
        if is_auth:
            import uuid
            session_token = str(uuid.uuid4())
            authenticated_users[session_token] = {
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence
            }
        
        return {
            "authenticated": is_auth,
            "confidence": round(confidence, 1),
            "message": message,
            "session_token": session_token
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"authenticated": False, "message": f"Error: {str(e)}"}
        )


@app.get("/auth/check_session")
async def check_session(token: str):
    """
    Check if session token is valid
    """
    if token in authenticated_users:
        return {
            "valid": True,
            "authenticated_at": authenticated_users[token]["timestamp"],
            "confidence": authenticated_users[token]["confidence"]
        }
    return {"valid": False}


@app.get("/auth/logout")
async def logout(token: str):
    """
    Logout and invalidate session token
    """
    if token in authenticated_users:
        del authenticated_users[token]
        return {"success": True, "message": "Logged out successfully"}
    return {"success": False, "message": "Invalid token"}


# ==================== CHAT ENDPOINTS ====================

@app.post("/chat")
async def chat(request: dict):
    """Optimized chat with command detection"""
    global conversation_history
    
    prompt = request.get("message", "")
    if not prompt:
        return {"error": "No message provided", "response": None, "audio_path": None}

    # Step 1: Try to match ALL system commands (FAST, no LLM needed)
    commands = extract_all_commands(prompt)

    if commands:
        print(f"[INFO] {len(commands)} command(s) detected in: {prompt}")
        reply = handle_all_commands(commands, prompt)
    else:
        # Step 2: No commands found — use LLM
        conversation_history.append({"role": "user", "content": prompt})

        if len(conversation_history) > MAX_HISTORY:
            conversation_history = conversation_history[-MAX_HISTORY:]

        full_prompt = ""
        for msg in conversation_history:
            full_prompt += f"{msg['role']}: {msg['content']}\n"
        full_prompt += "assistant:"

        reply = call_llm(full_prompt, tokens=50)

        conversation_history.append({"role": "assistant", "content": reply})
        save_conversation_memory()

    # Generate audio if TTS is available
    audio_file = None
    if TTS_AVAILABLE:
        audio_file, _ = synthesize_speech(reply)

    return {"response": reply, "audio_path": audio_file, "command_executed": bool(commands)}


@app.post("/stt")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe audio file to text
    """
    if not STT_AVAILABLE:
        return {"error": "STT not available. Install faster_whisper.", "text": None}
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Transcribe with auto language detection
        text, error = transcribe_audio(tmp_path, language=None)
        
        # Clean up
        os.unlink(tmp_path)
        
        if error:
            return {"error": error, "text": None}
        
        return {"text": text, "error": None}
    except Exception as e:
        return {"error": str(e), "text": None}

@app.post("/tts")
def text_to_speech(text: str):
    """
    Convert text to speech
    """
    if not TTS_AVAILABLE:
        return {"error": "TTS not available. Install Piper.", "audio_file": None}
    
    try:
        output_file, error = synthesize_speech(text)
        
        if error:
            return {"error": error, "audio_file": None}
        
        return {"audio_file": output_file, "error": None}
    except Exception as e:
        return {"error": str(e), "audio_file": None}

@app.post("/voice_chat")
async def voice_chat(file: UploadFile = File(...)):
    """
    Complete voice interaction: STT -> LLM -> TTS
    """
    global conversation_history
    
    if not STT_AVAILABLE:
        print("[ERROR] STT not available")
        return JSONResponse(
            status_code=503,
            content={"error": "STT not available", "transcript": None, "response": None, "audio_path": None}
        )
    
    try:
        print("[INFO] Received audio file, starting transcription...")
        # Step 1: Transcribe audio (handle both WAV and WebM)
        file_ext = ".webm" if file.content_type == "audio/webm" else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        print(f"[INFO] Saved audio to {tmp_path}, transcribing...")
        # Auto-detect language (supports English, Malayalam, and many others)
        user_text, stt_error = transcribe_audio(tmp_path, language=None)
        
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        if stt_error or not user_text:
            print(f"[ERROR] Transcription failed: {stt_error or 'No speech detected'}")
            return JSONResponse(
                status_code=400,
                content={"error": stt_error or "No speech detected", "transcript": None, "response": None, "audio_path": None}
            )
        
        print(f"[INFO] Transcript: {user_text}")
        
        # Step 2: Check for ALL commands (FAST pattern matching, no LLM)
        commands = extract_all_commands(user_text)

        if commands:
            print(f"[INFO] {len(commands)} command(s) detected in: {user_text}")
            reply = handle_all_commands(commands, user_text)
        else:
            # Normal conversation with LLM
            print("[INFO] Processing with LLM...")
            conversation_history.append({"role": "user", "content": user_text})

            if len(conversation_history) > MAX_HISTORY:
                conversation_history = conversation_history[-MAX_HISTORY:]

            full_prompt = ""
            for msg in conversation_history:
                full_prompt += f"{msg['role']}: {msg['content']}\n"
            full_prompt += "assistant:"

            reply = call_llm(full_prompt, tokens=50)
            print(f"[INFO] LLM response: {reply[:100]}...")
            conversation_history.append({"role": "assistant", "content": reply})
            save_conversation_memory()
        
        # Step 3: Convert response to speech
        audio_file = None
        tts_error = None
        
        if TTS_AVAILABLE:
            print("[INFO] Generating speech...")
            audio_file, tts_error = synthesize_speech(reply)
            if tts_error:
                print(f"[WARNING] TTS error: {tts_error}")
            else:
                print(f"[INFO] Audio generated: {audio_file}")
        
        result = {
            "transcript": user_text,
            "response": reply,
            "audio_path": audio_file,
            "tts_error": tts_error
        }
        print(f"[INFO] Returning response: transcript={bool(user_text)}, response={bool(reply)}, audio={bool(audio_file)}")
        return result
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"[ERROR] Exception in voice_chat: {e}")
        print(f"[ERROR] Traceback: {error_trace}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "transcript": None, "response": None, "audio_path": None}
        )

@app.get("/")
def root():
    """Redirect to authentication page"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui/auth.html")

@app.get("/audio/{filename}")
def get_audio(filename: str):
    """Serve audio files from workspace directory"""
    # Look in the current working directory
    file_path = Path.cwd() / filename
    if file_path.exists():
        return FileResponse(str(file_path), media_type="audio/wav")
    # Fallback: try as absolute path
    fallback_path = Path(filename)
    if fallback_path.exists():
        return FileResponse(str(fallback_path), media_type="audio/wav")
    return {"error": f"File not found: {filename}"}

@app.get("/memory")
def get_memory():
    """Get conversation history"""
    return {
        "history": conversation_history,
        "count": len(conversation_history)
    }

@app.post("/memory/clear")
def clear_memory():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    save_conversation_memory()
    return {"message": "Memory cleared", "count": 0}


@app.get("/status")
def status():
    """
    Check system status and available features
    """
    return {
        "status": "running",
        "stt_available": STT_AVAILABLE,
        "tts_available": TTS_AVAILABLE,
        "tts_voice": PIPER_VOICE.split('/')[-1] if TTS_AVAILABLE else None,
        "llm_url": OLLAMA_URL,
        "face_auth_available": FACE_AUTH_AVAILABLE,
        "conversation_history_size": len(conversation_history),
        "authenticated_sessions": len(authenticated_users)
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" JUNO - AI Voice Assistant Server with Face Authentication")
    print("="*60)
    print(f" Face Auth: {'[OK] Available' if FACE_AUTH_AVAILABLE else '[WARNING] Not available'}")
    print(f" STT: {'[OK] Available' if STT_AVAILABLE else '[ERROR] Not available'}")
    print(f" TTS: {'[OK] Available' if TTS_AVAILABLE else '[ERROR] Not available'}")
    print(f" LLM: {OLLAMA_URL}")
    print("="*60)
    print(" Server URLs:")
    print("   • API:    http://localhost:8000")
    print("   • Web UI: http://localhost:8000/ui/index.html")
    print("="*60)
    if FACE_AUTH_AVAILABLE:
        print("\n ℹ️  Face Authentication Enabled:")
        print("   Users must verify their face before using voice assistant")
        print("   Register faces using: python recognition_advanced.py")
    print("\n" + "="*60 + "\n")

    # Preload LLM model into RAM so first request is fast
    def _preload_llm():
        try:
            print(f"[INFO] Preloading {OLLAMA_MODEL} into memory...")
            r = requests.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL, "prompt": "hi", "stream": False,
                "options": {"num_predict": 1}
            }, timeout=60)
            if r.status_code == 200:
                print(f"[OK] {OLLAMA_MODEL} loaded and ready (fast responses expected)")
            else:
                print(f"[WARNING] Preload returned status {r.status_code}")
        except Exception as e:
            print(f"[WARNING] LLM preload failed: {e}")

    import threading
    threading.Thread(target=_preload_llm, daemon=True).start()

    uvicorn.run(app, host="127.0.0.1", port=8000)
