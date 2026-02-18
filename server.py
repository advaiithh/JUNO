from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import requests
import os
import json
import subprocess
import tempfile
from pathlib import Path

try:
    from faster_whisper import WhisperModel
    STT_AVAILABLE = True
    # Use CPU-only to avoid GPU/CUDA issues
    try:
        stt_model = WhisperModel("small", device="cpu", compute_type="int8")
    except:
        # Fallback to even smaller model
        stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")
except ImportError:
    STT_AVAILABLE = False
    print("⚠ Warning: faster_whisper not installed. STT features disabled.")
except Exception as e:
    STT_AVAILABLE = False
    print(f"⚠ Warning: Could not load Whisper model: {e}")

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"
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
    print(f"✓ TTS Voice loaded: {PIPER_VOICE}")
else:
    print(f"⚠ TTS not available - Voice file missing: {PIPER_VOICE}")

conversation_history = []
MAX_HISTORY = 10

def call_llm(prompt, tokens=150):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": tokens}
        }
    )
    return response.json()["response"]

def classify_intent(text):
    classification_prompt = f"""
    You are an intent classifier.
    Return ONLY valid JSON.

    Possible intents:
    - general_query
    - open_notepad
    - unknown

    Format:
    {{
      "intent": "intent_name",
      "action_required": true or false
    }}

    Input: {text}
    """

    result = call_llm(classification_prompt, tokens=80)
    return result

def execute_action(intent):
    if intent == "open_notepad":
        os.system("notepad")
        return "Opening Notepad."
    return None

def transcribe_audio(audio_path):
    """
    Convert audio file to text using faster_whisper
    """
    if not STT_AVAILABLE:
        return None, "STT not available. Install faster_whisper."
    
    try:
        segments, info = stt_model.transcribe(audio_path)
        text = ""
        for segment in segments:
            text += segment.text + " "
        return text.strip(), None
    except Exception as e:
        return None, f"Transcription error: {str(e)}"

def synthesize_speech(text, output_file="response.wav"):
    """
    Convert text to speech using Piper TTS
    """
    if not TTS_AVAILABLE:
        return None, "TTS not available. Install Piper."
    
    try:
        result = subprocess.run([
            PIPER_EXE,
            "--model", PIPER_VOICE,
            "--output_file", output_file
        ], input=text.encode(), capture_output=True, timeout=30)
        
        if result.returncode == 0 and os.path.exists(output_file):
            return output_file, None
        else:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            return None, f"TTS error: {error_msg}"
    except subprocess.TimeoutExpired:
        return None, "TTS synthesis timed out"
    except Exception as e:
        return None, f"TTS error: {str(e)}"

@app.post("/chat")
def chat(prompt: str):

    global conversation_history

    # Step 1: Intent classification
    intent_result = classify_intent(prompt)

    try:
        intent_json = json.loads(intent_result)
        intent = intent_json["intent"]
        action_required = intent_json["action_required"]
    except:
        intent = "general_query"
        action_required = False

    # Step 2: Execute action if required
    if action_required:
        action_response = execute_action(intent)
        return {"reply": action_response}

    # Step 3: Normal conversation
    conversation_history.append({"role": "user", "content": prompt})

    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    full_prompt = ""
    for msg in conversation_history:
        full_prompt += f"{msg['role']}: {msg['content']}\n"

    reply = call_llm(full_prompt)

    conversation_history.append({"role": "assistant", "content": reply})

    return {"reply": reply}

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
        
        # Transcribe
        text, error = transcribe_audio(tmp_path)
        
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
        return {"error": "STT not available", "text": None, "reply": None, "audio_file": None}
    
    try:
        # Step 1: Transcribe audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        user_text, stt_error = transcribe_audio(tmp_path)
        os.unlink(tmp_path)
        
        if stt_error:
            return {"error": stt_error, "text": None, "reply": None, "audio_file": None}
        
        # Step 2: Classify intent and generate response
        intent_result = classify_intent(user_text)
        try:
            intent_json = json.loads(intent_result)
            intent = intent_json["intent"]
            action_required = intent_json["action_required"]
        except:
            intent = "general_query"
            action_required = False
        
        if action_required:
            reply = execute_action(intent)
        else:
            # Normal conversation
            conversation_history.append({"role": "user", "content": user_text})
            
            if len(conversation_history) > MAX_HISTORY:
                conversation_history = conversation_history[-MAX_HISTORY:]
            
            full_prompt = ""
            for msg in conversation_history:
                full_prompt += f"{msg['role']}: {msg['content']}\n"
            
            reply = call_llm(full_prompt)
            conversation_history.append({"role": "assistant", "content": reply})
        
        # Step 3: Convert response to speech
        audio_file = None
        tts_error = None
        
        if TTS_AVAILABLE:
            audio_file, tts_error = synthesize_speech(reply)
        
        return {
            "text": user_text,
            "reply": reply,
            "audio_file": audio_file,
            "tts_error": tts_error
        }
    except Exception as e:
        return {"error": str(e), "text": None, "reply": None, "audio_file": None}

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
        "conversation_history_size": len(conversation_history)
    }
