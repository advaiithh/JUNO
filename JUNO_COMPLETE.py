"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    JUNO - COMPLETE AI VOICE ASSISTANT                        ║
║                                                                              ║
║  All-in-one voice assistant with:                                           ║
║    ✓ Real-time audio recording with Voice Activity Detection (VAD)          ║
║    ✓ Speech-to-Text (Whisper - fast & accurate)                             ║
║    ✓ AI conversation with memory (Ollama LLM)                               ║
║    ✓ Text-to-Speech (Piper - natural voice)                                 ║
║    ✓ Conversation memory persistence                                        ║
║    ✓ Smart command detection                                               ║
║    ✓ Zero latency optimizations                                             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import wave
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# Audio processing
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠ Warning: pyaudio not installed. Install: pip install pyaudio")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("⚠ Warning: numpy not installed. Install: pip install numpy")

# AI Models
try:
    from faster_whisper import WhisperModel
    STT_AVAILABLE = True
except ImportError:
    STT_AVAILABLE = False
    print("⚠ Warning: faster_whisper not installed. Install: pip install faster-whisper")

# Windows audio playback
try:
    import winsound
    AUDIO_PLAYBACK = True
except ImportError:
    AUDIO_PLAYBACK = False
    print("⚠ Warning: winsound not available (Windows only)")

# HTTP for Ollama
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠ Warning: requests not installed. Install: pip install requests")


# ═══════════════════════════════════════════════════════════════════════════
#                          CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """Centralized configuration"""
    
    # Audio settings
    SAMPLE_RATE = 16000
    CHUNK_DURATION_MS = 30
    SILENCE_THRESHOLD_FRAMES = 8
    ENERGY_THRESHOLD = 300
    MIN_RECORDING_MS = 500
    MAX_RECORDING_SECONDS = 30
    
    # Paths
    PIPER_EXE = "piper/piper.exe"
    PIPER_VOICE = "piper/voices/en_US-lessac-medium.onnx"
    MEMORY_FILE = "conversation_memory.json"
    
    # LLM settings
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3.1:8b"
    MAX_HISTORY = 20
    MAX_TOKENS = 80
    
    # Whisper settings
    WHISPER_MODEL = "base"
    WHISPER_BEAM_SIZE = 5
    
    # Files
    TEMP_AUDIO = "temp_recording.wav"
    RESPONSE_AUDIO = "response.wav"


# ═══════════════════════════════════════════════════════════════════════════
#                       AUDIO RECORDER WITH VAD
# ═══════════════════════════════════════════════════════════════════════════

class AudioRecorder:
    """Real-time audio recording with energy-based Voice Activity Detection"""
    
    def __init__(self):
        if not PYAUDIO_AVAILABLE or not NUMPY_AVAILABLE:
            self.available = False
            return
        
        self.available = True
        self.sample_rate = Config.SAMPLE_RATE
        self.chunk_duration_ms = Config.CHUNK_DURATION_MS
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
        self.silence_threshold = Config.SILENCE_THRESHOLD_FRAMES
        self.energy_threshold = Config.ENERGY_THRESHOLD
        
    def _get_audio_energy(self, audio_chunk):
        """Calculate RMS energy of audio chunk"""
        if not NUMPY_AVAILABLE:
            return 0
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        energy = np.sqrt(np.sum(audio_data ** 2) / len(audio_data))
        return energy
    
    def _is_speech(self, audio_chunk):
        """Detect if chunk contains speech"""
        energy = self._get_audio_energy(audio_chunk)
        return energy > self.energy_threshold
    
    def record_until_silence(self):
        """Record audio until user stops speaking"""
        if not self.available:
            print("\n✗ Audio recording not available (missing pyaudio or numpy)")
            return None
        
        print("\n" + "═"*70)
        print("🎤 RECORDING - Speak now!")
        print("═"*70)
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            silence_frames = 0
            frame_count = 0
            speech_started = False
            
            min_frames = int(Config.MIN_RECORDING_MS / self.chunk_duration_ms)
            max_frames = int(Config.MAX_RECORDING_SECONDS * 1000 / self.chunk_duration_ms)
            
            while frame_count < max_frames:
                try:
                    chunk = stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"\n✗ Audio error: {e}")
                    break
                
                frame_count += 1
                frames.append(chunk)
                
                is_speech = self._is_speech(chunk)
                energy = self._get_audio_energy(chunk)
                
                if is_speech:
                    speech_started = True
                    silence_frames = 0
                    print(f"🔴 Recording... [{frame_count} frames] Energy: {energy:6.0f} 🔊", end="\r")
                else:
                    if speech_started:
                        silence_frames += 1
                        print(f"⏸️  Silence detected... [{silence_frames}/{self.silence_threshold}]  ", end="\r")
                        
                        # Stop if enough silence
                        if silence_frames > self.silence_threshold and frame_count >= min_frames:
                            print("\n✓ Recording complete!")
                            break
                    else:
                        print(f"⚪ Waiting for speech... (Energy: {energy:6.0f})     ", end="\r")
            
            stream.stop_stream()
            stream.close()
            
            if len(frames) == 0:
                print("\n✗ No audio recorded")
                return None
            
            # Save audio
            self._save_audio(frames, Config.TEMP_AUDIO)
            
            duration = len(frames) * self.chunk_duration_ms / 1000
            print(f"✓ Duration: {duration:.2f}s | Frames: {len(frames)}")
            print("═"*70 + "\n")
            
            return Config.TEMP_AUDIO
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Recording stopped by user\n")
            return None
        finally:
            p.terminate()
    
    def _save_audio(self, frames, filename):
        """Save recorded frames to WAV file"""
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b''.join(frames))


# ═══════════════════════════════════════════════════════════════════════════
#                     SPEECH-TO-TEXT (WHISPER)
# ═══════════════════════════════════════════════════════════════════════════

class SpeechToText:
    """Fast and accurate speech transcription using Whisper"""
    
    def __init__(self):
        if not STT_AVAILABLE:
            self.model = None
            return
        
        try:
            print("⏳ Loading Whisper model...")
            self.model = WhisperModel(
                Config.WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
                num_workers=4
            )
            print(f"✓ Whisper model loaded: {Config.WHISPER_MODEL}")
        except Exception as e:
            print(f"✗ Failed to load Whisper: {e}")
            self.model = None
    
    def transcribe(self, audio_path):
        """Convert audio to text"""
        if not self.model:
            return None, "STT not available"
        
        try:
            print("🎧 Transcribing audio...")
            start_time = time.time()
            
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=Config.WHISPER_BEAM_SIZE,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    threshold=0.5
                ),
                condition_on_previous_text=True,
                temperature=0.0
            )
            
            text = ""
            for segment in segments:
                text += segment.text + " "
            
            text = text.strip()
            elapsed = time.time() - start_time
            
            print(f"✓ Transcribed in {elapsed:.2f}s: '{text[:60]}...'")
            return text, None
            
        except Exception as e:
            return None, f"Transcription error: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
#                      TEXT-TO-SPEECH (PIPER)
# ═══════════════════════════════════════════════════════════════════════════

class TextToSpeech:
    """Natural voice synthesis using Piper"""
    
    def __init__(self):
        self.piper_exe = Config.PIPER_EXE
        self.piper_voice = Config.PIPER_VOICE
        self.available = os.path.exists(self.piper_exe) and os.path.exists(self.piper_voice)
        
        if self.available:
            print(f"✓ TTS available: {Path(self.piper_voice).name}")
        else:
            print("⚠ TTS not available - Piper not found")
    
    def synthesize(self, text, output_file=None):
        """Convert text to speech audio file"""
        if not self.available:
            return None, "TTS not available"
        
        if output_file is None:
            output_file = Config.RESPONSE_AUDIO
        
        try:
            # Remove old file
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass
            
            print("🗣️  Synthesizing speech...")
            start_time = time.time()
            
            result = subprocess.run([
                self.piper_exe,
                "--model", self.piper_voice,
                "--output_file", output_file
            ], input=text.encode(), capture_output=True, timeout=30)
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0 and os.path.exists(output_file):
                print(f"✓ Speech synthesized in {elapsed:.2f}s")
                return output_file, None
            else:
                error = result.stderr.decode() if result.stderr else "Unknown error"
                return None, f"TTS error: {error}"
                
        except subprocess.TimeoutExpired:
            return None, "TTS timeout"
        except Exception as e:
            return None, f"TTS error: {str(e)}"
    
    def play_audio(self, audio_file):
        """Play audio file using Windows native playback"""
        if not os.path.exists(audio_file):
            print(f"⚠ Audio file not found: {audio_file}")
            return False
        
        try:
            print(f"🔊 Playing audio...")
            if AUDIO_PLAYBACK:
                winsound.PlaySound(str(audio_file), winsound.SND_FILENAME)
            else:
                os.startfile(audio_file)
            return True
        except Exception as e:
            print(f"⚠ Could not play audio: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
#                    CONVERSATION MEMORY MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class ConversationMemory:
    """Persistent conversation history with automatic saving"""
    
    def __init__(self):
        self.history = []
        self.max_history = Config.MAX_HISTORY
        self.memory_file = Config.MEMORY_FILE
        self.load()
    
    def load(self):
        """Load conversation history from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                print(f"✓ Loaded {len(self.history)} messages from memory")
            except Exception as e:
                print(f"⚠ Could not load memory: {e}")
                self.history = []
        else:
            self.history = []
    
    def save(self):
        """Save conversation history to file"""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ Could not save memory: {e}")
    
    def add_user_message(self, text):
        """Add user message to history"""
        self.history.append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_history()
        self.save()
    
    def add_assistant_message(self, text):
        """Add assistant message to history"""
        self.history.append({
            "role": "assistant",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_history()
        self.save()
    
    def _trim_history(self):
        """Keep only recent messages"""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def build_prompt(self):
        """Build conversation prompt for LLM"""
        prompt = ""
        for msg in self.history:
            prompt += f"{msg['role']}: {msg['content']}\n"
        prompt += "assistant:"
        return prompt
    
    def clear(self):
        """Clear all conversation history"""
        self.history = []
        self.save()
    
    def get_summary(self):
        """Get conversation summary"""
        if not self.history:
            return "No conversation history"
        
        summary = f"Conversation History ({len(self.history)} messages):\n"
        summary += "─"*70 + "\n"
        for msg in self.history[-10:]:  # Show last 10
            role = "You" if msg["role"] == "user" else "JUNO"
            content = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
            summary += f"{role}: {content}\n"
        return summary


# ═══════════════════════════════════════════════════════════════════════════
#                         AI BRAIN (LLM INTEGRATION)
# ═══════════════════════════════════════════════════════════════════════════

class AIBrain:
    """AI reasoning and response generation using Ollama"""
    
    def __init__(self):
        self.ollama_url = Config.OLLAMA_URL
        self.model = Config.OLLAMA_MODEL
        self.check_connection()
    
    def check_connection(self):
        """Check if Ollama is running"""
        if not REQUESTS_AVAILABLE:
            print("⚠ requests library not available - cannot check Ollama")
            return False
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✓ Ollama connected: {self.model}")
                return True
        except:
            pass
        print("⚠ Ollama not connected - voice chat will not work")
        print("  Start Ollama: ollama serve")
        return False
    
    def generate_response(self, prompt):
        """Generate AI response"""
        if not REQUESTS_AVAILABLE:
            return None, "requests library not available"
        
        try:
            print("🧠 Thinking...")
            start_time = time.time()
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": Config.MAX_TOKENS,
                        "temperature": 0.7,
                        "top_k": 40,
                        "top_p": 0.9,
                        "num_ctx": 2048
                    }
                },
                timeout=30
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                reply = response.json()["response"]
                print(f"✓ Response generated in {elapsed:.2f}s")
                return reply, None
            else:
                return None, f"LLM Error: {response.status_code}"
                
        except requests.exceptions.Timeout:
            return None, "LLM timeout (model may be loading, try again)"
        except requests.exceptions.ConnectionError:
            return None, "Cannot connect to Ollama. Start it with: ollama serve"
        except Exception as e:
            return None, f"LLM Error: {str(e)}"
    
    def process_command(self, text):
        """Check if text is a system command"""
        text_lower = text.lower().strip()
        
        # Time commands
        if any(word in text_lower for word in ["time", "clock", "what time"]):
            current_time = datetime.now().strftime("%I:%M %p")
            return True, f"The current time is {current_time}"
        
        # Date commands
        if any(word in text_lower for word in ["date", "today", "what day"]):
            current_date = datetime.now().strftime("%B %d, %Y")
            return True, f"Today is {current_date}"
        
        # Clear memory
        if "clear memory" in text_lower or "forget everything" in text_lower:
            return True, "CLEAR_MEMORY"
        
        return False, None


# ═══════════════════════════════════════════════════════════════════════════
#                      MAIN VOICE ASSISTANT
# ═══════════════════════════════════════════════════════════════════════════

class JunoVoiceAssistant:
    """Complete voice assistant orchestrator"""
    
    def __init__(self):
        print("\n" + "═"*70)
        print("  🚀 INITIALIZING JUNO - AI VOICE ASSISTANT")
        print("═"*70 + "\n")
        
        self.recorder = AudioRecorder()
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.memory = ConversationMemory()
        self.brain = AIBrain()
        self.conversation_count = 0
        
        print("\n" + "═"*70)
        print("  ✓ JUNO READY FOR VOICE INTERACTION")
        print("═"*70 + "\n")
    
    def voice_chat_turn(self):
        """Execute one complete voice interaction turn"""
        
        # Check if recording is available
        if not self.recorder.available:
            print("\n✗ Cannot record audio - pyaudio or numpy not installed")
            print("  Install: pip install pyaudio numpy")
            return False
        
        # Step 1: Record audio
        audio_file = self.recorder.record_until_silence()
        if not audio_file:
            return False
        
        # Step 2: Transcribe
        user_text, stt_error = self.stt.transcribe(audio_file)
        if stt_error or not user_text:
            print(f"✗ {stt_error or 'No speech detected'}")
            return True
        
        print(f"\n📝 You said: \"{user_text}\"")
        
        # Check for exit commands
        if any(word in user_text.lower() for word in ["exit", "quit", "goodbye", "bye bye"]):
            print("\n👋 Goodbye! See you next time!")
            return False
        
        # Step 3: Check for system commands
        is_command, command_result = self.brain.process_command(user_text)
        
        if is_command:
            if command_result == "CLEAR_MEMORY":
                self.memory.clear()
                reply = "Memory cleared. Starting fresh conversation."
                print(f"🤖 JUNO: {reply}")
            else:
                reply = command_result
                print(f"🤖 JUNO: {reply}")
        else:
            # Step 4: Add to memory and generate AI response
            self.memory.add_user_message(user_text)
            prompt = self.memory.build_prompt()
            
            reply, error = self.brain.generate_response(prompt)
            if error:
                print(f"✗ {error}")
                return True
            
            self.memory.add_assistant_message(reply)
            print(f"🤖 JUNO: {reply}")
        
        # Step 5: Synthesize and play response
        audio_file, tts_error = self.tts.synthesize(reply)
        if not tts_error and audio_file:
            self.tts.play_audio(audio_file)
        
        self.conversation_count += 1
        print(f"\n{'─'*70}")
        print(f"✓ Turn {self.conversation_count} complete")
        print("─"*70 + "\n")
        
        # Cleanup
        if os.path.exists(Config.TEMP_AUDIO):
            try:
                os.remove(Config.TEMP_AUDIO)
            except:
                pass
        
        return True
    
    def interactive_mode(self):
        """Run continuous interactive voice chat"""
        print("\n" + "═"*70)
        print("  🎤 INTERACTIVE VOICE CHAT MODE")
        print("═"*70)
        print("\n  Talk naturally. JUNO will respond with voice.")
        print("  Say 'exit', 'quit', or 'goodbye' to end.")
        print("  Press Ctrl+C to stop anytime.")
        print("\n" + "═"*70 + "\n")
        
        try:
            while True:
                continue_chat = self.voice_chat_turn()
                if not continue_chat:
                    break
        except KeyboardInterrupt:
            print("\n\n⏹️  Voice chat stopped by user\n")
        
        print(f"\n✓ Session complete: {self.conversation_count} turns")
        print(self.memory.get_summary())
    
    def show_menu(self):
        """Show interactive menu"""
        print("\n" + "═"*70)
        print("  JUNO - MAIN MENU")
        print("═"*70)
        print("\n  1. 🎤 Interactive Voice Chat (continuous)")
        print("  2. 📝 Single Voice Query")
        print("  3. 📜 View Conversation History")
        print("  4. 🗑️  Clear Memory")
        print("  5. ❌ Exit")
        print("\n" + "═"*70)
        
        choice = input("\n  Select option (1-5): ").strip()
        return choice
    
    def single_query(self):
        """Process a single voice query"""
        print("\n" + "═"*70)
        print("  🎤 SINGLE VOICE QUERY MODE")
        print("═"*70 + "\n")
        
        self.voice_chat_turn()
    
    def show_history(self):
        """Display conversation history"""
        print("\n" + "═"*70)
        print(self.memory.get_summary())
        print("═"*70)
        input("\nPress Enter to continue...")
    
    def run(self):
        """Main application loop"""
        while True:
            choice = self.show_menu()
            
            if choice == "1":
                self.interactive_mode()
            elif choice == "2":
                self.single_query()
            elif choice == "3":
                self.show_history()
            elif choice == "4":
                self.memory.clear()
                print("\n✓ Memory cleared!")
                input("Press Enter to continue...")
            elif choice == "5":
                print("\n👋 Goodbye!\n")
                break
            else:
                print("\n✗ Invalid choice. Try again.")


# ═══════════════════════════════════════════════════════════════════════════
#                            MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def print_banner():
    """Display welcome banner"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║                                                                       ║")
    print("║        ██╗██╗   ██╗███╗   ██╗ ██████╗                                ║")
    print("║        ██║██║   ██║████╗  ██║██╔═══██╗                               ║")
    print("║        ██║██║   ██║██╔██╗ ██║██║   ██║                               ║")
    print("║   ██   ██║██║   ██║██║╚██╗██║██║   ██║                               ║")
    print("║   ╚█████╔╝╚██████╔╝██║ ╚████║╚██████╔╝                               ║")
    print("║    ╚════╝  ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝                                ║")
    print("║                                                                       ║")
    print("║              Complete AI Voice Assistant - All-in-One                ║")
    print("║                                                                       ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")
    print()


def check_dependencies():
    """Check and report system dependencies"""
    print("🔍 Checking system dependencies...\n")
    
    all_good = True
    
    # Check PyAudio
    if PYAUDIO_AVAILABLE:
        print("  ✓ PyAudio installed")
    else:
        print("  ✗ PyAudio NOT installed - Required for recording")
        print("    Install: pip install pyaudio")
        all_good = False
    
    # Check NumPy
    if NUMPY_AVAILABLE:
        print("  ✓ NumPy installed")
    else:
        print("  ✗ NumPy NOT installed")
        print("    Install: pip install numpy")
        all_good = False
    
    # Check faster-whisper
    if STT_AVAILABLE:
        print("  ✓ faster-whisper installed")
    else:
        print("  ✗ faster-whisper NOT installed - Required for speech recognition")
        print("    Install: pip install faster-whisper")
        all_good = False
    
    # Check Piper
    if os.path.exists(Config.PIPER_EXE):
        print(f"  ✓ Piper found: {Config.PIPER_EXE}")
    else:
        print(f"  ✗ Piper NOT found: {Config.PIPER_EXE}")
        print("    Download from: https://github.com/rhasspy/piper/releases")
        all_good = False
    
    # Check Piper voice
    if os.path.exists(Config.PIPER_VOICE):
        print(f"  ✓ Voice model found: {Path(Config.PIPER_VOICE).name}")
    else:
        print(f"  ✗ Voice model NOT found: {Config.PIPER_VOICE}")
        all_good = False
    
    # Check requests
    if REQUESTS_AVAILABLE:
        print("  ✓ requests installed")
    else:
        print("  ✗ requests NOT installed")
        print("    Install: pip install requests")
        all_good = False
    
    print()
    
    if not all_good:
        print("⚠ WARNING: Some dependencies are missing!")
        print("  JUNO may not work properly.\n")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            print("\n❌ Exiting. Please install missing dependencies.\n")
            sys.exit(1)
    else:
        print("✓ All dependencies satisfied!\n")
    
    return all_good


def main():
    """Main application entry point"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠ Running with missing dependencies...\n")
    
    # Create and run assistant
    try:
        assistant = JunoVoiceAssistant()
        assistant.run()
    except KeyboardInterrupt:
        print("\n\n⏹️  JUNO stopped by user. Goodbye! 👋\n")
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
