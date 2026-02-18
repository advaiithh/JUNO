"""
Live Voice Client - Real-time interactive voice chat
Similar to ChatGPT's voice mode
Works with Windows native audio playback
"""
import sys
import os
import time
import requests
from pathlib import Path

# Try to import audio components
try:
    from audio_recorder import AudioRecorder
    RECORDER_AVAILABLE = True
except ImportError:
    RECORDER_AVAILABLE = False

# Windows native audio playback
try:
    import winsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False


class LiveVoiceClient:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
        self.recorder = AudioRecorder() if RECORDER_AVAILABLE else None
        self.conversation_count = 0
    
    def check_connection(self):
        """Check if server is running"""
        try:
            response = requests.get(f"{self.server_url}/status", timeout=2)
            status = response.json()
            return status["status"] == "running"
        except:
            return False
    
    def play_audio(self, audio_file):
        """Play audio file using Windows native sound"""
        if not Path(audio_file).exists():
            print(f"⚠ Audio file not found: {audio_file}")
            return False
        
        try:
            if PLAYSOUND_AVAILABLE:
                # Use winsound (Windows only)
                winsound.PlaySound(audio_file, winsound.SND_FILENAME)
                return True
            else:
                # Fallback: try to open with default player
                print(f"🔊 Playing: {audio_file}")
                os.startfile(audio_file)
                return True
        except Exception as e:
            print(f"⚠ Could not play audio: {e}")
            print(f"   File: {audio_file}")
            return False
    
    def interactive_mode(self):
        """
        Interactive voice chat mode
        1. Record until user stops speaking
        2. Send to server
        3. Get response
        4. Play response
        5. Repeat
        """
        if not self.recorder:
            print("✗ Audio recorder not available. Install: pip install pyaudio")
            return
        
        print("\n" + "="*60)
        print("🎤 LIVE VOICE CHAT - Interactive Mode")
        print("="*60)
        print("Talk naturally. System detects when you stop speaking.")
        print("Commands: 'quit' to exit, 'help' for more")
        print("="*60 + "\n")
        
        while True:
            try:
                # Record until silence
                audio_file = self.recorder.record_until_silence(timeout_seconds=30)
                
                if not audio_file:
                    print("✗ No audio recorded. Try again.\n")
                    continue
                
                # Send to server for voice chat
                print("\n⏳ Processing your request...")
                print("   (This may take 20-60 seconds for CPU processing)")
                result = self._send_voice_chat(audio_file)
                
                if "error" in result and result["error"]:
                    print(f"✗ Error: {result['error']}\n")
                    continue
                
                # Display results
                print("\n" + "-"*60)
                print(f"📝 You: {result.get('text', 'N/A')}")
                print(f"🤖 Robo Buddy: {result.get('reply', 'N/A')}")
                print("-"*60)
                
                # Play response
                audio_response = result.get('audio_file')
                if audio_response:
                    print("\n🔊 Playing response...")
                    self.play_audio(audio_response)
                
                # Increment conversation counter
                self.conversation_count += 1
                print(f"\n✓ Turn {self.conversation_count} complete\n")
                
                # Cleanup
                if Path(audio_file).exists():
                    os.remove(audio_file)
                
            except KeyboardInterrupt:
                print("\n\n✓ Exiting voice chat mode...")
                break
            except Exception as e:
                print(f"✗ Error: {e}\n")
    
    def _send_voice_chat(self, audio_file):
        """Send audio to server for voice chat"""
        try:
            with open(audio_file, 'rb') as f:
                # Increased timeout for CPU-based STT
                response = requests.post(
                    f"{self.server_url}/voice_chat",
                    files={"file": f},
                    timeout=120  # Increased from 60 to 120 seconds
                )
            return response.json()
        except requests.Timeout:
            return {"error": "Request timeout - CPU processing takes longer. Please try again."}
        except Exception as e:
            return {"error": str(e)}
    
    def batch_mode(self, audio_file):
        """Process a single audio file and respond"""
        if not Path(audio_file).exists():
            print(f"✗ File not found: {audio_file}")
            return
        
        print(f"\n📝 Processing: {audio_file}")
        result = self._send_voice_chat(audio_file)
        
        if "error" in result and result["error"]:
            print(f"✗ Error: {result['error']}")
            return
        
        print("\n" + "="*60)
        print(f"📝 You: {result.get('text', 'N/A')}")
        print(f"🤖 Robo Buddy: {result.get('reply', 'N/A')}")
        print("="*60)
        
        if result.get('audio_file'):
            print(f"\n🔊 Audio response: {result['audio_file']}")
            if input("\nPlay response? (y/n): ").lower() == 'y':
                self.play_audio(result['audio_file'])


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*60)
    print("  ROBO BUDDY - LIVE VOICE INTERFACE")
    print("="*60)
    print("\nFeatures:")
    print("  ✓ Real-time audio recording")
    print("  ✓ Automatic speech detection")
    print("  ✓ Live transcription (STT)")
    print("  ✓ AI response generation")
    print("  ✓ Voice synthesis (TTS)")
    print("  ✓ Auto playback")
    print("\n" + "="*60)


def main():
    print_banner()
    
    client = LiveVoiceClient()
    
    # Check server connection
    print("\n⏳ Checking server connection...")
    if not client.check_connection():
        print("✗ Cannot connect to server at http://localhost:8000")
        print("   Start the server first:")
        print("   > uvicorn server:app --reload")
        return
    
    print("✓ Server connected")
    
    # Check dependencies
    if not RECORDER_AVAILABLE:
        print("✗ Audio recorder not available")
        print("  Install: pip install pyaudio")
        return
    
    print("✓ Audio recorder ready")
    
    if not PLAYSOUND_AVAILABLE:
        print("⚠ Audio playback requires: pip install winsound (Windows only)")
        print("   Will fall back to default media player")
    else:
        print("✓ Audio playback ready")
    
    print("\n" + "="*60)
    print("MODE SELECTION")
    print("="*60)
    print("1. Interactive mode (continuous voice chat)")
    print("2. Single query mode (record once, get response)")
    print("3. Process file (send pre-recorded audio)")
    print("4. Exit")
    print("="*60)
    
    choice = input("\nSelect mode (1-4): ").strip()
    
    if choice == "1":
        client.interactive_mode()
    elif choice == "2":
        print("\n🎤 Recording until you stop speaking...")
        audio_file = client.recorder.record_until_silence(timeout_seconds=30)
        if audio_file:
            client.batch_mode(audio_file)
    elif choice == "3":
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
        else:
            audio_file = input("Enter audio file path: ").strip()
        client.batch_mode(audio_file)
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
