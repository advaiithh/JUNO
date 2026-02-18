"""
RoboBuddy Voice API Client
Easy testing and integration with the server
"""
import requests
import os
import sys
from pathlib import Path

SERVER_URL = "http://localhost:8000"

class RoboBuddyClient:
    def __init__(self, server_url=SERVER_URL):
        self.server_url = server_url
    
    def check_status(self):
        """Check if server is running and what features are available"""
        try:
            response = requests.get(f"{self.server_url}/status")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def text_chat(self, prompt):
        """Send a text prompt and get response"""
        try:
            response = requests.post(
                f"{self.server_url}/chat",
                params={"prompt": prompt}
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def text_to_speech(self, text, output_file="response.wav"):
        """Convert text to speech"""
        try:
            response = requests.post(
                f"{self.server_url}/tts",
                params={"text": text}
            )
            result = response.json()
            
            if result.get("audio_file"):
                print(f"✓ Audio file: {result['audio_file']}")
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def speech_to_text(self, audio_file):
        """Transcribe audio file to text"""
        try:
            if not os.path.exists(audio_file):
                return {"error": f"File not found: {audio_file}"}
            
            with open(audio_file, 'rb') as f:
                response = requests.post(
                    f"{self.server_url}/stt",
                    files={"file": f}
                )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def voice_chat(self, audio_file):
        """Complete voice interaction: STT -> LLM -> TTS"""
        try:
            if not os.path.exists(audio_file):
                return {"error": f"File not found: {audio_file}"}
            
            with open(audio_file, 'rb') as f:
                response = requests.post(
                    f"{self.server_url}/voice_chat",
                    files={"file": f}
                )
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def print_status(status):
    """Pretty print status"""
    print("\n" + "="*60)
    print("SERVER STATUS")
    print("="*60)
    if "error" in status:
        print(f"✗ Server Error: {status['error']}")
    else:
        print(f"✓ Status: {status.get('status', 'unknown')}")
        print(f"  STT Available: {'✓' if status.get('stt_available') else '✗'}")
        print(f"  TTS Available: {'✓' if status.get('tts_available') else '✗'}")
        print(f"  Conversation History: {status.get('conversation_history_size', 0)} messages")
    print("="*60 + "\n")


def main():
    client = RoboBuddyClient()
    
    print("\n" + "="*60)
    print("ROBO BUDDY - API CLIENT")
    print("="*60)
    print("\nUsage:")
    print("  python client.py status               # Check server status")
    print("  python client.py chat \"Your prompt\"   # Text chat")
    print("  python client.py tts \"Text to speak\"  # Text to speech")
    print("  python client.py stt audio.wav         # Speech to text")
    print("  python client.py voice audio.wav       # Complete voice chat")
    print("="*60 + "\n")
    
    if len(sys.argv) < 2:
        # Default: check status
        status = client.check_status()
        print_status(status)
        return
    
    command = sys.argv[1].lower()
    
    # Status check
    if command == "status":
        status = client.check_status()
        print_status(status)
    
    # Text chat
    elif command == "chat":
        if len(sys.argv) < 3:
            print("✗ Please provide a prompt")
            print("  Usage: python client.py chat \"Your message\"")
            return
        
        prompt = " ".join(sys.argv[2:])
        print(f"\n📝 Chat: {prompt}")
        result = client.text_chat(prompt)
        
        if "error" in result:
            print(f"✗ Error: {result['error']}")
        else:
            print(f"🤖 Response: {result['reply']}\n")
    
    # Text to speech
    elif command == "tts":
        if len(sys.argv) < 3:
            print("✗ Please provide text")
            print("  Usage: python client.py tts \"Text to speak\"")
            return
        
        text = " ".join(sys.argv[2:])
        print(f"\n🔊 Converting to speech: {text}")
        result = client.text_to_speech(text)
        
        if "error" in result:
            print(f"✗ Error: {result['error']}")
        else:
            print(f"✓ Audio generated: {result.get('audio_file', 'Unknown')}\n")
    
    # Speech to text
    elif command == "stt":
        if len(sys.argv) < 3:
            print("✗ Please provide audio file")
            print("  Usage: python client.py stt audio.wav")
            return
        
        audio_file = sys.argv[2]
        print(f"\n🎤 Transcribing: {audio_file}")
        result = client.speech_to_text(audio_file)
        
        if "error" in result:
            print(f"✗ Error: {result['error']}")
        else:
            print(f"📝 Transcribed: {result.get('text', 'N/A')}\n")
    
    # Voice chat (complete pipeline)
    elif command == "voice":
        if len(sys.argv) < 3:
            print("✗ Please provide audio file")
            print("  Usage: python client.py voice audio.wav")
            return
        
        audio_file = sys.argv[2]
        print(f"\n🎤 Voice Chat: {audio_file}")
        result = client.voice_chat(audio_file)
        
        if "error" in result:
            print(f"✗ Error: {result['error']}")
        else:
            print(f"📝 You: {result.get('text', 'N/A')}")
            print(f"🤖 Robo Buddy: {result.get('reply', 'N/A')}")
            audio_out = result.get('audio_file')
            if audio_out:
                print(f"🔊 Audio saved: {audio_out}")
            if result.get('tts_error'):
                print(f"⚠ TTS Warning: {result['tts_error']}")
            print()
    
    else:
        print(f"✗ Unknown command: {command}")
        print("\nAvailable commands: status, chat, tts, stt, voice")


if __name__ == "__main__":
    main()
