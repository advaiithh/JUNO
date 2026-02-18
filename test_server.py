"""
Test client for RoboBuddy Voice Server
"""
import requests
import subprocess
import json
from pathlib import Path

SERVER_URL = "http://localhost:8000"

def check_server():
    """Check server status"""
    try:
        response = requests.get(f"{SERVER_URL}/status")
        status = response.json()
        print("\n" + "="*60)
        print("SERVER STATUS")
        print("="*60)
        print(f"✓ Server is running")
        print(f"  STT Available: {'✓' if status['stt_available'] else '✗'}")
        print(f"  TTS Available: {'✓' if status['tts_available'] else '✗'}")
        print(f"  Conversation History: {status['conversation_history_size']} messages")
        print("="*60 + "\n")
        return True
    except Exception as e:
        print(f"\n✗ Server not responding: {e}")
        print("  Start the server first: uvicorn server:app --reload")
        return False

def test_text_chat(prompt):
    """Test text-based chat"""
    print(f"\n📝 Text Chat: {prompt}")
    try:
        response = requests.post(f"{SERVER_URL}/chat", params={"prompt": prompt})
        result = response.json()
        print(f"🤖 Response: {result['reply']}\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False

def test_tts(text):
    """Test text-to-speech"""
    print(f"\n🔊 TTS: {text}")
    try:
        response = requests.post(f"{SERVER_URL}/tts", params={"text": text})
        result = response.json()
        
        if result['audio_file']:
            print(f"✓ Audio generated: {result['audio_file']}\n")
            return True
        else:
            print(f"✗ Error: {result['error']}\n")
            return False
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False

def test_voice_chat_with_sample():
    """Test voice chat if sample.wav exists"""
    sample_path = "sample.wav"
    
    if not Path(sample_path).exists():
        print(f"\n⚠ {sample_path} not found. Cannot test voice chat.")
        print("  Create a sample.wav file first.\n")
        return False
    
    print(f"\n🎤 Voice Chat with {sample_path}")
    try:
        with open(sample_path, 'rb') as f:
            response = requests.post(
                f"{SERVER_URL}/voice_chat",
                files={"file": f}
            )
        result = response.json()
        
        print(f"📝 Transcribed: {result.get('text', 'N/A')}")
        print(f"🤖 Response: {result.get('reply', 'N/A')}")
        print(f"🔊 Audio File: {result.get('audio_file', 'N/A')}\n")
        return True
    except Exception as e:
        print(f"✗ Error: {e}\n")
        return False

def main():
    print("\n" + "="*60)
    print("ROBO BUDDY - SERVER TEST CLIENT")
    print("="*60)
    
    # Check server
    if not check_server():
        return
    
    # Test endpoints
    print("\nTesting Endpoints:")
    print("-" * 60)
    
    # 1. Test text chat
    test_text_chat("Hello, how are you?")
    
    # 2. Test TTS
    test_tts("Hello, I am Robo Buddy, your personal assistant.")
    
    # 3. Test voice chat
    test_voice_chat_with_sample()
    
    print("✓ Testing complete!")
    print("\nAPI Documentation:")
    print("  /chat (POST) - Text chat with LLM")
    print("  /stt (POST) - Upload and transcribe audio")
    print("  /tts (POST) - Generate speech from text")
    print("  /voice_chat (POST) - Complete voice interaction")
    print("  /status (GET) - Check system status")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
