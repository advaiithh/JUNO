"""
Piper Voice Model Manager
Download and manage multiple high-quality TTS voices
"""
import os
import urllib.request
import json

VOICE_DIR = "piper/voices"
VOICE_CONFIG_FILE = "piper_voice_config.json"

# Best Piper voices: Quality + Speed balance
AVAILABLE_VOICES = {
    "1": {
        "name": "Amy (US English Female - High Quality)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "file": "en_US-amy-medium.onnx",
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "Fast",
        "gender": "Female"
    },
    "2": {
        "name": "Joe (US English Male - High Quality)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/medium/en_US-joe-medium.onnx",
        "file": "en_US-joe-medium.onnx",
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "Fast",
        "gender": "Male"
    },
    "3": {
        "name": "Lessac (US English Male - Current)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "file": "en_US-lessac-medium.onnx",
        "quality": "⭐⭐⭐⭐",
        "speed": "Fast",
        "gender": "Male"
    },
    "4": {
        "name": "Libritts (US English High Quality - Slower)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/libritts/high/en_US-libritts-high.onnx",
        "file": "en_US-libritts-high.onnx",
        "quality": "⭐⭐⭐⭐⭐⭐",
        "speed": "Medium",
        "gender": "Female"
    },
    "5": {
        "name": "Ryan (US English Male - Balanced)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "file": "en_US-ryan-high.onnx",
        "quality": "⭐⭐⭐⭐⭐",
        "speed": "Medium",
        "gender": "Male"
    }
}

def download_voice(voice_info):
    """Download a voice model"""
    url = voice_info["url"]
    output_path = os.path.join(VOICE_DIR, voice_info["file"])
    
    if os.path.exists(output_path):
        print(f"  ✓ Already downloaded: {voice_info['file']}")
        return True
    
    try:
        print(f"  ⏳ Downloading {voice_info['name']}...")
        print(f"     URL: {url}")
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, int((downloaded / total_size) * 100))
            bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
            print(f"\r     [{bar}] {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, show_progress)
        print(f"\n  ✓ Downloaded: {voice_info['file']}\n")
        return True
        
    except Exception as e:
        print(f"\n  ✗ Failed to download: {e}\n")
        return False

def set_active_voice(voice_file):
    """Set the active voice in config"""
    config = {"active_voice": voice_file}
    with open(VOICE_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Active voice set to: {voice_file}")

def get_active_voice():
    """Get the currently active voice"""
    if os.path.exists(VOICE_CONFIG_FILE):
        with open(VOICE_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get("active_voice")
    return None

def list_downloaded_voices():
    """List all downloaded voices"""
    if not os.path.exists(VOICE_DIR):
        return []
    
    voices = []
    for file in os.listdir(VOICE_DIR):
        if file.endswith(".onnx"):
            voices.append(file)
    return voices

def main():
    print("\n" + "="*70)
    print("PIPER VOICE MODEL MANAGER")
    print("="*70)
    
    # Show current voice
    current = get_active_voice()
    print(f"\nCurrent active voice: {current if current else 'None'}")
    
    # Show downloaded voices
    downloaded = list_downloaded_voices()
    if downloaded:
        print(f"\nDownloaded voices: {len(downloaded)}")
        for voice in downloaded:
            status = " (ACTIVE)" if voice == current else ""
            print(f"  • {voice}{status}")
    
    print("\n" + "="*70)
    print("AVAILABLE VOICES")
    print("="*70)
    
    for key, voice in AVAILABLE_VOICES.items():
        status = "✓ Downloaded" if voice["file"] in downloaded else "⬇ Not downloaded"
        active = " (ACTIVE)" if voice["file"] == current else ""
        
        print(f"\n{key}. {voice['name']}{active}")
        print(f"   Quality: {voice['quality']} | Speed: {voice['speed']} | Gender: {voice['gender']}")
        print(f"   File: {voice['file']}")
        print(f"   Status: {status}")
    
    print("\n" + "="*70)
    print("ACTIONS")
    print("="*70)
    print("D <number> - Download voice (e.g., 'D 1' for Amy)")
    print("S <number> - Set as active voice (e.g., 'S 1')")
    print("A - Download all voices")
    print("Q - Quit")
    print("="*70)
    
    while True:
        choice = input("\nEnter command: ").strip().upper()
        
        if choice == 'Q':
            break
        
        elif choice == 'A':
            print("\n📦 Downloading all voices...")
            for key, voice in AVAILABLE_VOICES.items():
                download_voice(voice)
            print("\n✓ All voices downloaded!")
        
        elif choice.startswith('D '):
            num = choice.split()[1]
            if num in AVAILABLE_VOICES:
                download_voice(AVAILABLE_VOICES[num])
            else:
                print("✗ Invalid voice number")
        
        elif choice.startswith('S '):
            num = choice.split()[1]
            if num in AVAILABLE_VOICES:
                voice = AVAILABLE_VOICES[num]
                if voice["file"] in downloaded or download_voice(voice):
                    set_active_voice(voice["file"])
                    print("\n✓ Voice activated! Restart your server for changes to take effect.")
                    print("   Command: uvicorn server:app --reload")
            else:
                print("✗ Invalid voice number")
        
        else:
            print("✗ Invalid command")

if __name__ == "__main__":
    main()
