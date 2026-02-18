"""
Quick Setup: Download Best TTS Voice
Recommended: Amy (High Quality Female Voice with Fast Processing)
"""
import os
import urllib.request
import json

VOICE_DIR = "piper/voices"
VOICE_CONFIG_FILE = "piper_voice_config.json"

# Recommended voice for best quality + speed
RECOMMENDED_VOICE = {
    "name": "Amy (US English Female - High Quality)",
    "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
    "file": "en_US-amy-medium.onnx",
    "quality": "⭐⭐⭐⭐⭐",
    "speed": "Fast",
    "gender": "Female",
    "description": "Natural, clear female voice with excellent pronunciation and fast synthesis"
}

def download_voice():
    """Download the recommended voice"""
    output_path = os.path.join(VOICE_DIR, RECOMMENDED_VOICE["file"])
    
    if os.path.exists(output_path):
        print(f"✓ Voice already downloaded: {RECOMMENDED_VOICE['file']}")
        return True
    
    try:
        print(f"\n📦 Downloading {RECOMMENDED_VOICE['name']}...")
        print(f"   Quality: {RECOMMENDED_VOICE['quality']}")
        print(f"   Speed: {RECOMMENDED_VOICE['speed']}")
        print(f"   Description: {RECOMMENDED_VOICE['description']}")
        print()
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, int((downloaded / total_size) * 100))
            bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
            print(f"\r   [{bar}] {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(RECOMMENDED_VOICE["url"], output_path, show_progress)
        print(f"\n\n✓ Downloaded successfully!\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to download: {e}\n")
        return False

def set_active_voice():
    """Set as active voice"""
    config = {"active_voice": RECOMMENDED_VOICE["file"]}
    with open(VOICE_CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Active voice set to: {RECOMMENDED_VOICE['file']}")

def main():
    print("\n" + "="*70)
    print("ROBO BUDDY - BEST TTS VOICE SETUP")
    print("="*70)
    print(f"\nRecommended Voice: {RECOMMENDED_VOICE['name']}")
    print(f"Quality: {RECOMMENDED_VOICE['quality']}")
    print(f"Speed: {RECOMMENDED_VOICE['speed']}")
    print(f"Gender: {RECOMMENDED_VOICE['gender']}")
    print(f"\nDescription: {RECOMMENDED_VOICE['description']}")
    print("\n" + "="*70)
    
    proceed = input("\nDownload and activate this voice? (y/n): ").strip().lower()
    
    if proceed == 'y':
        # Create directory if needed
        if not os.path.exists(VOICE_DIR):
            os.makedirs(VOICE_DIR)
        
        # Download
        if download_voice():
            # Set as active
            set_active_voice()
            
            print("\n" + "="*70)
            print("✓ SETUP COMPLETE!")
            print("="*70)
            print("\nNext steps:")
            print("1. Restart your server:")
            print("   > uvicorn server:app --reload")
            print("\n2. Start voice chat:")
            print("   > python live_voice_client.py")
            print("\n3. Enjoy high-quality TTS responses!")
            print("="*70 + "\n")
        else:
            print("\n✗ Setup failed. Please try again or use voice_manager.py")
    else:
        print("\nSetup cancelled.")
        print("Run 'python voice_manager.py' to see all voice options.")

if __name__ == "__main__":
    main()
