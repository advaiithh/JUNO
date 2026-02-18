"""
Auto-download best TTS voice (Amy - High Quality Female)
"""
import os
import urllib.request
import json

VOICE_DIR = "piper/voices"
VOICE_CONFIG_FILE = "piper_voice_config.json"

RECOMMENDED_VOICE = {
    "name": "Amy (US English Female - High Quality)",
    "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
    "file": "en_US-amy-medium.onnx"
}

print("\n" + "="*70)
print("DOWNLOADING BEST TTS VOICE")
print("="*70)
print(f"\nVoice: {RECOMMENDED_VOICE['name']}")
print(f"File: {RECOMMENDED_VOICE['file']}")
print()

# Create directory
if not os.path.exists(VOICE_DIR):
    os.makedirs(VOICE_DIR)

output_path = os.path.join(VOICE_DIR, RECOMMENDED_VOICE["file"])

if os.path.exists(output_path):
    print(f"✓ Voice already exists: {RECOMMENDED_VOICE['file']}")
else:
    print("📦 Downloading...")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, int((downloaded / total_size) * 100))
        bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
        print(f"\r[{bar}] {percent}%", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(RECOMMENDED_VOICE["url"], output_path, show_progress)
        print("\n\n✓ Downloaded successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)

# Set as active
config = {"active_voice": RECOMMENDED_VOICE["file"]}
with open(VOICE_CONFIG_FILE, 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Active voice set to: {RECOMMENDED_VOICE['file']}")
print("\n" + "="*70)
print("✓ SETUP COMPLETE!")
print("="*70)
print("\nRestart your server to use the new voice:")
print("  uvicorn server:app --reload")
print("\n" + "="*70 + "\n")
