"""
Download Amy voice with config file
"""
import os
import urllib.request
import json

VOICE_DIR = "piper/voices"
VOICE_CONFIG_FILE = "piper_voice_config.json"

RECOMMENDED_VOICE = {
    "name": "Amy (US English Female - High Quality)",
    "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
    "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    "file": "en_US-amy-medium.onnx",
    "json_file": "en_US-amy-medium.onnx.json"
}

print("\n" + "="*70)
print("DOWNLOADING AMY VOICE WITH CONFIG")
print("="*70)
print(f"\nVoice: {RECOMMENDED_VOICE['name']}")
print()

# Create directory
if not os.path.exists(VOICE_DIR):
    os.makedirs(VOICE_DIR)

# Download ONNX model
onnx_path = os.path.join(VOICE_DIR, RECOMMENDED_VOICE["file"])
if os.path.exists(onnx_path):
    print(f"✓ Model file exists: {RECOMMENDED_VOICE['file']}")
else:
    print(f"📦 Downloading model file...")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, int((downloaded / total_size) * 100))
        bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
        print(f"\r[{bar}] {percent}%", end="", flush=True)
    
    try:
        urllib.request.urlretrieve(RECOMMENDED_VOICE["onnx_url"], onnx_path, show_progress)
        print("\n✓ Model downloaded!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        exit(1)

# Download JSON config
json_path = os.path.join(VOICE_DIR, RECOMMENDED_VOICE["json_file"])
if os.path.exists(json_path):
    print(f"✓ Config file exists: {RECOMMENDED_VOICE['json_file']}")
else:
    print(f"\n📦 Downloading config file...")
    try:
        urllib.request.urlretrieve(RECOMMENDED_VOICE["json_url"], json_path)
        print("✓ Config downloaded!")
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)

# Set as active
config = {"active_voice": RECOMMENDED_VOICE["file"]}
with open(VOICE_CONFIG_FILE, 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n✓ Active voice set to: {RECOMMENDED_VOICE['file']}")
print("\n" + "="*70)
print("✓ SETUP COMPLETE!")
print("="*70)
print("\nFiles downloaded:")
print(f"  • {RECOMMENDED_VOICE['file']}")
print(f"  • {RECOMMENDED_VOICE['json_file']}")
print("\nRestart your server to use the new voice:")
print("  uvicorn server:app --reload")
print("\n" + "="*70 + "\n")
