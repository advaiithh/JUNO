"""
Setup Live Voice Features - Install dependencies
"""
import subprocess
import sys

print("\n" + "="*60)
print("INSTALLING LIVE VOICE DEPENDENCIES")
print("="*60 + "\n")

packages = [
    ("pyaudio", "PyAudio (audio input/output)"),
    ("webrtcvad", "WebRTC VAD (voice activity detection)"),
    ("playsound", "playsound (audio playback)"),
]

failed = []

for package, description in packages:
    print(f"📦 Installing {description}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print(f"   ✓ {package} installed\n")
    except Exception as e:
        print(f"   ✗ Failed to install {package}: {e}\n")
        failed.append(package)

print("="*60)
print("INSTALLATION SUMMARY")
print("="*60)

if not failed:
    print("✓ All dependencies installed successfully!")
    print("\nYou can now use:")
    print("  python live_voice_client.py      ← Live voice chat")
    print("  python audio_recorder.py         ← Record audio")
else:
    print(f"⚠ Failed to install: {', '.join(failed)}")
    print("\nTry installing manually:")
    for pkg in failed:
        print(f"  pip install {pkg}")

print("="*60 + "\n")
