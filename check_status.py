"""
Quick TTS & STT Status Check - No Model Loading
"""
import os
import sys

print("\n" + "="*60)
print("ROBO BUDDY - TTS & STT STATUS CHECK")
print("="*60 + "\n")

# ===== STT Check =====
print("SPEECH-TO-TEXT (STT) STATUS:")
print("-" * 60)
try:
    from faster_whisper import WhisperModel
    print("✓ faster_whisper library: INSTALLED")
    print("✓ STT: READY TO USE")
    print("  (Model will auto-download on first use)\n")
except ImportError:
    print("✗ faster_whisper library: NOT INSTALLED")
    print("✗ STT: NOT WORKING")
    print("  Fix: pip install faster-whisper\n")

# ===== TTS Check =====
print("TEXT-TO-SPEECH (TTS) STATUS:")
print("-" * 60)

piper_exe = os.path.exists("piper/piper.exe")
piper_voice = os.path.exists("piper/voices/en_US-lessac-medium.onnx")

if piper_exe and piper_voice:
    print("✓ Piper executable: FOUND")
    print("✓ Voice model: FOUND")
    print("✓ TTS: READY TO USE\n")
elif piper_exe:
    print("✓ Piper executable: FOUND")
    print("✗ Voice model: MISSING")
    print("✗ TTS: NOT WORKING")
    print("  Fix: Download voice model from Piper releases\n")
elif piper_voice:
    print("✗ Piper executable: MISSING")
    print("✓ Voice model: FOUND")
    print("✗ TTS: NOT WORKING\n")
else:
    print("✗ Piper executable: MISSING")
    print("✗ Voice model: MISSING")
    print("✗ TTS: NOT WORKING\n")

# ===== Summary =====
print("="*60)
print("SUMMARY:")
print("="*60)
print(f"STT (faster_whisper):     {'✓ READY' if 'from faster_whisper' in dir() else '✗ MISSING'}")
print(f"TTS (Piper):              {'✓ READY' if (piper_exe and piper_voice) else '✗ MISSING'}")
print("\nDOWNLOAD PIPER:")
print("Visit: https://github.com/rhasspy/piper/releases")
print("Extract piper.exe to: piper/piper.exe")
print("Download voice model to: piper/voices/en_US-lessac-medium.onnx")
print("="*60 + "\n")
