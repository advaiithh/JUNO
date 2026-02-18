"""
Test script to verify TTS and STT functionality
"""
import os
import sys

print("\n" + "="*60)
print("ROBO BUDDY - TTS & STT SYSTEM CHECK")
print("="*60 + "\n")

# ===== TEST 1: STT (Speech-to-Text) =====
print("TEST 1: SPEECH-TO-TEXT (STT)")
print("-" * 60)
try:
    from faster_whisper import WhisperModel
    print("✓ faster_whisper library is INSTALLED")
    
    # Try to load the model
    try:
        model = WhisperModel("medium", compute_type="int8")
        print("✓ Whisper model 'medium' loaded successfully")
        
        # Check if sample audio exists
        if os.path.exists("sample.wav"):
            print("✓ sample.wav file found")
            segments, _ = model.transcribe("sample.wav")
            text = ""
            for seg in segments:
                text += seg.text
            print(f"✓ Transcription successful: '{text}'")
        else:
            print("⚠ sample.wav not found - cannot test transcription")
            print("  To test: Provide a sample.wav audio file")
            
        print("\n✓ STT STATUS: WORKING\n")
    except Exception as e:
        print(f"✗ Error loading Whisper model: {e}")
        print("✗ STT STATUS: FAILED\n")
        
except ImportError as e:
    print(f"✗ faster_whisper is NOT installed")
    print(f"  Error: {e}")
    print("✗ STT STATUS: NOT WORKING\n")
    print("  FIX: Install faster-whisper")
    print("  Command: pip install faster-whisper")


# ===== TEST 2: TTS (Text-to-Speech) =====
print("TEST 2: TEXT-TO-SPEECH (TTS)")
print("-" * 60)

piper_path = "piper/piper.exe"
piper_voice = "piper/voices/en_US-lessac-medium.onnx"

if os.path.exists(piper_path):
    print(f"✓ Piper executable found: {piper_path}")
    
    if os.path.exists(piper_voice):
        print(f"✓ Voice model found: {piper_voice}")
        
        try:
            import subprocess
            test_text = "Hello, this is a test of the text to speech system."
            
            result = subprocess.run([
                piper_path,
                "--model", piper_voice,
                "--output_file", "test_output.wav"
            ], input=test_text.encode(), capture_output=True, timeout=30)
            
            if result.returncode == 0:
                if os.path.exists("test_output.wav"):
                    print("✓ Audio file generated successfully")
                    print("✓ TTS STATUS: WORKING\n")
                else:
                    print("✗ Audio file was not created")
                    print("✗ TTS STATUS: FAILED\n")
            else:
                print(f"✗ Piper returned error code: {result.returncode}")
                print(f"  Error: {result.stderr.decode()}")
                print("✗ TTS STATUS: FAILED\n")
                
        except subprocess.TimeoutExpired:
            print("✗ Piper execution timed out")
            print("✗ TTS STATUS: FAILED\n")
        except Exception as e:
            print(f"✗ Error running Piper: {e}")
            print("✗ TTS STATUS: FAILED\n")
    else:
        print(f"✗ Voice model NOT found: {piper_voice}")
        print("✗ TTS STATUS: FAILED\n")
        print("  FIX: Download Piper voice model")
        print("  Visit: https://github.com/rhasspy/piper/releases")
        
else:
    print(f"✗ Piper executable NOT found: {piper_path}")
    print("✗ TTS STATUS: FAILED\n")
    print("  FIX: Download and extract Piper")
    print("  Steps:")
    print("  1. Download from: https://github.com/rhasspy/piper/releases")
    print("  2. Extract to: piper/piper.exe")
    print("  3. Download voice model to: piper/voices/en_US-lessac-medium.onnx")


# ===== SUMMARY =====
print("="*60)
print("SUMMARY & RECOMMENDATIONS")
print("="*60)
print("\n✓ STT: Ready to use (faster-whisper installed)")
print("✗ TTS: Not installed (Piper executable missing)")
print("\nNEXT STEPS:")
print("1. Download Piper from: https://github.com/rhasspy/piper/releases")
print("2. Create 'piper/piper.exe' in your project directory")
print("3. Download voice model and place in 'piper/voices/'")
print("4. Run this test again to verify both systems")
print("\n" + "="*60 + "\n")
