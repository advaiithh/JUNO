"""
Quick TTS Test - Verify voice is working
"""
import subprocess
import os

PIPER_EXE = "piper/piper.exe"
PIPER_VOICE = "piper/voices/en_US-amy-medium.onnx"
OUTPUT_FILE = "test_tts_output.wav"

print("\n" + "="*70)
print("TESTING TTS DIRECTLY")
print("="*70)
print(f"\nVoice: {PIPER_VOICE}")
print(f"Output: {OUTPUT_FILE}")
print()

# Check files exist
if not os.path.exists(PIPER_EXE):
    print(f"✗ Piper executable not found: {PIPER_EXE}")
    exit(1)

if not os.path.exists(PIPER_VOICE):
    print(f"✗ Voice model not found: {PIPER_VOICE}")
    exit(1)

json_config = PIPER_VOICE + ".json"
if not os.path.exists(json_config):
    print(f"✗ Voice config not found: {json_config}")
    exit(1)

print("✓ All files present")
print("\n📝 Generating speech...")

text = "Hello, I am Robo Buddy with the new Amy voice. This is a test of the text to speech system."

try:
    result = subprocess.run([
        PIPER_EXE,
        "--model", PIPER_VOICE,
        "--output_file", OUTPUT_FILE
    ], input=text.encode(), capture_output=True, timeout=30)
    
    if result.returncode == 0:
        if os.path.exists(OUTPUT_FILE):
            file_size = os.path.getsize(OUTPUT_FILE)
            print(f"\n✓ SUCCESS!")
            print(f"  Audio file: {OUTPUT_FILE}")
            print(f"  Size: {file_size:,} bytes")
            print(f"\n🔊 Playing audio...")
            
            # Play the audio
            try:
                import winsound
                winsound.PlaySound(OUTPUT_FILE, winsound.SND_FILENAME)
                print("✓ Audio played successfully!")
            except:
                print("⚠ Could not auto-play. Open the file manually.")
            
            print("\n" + "="*70)
            print("TTS IS WORKING! ✓")
            print("="*70)
            print("\nYour server is ready for voice chat!")
            print("Start with: python live_voice_client.py")
            print("="*70 + "\n")
        else:
            print("\n✗ Audio file was not created")
            print(f"Return code: {result.returncode}")
    else:
        print(f"\n✗ Piper failed with code: {result.returncode}")
        print(f"Error output: {result.stderr.decode()}")
        
except subprocess.TimeoutExpired:
    print("\n✗ Piper timed out")
except Exception as e:
    print(f"\n✗ Error: {e}")
