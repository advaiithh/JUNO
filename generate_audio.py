"""
Generate test audio files for RoboBuddy testing
"""
import numpy as np
import wave
import struct

def generate_test_audio(filename, duration=3, frequency=440):
    """
    Generate a simple test audio file
    duration: seconds
    frequency: Hz
    """
    sample_rate = 44100
    num_samples = int(sample_rate * duration)
    
    # Generate sine wave
    t = np.linspace(0, duration, num_samples)
    frequency2 = 554  # A5 note
    audio_data = 0.5 * np.sin(2 * np.pi * frequency * t) + 0.5 * np.sin(2 * np.pi * frequency2 * t)
    
    # Convert to 16-bit PCM
    audio_data = np.int16(audio_data * 32767)
    
    # Write WAV file
    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    print(f"✓ Generated: {filename}")

print("\n" + "="*60)
print("GENERATING TEST AUDIO FILES")
print("="*60 + "\n")

generate_test_audio("test_audio.wav", duration=2)
print("\nTest audio files ready!")
print("Use these with the client:")
print("  python client.py stt test_audio.wav")
print("="*60 + "\n")
