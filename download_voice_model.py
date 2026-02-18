"""
Download Piper Voice Model
"""
import urllib.request
import os

print("\n" + "="*60)
print("DOWNLOADING PIPER VOICE MODEL")
print("="*60 + "\n")

voice_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
output_path = "piper/voices/en_US-lessac-medium.onnx"

try:
    print(f"Downloading: {voice_url}")
    print(f"Saving to: {output_path}\n")
    
    # Create progress bar
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, int((downloaded / total_size) * 100))
        bar = '█' * (percent // 2) + '░' * (50 - percent // 2)
        print(f"\r[{bar}] {percent}%", end="", flush=True)
    
    urllib.request.urlretrieve(voice_url, output_path, show_progress)
    print("\n\n✓ Voice model downloaded successfully!")
    
except Exception as e:
    print(f"\n✗ Error downloading: {e}")
    print("\nAlternative: Download manually from:")
    print("https://huggingface.co/rhasspy/piper-voices")
    print(f"Place the .onnx file at: {output_path}")

print("\n" + "="*60 + "\n")
