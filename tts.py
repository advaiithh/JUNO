import subprocess

def speak(text):
    subprocess.run([
        "piper/piper.exe",
        "--model", "piper/voices/en_US-lessac-medium.onnx",
        "--output_file", "response.wav"
    ], input=text.encode())
