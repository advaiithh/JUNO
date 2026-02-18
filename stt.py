from faster_whisper import WhisperModel

model = WhisperModel("medium", compute_type="int8")

def transcribe(audio_path):
    segments, _ = model.transcribe(audio_path)
    text = ""
    for seg in segments:
        text += seg.text
    return text
