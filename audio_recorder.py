"""
Real-time Audio Recording with Voice Activity Detection
Records audio from microphone and detects when user stops speaking
Uses energy-based silence detection (no external dependencies)
"""
import pyaudio
import wave
import numpy as np
import sys

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration_ms=30):
        """
        Initialize audio recorder with energy-based VAD
        sample_rate: 16000 Hz
        chunk_duration_ms: 30 ms
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        self.recording = False
        self.frames = []
        self.silence_frames = 0
        self.silence_threshold = 8  # Frames of silence to stop recording
        self.energy_threshold = 300  # Threshold for detecting speech
    
    def _get_audio_energy(self, audio_chunk):
        """Calculate energy level of audio chunk"""
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
        energy = np.sqrt(np.sum(audio_data ** 2) / len(audio_data))
        return energy
    
    def _is_speech(self, audio_chunk):
        """Detect if chunk contains speech using energy threshold"""
        energy = self._get_audio_energy(audio_chunk)
        return energy > self.energy_threshold
    
    def record_until_silence(self, timeout_seconds=30, min_duration_ms=500):
        """
        Record audio from microphone until user stops speaking
        
        Args:
            timeout_seconds: Maximum recording time
            min_duration_ms: Minimum audio duration before stopping on silence
        
        Returns:
            audio_path: Path to saved WAV file
        """
        print("\n" + "="*60)
        print("🎤 RECORDING (Speak now...)")
        print("="*60)
        print("Listening... Press Ctrl+C to stop")
        print()
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.frames = []
            self.silence_frames = 0
            frame_count = 0
            speech_started = False
            min_frames = int(min_duration_ms / self.chunk_duration_ms)
            max_frames = int(timeout_seconds * 1000 / self.chunk_duration_ms)
            
            print(f"⏱️  Recording up to {timeout_seconds} seconds...")
            print()
            
            while frame_count < max_frames:
                try:
                    chunk = stream.read(self.chunk_size, exception_on_overflow=False)
                except Exception as e:
                    print(f"\n✗ Error reading audio: {e}")
                    break
                
                frame_count += 1
                self.frames.append(chunk)
                
                # Detect speech using energy
                is_speech = self._is_speech(chunk)
                energy = self._get_audio_energy(chunk)
                
                if is_speech:
                    speech_started = True
                    self.silence_frames = 0
                    print(f"🔴 {frame_count:3d} frames | Energy: {energy:6.0f} 🔊", end="\r")
                else:
                    if speech_started:
                        self.silence_frames += 1
                        print(f"⚪ {frame_count:3d} frames | Silence: {self.silence_frames}/{self.silence_threshold}", end="\r")
                    else:
                        print(f"⚪ {frame_count:3d} frames | Waiting for speech... (Energy: {energy:6.0f})", end="\r")
                
                # Stop if enough silence after speech
                if speech_started and self.silence_frames > self.silence_threshold:
                    if frame_count >= min_frames:
                        print("\n\n✓ Recording complete (silence detected)\n")
                        break
            
            stream.stop_stream()
            stream.close()
            
            # Save to file
            if len(self.frames) == 0:
                print("✗ No audio recorded!")
                return None
            
            audio_file = "recorded_audio.wav"
            self._save_audio(audio_file)
            
            duration = len(self.frames) * self.chunk_duration_ms / 1000
            print(f"✓ Saved: {audio_file}")
            print(f"✓ Duration: {duration:.2f} seconds")
            print(f"✓ Frames: {len(self.frames)}")
            print()
            
            return audio_file
            
        except KeyboardInterrupt:
            print("\n\n⏹️  Recording stopped by user\n")
            
            if len(self.frames) > 0:
                audio_file = "recorded_audio.wav"
                self._save_audio(audio_file)
                print(f"✓ Saved: {audio_file}\n")
                return audio_file
            return None
            
        finally:
            p.terminate()
    
    def _save_audio(self, filename):
        """Save recorded frames to WAV file"""
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b''.join(self.frames))
    
    def record_duration(self, duration_seconds=10):
        """Record for a fixed duration"""
        print(f"\n🎤 Recording for {duration_seconds} seconds...")
        
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            chunks_needed = int(duration_seconds * 1000 / self.chunk_duration_ms)
            
            for i in range(chunks_needed):
                try:
                    chunk = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(chunk)
                    print(f"Recording: {(i+1)/chunks_needed*100:.0f}%", end="\r")
                except Exception as e:
                    print(f"Error: {e}")
                    break
            
            print("\n✓ Recording complete")
            
            # Save
            audio_file = "recorded_audio.wav"
            with wave.open(audio_file, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(b''.join(frames))
            
            print(f"✓ Saved: {audio_file}\n")
            return audio_file
            
        finally:
            p.terminate()


def main():
    """Test the audio recorder"""
    print("\n" + "="*60)
    print("AUDIO RECORDER TEST")
    print("="*60)
    
    recorder = AudioRecorder()
    
    print("\nModes:")
    print("1. Record until silence (smart)")
    print("2. Record for 10 seconds (fixed)")
    
    choice = input("\nSelect mode (1-2): ").strip()
    
    if choice == "1":
        audio_file = recorder.record_until_silence()
    elif choice == "2":
        audio_file = recorder.record_duration(duration_seconds=10)
    else:
        print("Invalid choice")
        return
    
    if audio_file:
        print(f"✓ Recording saved: {audio_file}")
        print("You can now use it with the client:")
        print(f"  python client.py voice {audio_file}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
