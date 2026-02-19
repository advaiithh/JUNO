# JUNO - Complete AI Voice Assistant 🤖🎤

**The all-in-one AI voice assistant with web UI and standalone terminal version.**

JUNO is a powerful, privacy-focused voice assistant that runs completely on your local machine. No cloud services, no subscriptions, just you and your AI companion.

![JUNO Demo](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Mac-lightgrey)

## 🎯 What is JUNO?

JUNO is a complete voice assistant system featuring:
- 🖥️ **JUNO_COMPLETE.py** - Standalone terminal version, zero dependencies on external servers
- 🌐 **Web UI** - Beautiful browser interface with server.py backend
- 🔒 **100% Local** - All processing happens on your machine
- ⚡ **Fast & Efficient** - Optimized for minimal latency
- 💾 **Conversation Memory** - Persistent context across sessions

## ✨ Key Features

### Terminal Version (JUNO_COMPLETE.py)
- 🎤 **Voice Activity Detection (VAD)** - Automatically detects when you start/stop speaking
- 🗣️ **Speech-to-Text** - Fast Whisper integration
- 🧠 **AI Conversation** - Powered by Ollama LLM (llama3.1:8b)
- 🔊 **Text-to-Speech** - Natural voice with Piper TTS
- 💬 **Conversation Memory** - Remembers your chat history
- ⚡ **Zero Latency Design** - Smart optimizations for instant responses

### Web UI Version
- 🎨 **Beautiful Interface** - Modern gradient design with smooth animations
- 🎤 **Click-to-Talk** - Simple push-to-talk microphone button
- 📊 **Real-time Status** - Visual feedback for recording, processing, and speaking
- 🌐 **Browser-based** - No desktop app needed
- 🔄 **Live Updates** - See transcriptions and responses in real-time

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have:

1. **Python 3.11 or higher** - [Download](https://www.python.org/downloads/)
2. **Ollama** - [Download & Install](https://ollama.ai)
3. **Piper TTS** - [Download Release](https://github.com/rhasspy/piper/releases)
4. **Microphone** - For voice input
5. **Speakers/Headphones** - For audio output

### Installation Guide

#### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/JUNO.git
cd JUNO
```

#### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install fastapi uvicorn python-multipart faster-whisper requests pyaudio numpy
```

#### Step 3: Setup Piper TTS

1. **Download Piper** for your OS:
   - Windows: `piper_windows_amd64.zip`
   - Linux: `piper_linux_x86_64.tar.gz`
   - Mac: `piper_macos_x64.tar.gz`

2. **Extract to `piper/` folder** in project root

3. **Download a voice model**:
   ```bash
   # Example: Download a quality English voice
   wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx
   wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
   ```

4. **Place voice files in `piper/voices/`**

#### Step 4: Install & Configure Ollama

```bash
# Install Ollama first from https://ollama.ai

# Then pull the AI model
ollama pull llama3.1:8b

# Start Ollama service
ollama serve
```

### Running JUNO

#### Option 1: Web UI (Recommended for Beginners)

**Step 1: Start Ollama (in first terminal)**
```bash
ollama serve
```

**Step 2: Start JUNO Server (in second terminal)**
```bash
python -m uvicorn server:app --reload --port 8000
```

Or on Windows, just double-click:
```
start_server.bat
```

**Step 3: Open your browser**
```
http://localhost:8000
```

**That's it!** 🎉 Click the purple microphone and start talking!

---

#### Option 2: Terminal Version (For Advanced Users)

**Step 1: Start Ollama**
```bash
ollama serve
```

**Step 2: Run JUNO_COMPLETE.py**
```bash
python JUNO_COMPLETE.py
```

**Use voice commands:**
- Press ENTER to start recording
- Speak naturally into your microphone
- Press ENTER again to stop (or wait for auto-detection)
- Listen to JUNO's voice response
- Type 'exit' to quit

## 🎯 Usage Examples

### Web UI

1. **Click the purple microphone button** 🎤
2. **Allow microphone access** when browser prompts
3. **Start speaking naturally** - "Hey JUNO, how are you?"
4. **Click again to stop** (or it auto-detects silence)
5. **Watch the status change**: Recording → Processing → Speaking
6. **Listen to JUNO's voice response** 🔊

### Terminal Version (JUNO_COMPLETE.py)

1. **Press ENTER** to start recording
2. **Speak your question** into the microphone
3. **Press ENTER again** or wait for silence detection
4. **See transcription** of what you said
5. **Read AI response** in terminal
6. **Hear voice response** through speakers

### Example Voice Commands

**Conversations:**
```
🗣️ "Hello JUNO, how are you today?"
🗣️ "What's the weather like?"
🗣️ "Tell me an interesting fact about space"
🗣️ "Can you recommend a good book?"
🗣️ "Write a short poem about coding"
```

**Utility Commands:**
```
🗣️ "What time is it?"
🗣️ "What's the date today?"
🗣️ "Set a reminder for 3 PM"
🗣️ "Calculate 234 times 567"
```

**System Commands:**
```
🗣️ "Clear memory" - Resets conversation history
🗣️ "Exit" or "Quit" - Closes JUNO
```

## 📁 Project Structure

```
JUNO/
├── JUNO_COMPLETE.py       # ⭐ Standalone terminal version (all-in-one)
├── server.py              # FastAPI backend for web UI
├── requirements.txt       # Python dependencies
├── start_server.bat       # Windows quick-start script
├── .gitignore             # Git ignore rules
├── README_GITHUB.md       # This file!
├── conversation_memory.json  # Persistent chat history (auto-generated)
│
├── ui/                    # Web interface files
│   ├── index.html         # Main web UI
│   └── redirect.html      # Landing page
│
└── piper/                 # TTS engine (download separately)
    ├── piper.exe          # Piper executable
    ├── espeak-ng-data/    # Phoneme data
    └── voices/            # Voice models (.onnx files)
        ├── en_US-lessac-medium.onnx
        ├── en_US-lessac-medium.onnx.json
        └── ... (add more voices)
```

### Important Files

- **JUNO_COMPLETE.py** - The star of the show! Complete voice assistant in one file
- **server.py** - Web server backend using FastAPI
- **ui/index.html** - Beautiful web interface with purple gradients
- **conversation_memory.json** - Stores your chat history (auto-created)
- **requirements.txt** - All Python packages needed

## 🛠️ Technology Stack

### Core Technologies
- **Language**: Python 3.11+
- **Web Framework**: FastAPI (async web server)
- **Web Server**: Uvicorn (ASGI server)

### AI & ML Models
- **Speech-to-Text**: [faster-whisper](https://github.com/guillaumekln/faster-whisper) (Whisper model)
  - Model: `base` (151MB)
  - Optimized with CTranslate2
  - ~10x faster than OpenAI's original
  
- **Text-to-Speech**: [Piper TTS](https://github.com/rhasspy/piper)
  - Neural text-to-speech
  - High quality, natural voices
  - Fast inference (<1s)
  
- **Language Model**: Ollama
  - Model: llama3.1:8b (default)
  - Alternative: llama2, mistral, etc.
  - 100% local, no cloud API

### Audio Processing
- **Audio I/O**: PyAudio (PortAudio wrapper)
- **VAD**: Custom energy-based voice activity detection
- **Format**: 16kHz, 16-bit, mono WAV
- **Playback**: winsound (Windows) / platform-agnostic options

### Frontend
- **Pure HTML5, CSS3, JavaScript** (no frameworks!)
- **Web Audio API** for microphone access
- **Fetch API** for async server communication
- **Modern CSS** with gradients and animations

### Data Storage
- **Format**: JSON for conversation memory
- **Persistence**: File-based (no database needed)

## ⚙️ Configuration

### Change Voice Model

Edit **server.py** (for web UI):
```python
DEFAULT_VOICE = "piper/voices/en_US-amy-medium.onnx"  # Female voice
# Or
DEFAULT_VOICE = "piper/voices/en_GB-alan-medium.onnx"  # Male British
```

Edit **JUNO_COMPLETE.py** (for terminal version):
```python
class Config:
    PIPER_VOICE = "piper/voices/en_US-lessac-medium.onnx"
```

### Adjust Recording Sensitivity

In **JUNO_COMPLETE.py**:
```python
class Config:
    ENERGY_THRESHOLD = 300  # Lower = more sensitive (100-500)
    SILENCE_THRESHOLD_FRAMES = 8  # Higher = waits longer for silence
    MIN_RECORDING_MS = 500  # Minimum recording duration
    MAX_RECORDING_SECONDS = 30  # Maximum recording duration
```

### Change AI Model

```bash
# List available models
ollama list

# Pull a different model
ollama pull mistral
ollama pull llama2
ollama pull codellama
```

Then edit the config:
```python
class Config:
    OLLAMA_MODEL = "mistral"  # Or any other model
    MAX_TOKENS = 80  # Response length
```

### Adjust Response Speed

```python
class Config:
    WHISPER_MODEL = "tiny"    # Fastest, less accurate
    # WHISPER_MODEL = "base"  # Balanced (default)
    # WHISPER_MODEL = "small" # Slower, more accurate
    # WHISPER_MODEL = "medium" # Even more accurate
```

### Custom System Prompt

In **JUNO_COMPLETE.py**, modify `handle_conversation()`:
```python
system_prompt = "You are JUNO, a helpful AI assistant. Be concise and friendly."
```

## 🐛 Troubleshooting

### Server Issues

**Problem: Server won't start**
```bash
# Check if port 8000 is already in use
netstat -ano | findstr :8000

# Kill the process using the port (Windows)
taskkill /F /PID <process_id>

# Or use a different port
python -m uvicorn server:app --port 8080
```

**Problem: "Module not found" errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Or install individually
pip install fastapi uvicorn python-multipart faster-whisper requests pyaudio numpy
```

### Microphone Issues

**Problem: Microphone not detected**
- Grant browser microphone permission (check browser settings)
- Check Windows sound settings: Settings > System > Sound > Input
- Test microphone in Windows Voice Recorder
- Try a different browser (Chrome/Edge recommended)

**Problem: Microphone too sensitive / not sensitive enough**
- Adjust `ENERGY_THRESHOLD` in config (see Configuration section)
- Check Windows microphone boost settings
- Move closer/farther from microphone

### AI Response Issues

**Problem: No AI responses / "Connection refused"**
```bash
# Make sure Ollama is running
ollama serve

# Check if model is installed
ollama list

# Pull model if missing
ollama pull llama3.1:8b

# Test Ollama directly
ollama run llama3.1:8b "Hello!"
```

**Problem: Responses are too slow**
- Use smaller Whisper model: `WHISPER_MODEL = "tiny"`
- Use faster Ollama model: `ollama pull llama2:7b`
- Close other heavy applications
- Reduce `MAX_TOKENS` for shorter responses

**Problem: Responses are nonsensical**
- Use larger Whisper model: `WHISPER_MODEL = "small"`
- Speak more clearly into microphone
- Check conversation memory isn't corrupted (delete `conversation_memory.json`)

### Audio Playback Issues

**Problem: No audio output / Can't hear JUNO**
- Check speaker volume
- Verify Piper TTS files exist in `piper/` folder
- Test Piper directly: `piper/piper.exe --model piper/voices/en_US-lessac-medium.onnx --output_file test.wav < input.txt`
- Check Windows sound output device

**Problem: Audio is choppy or distorted**
- Close other audio applications
- Try a different voice model
- Check system CPU usage

### Installation Issues

**Problem: PyAudio won't install**
```bash
# Windows - Download wheel file
pip install pipwin
pipwin install pyaudio

# Or download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

**Problem: faster-whisper won't install**
```bash
# Make sure you have Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

pip install faster-whisper --no-cache-dir
```

### Performance Issues

**Problem: High CPU usage**
- Use smaller Whisper model
- Reduce recording sample rate (not recommended)
- Close background applications
- Consider upgrading hardware

**Problem: High memory usage**
- Clear conversation memory: `conversation_memory.json`
- Reduce `MAX_HISTORY` in config
- Restart JUNO periodically

### Common Error Messages

**"No module named 'pyaudio'"**
```bash
pip install pyaudio
```

**"Piper executable not found"**
- Download Piper from releases
- Place `piper.exe` in `piper/` folder
- Check path in config matches actual location

**"Voice model not found"**
- Download voice model (.onnx file)
- Place in `piper/voices/` folder
- Update config with correct filename

**"Ollama connection refused"**
```bash
# Start Ollama service
ollama serve

# Or check if it's running
tasklist | findstr ollama
```

## 📦 Dependencies

```
fastapi>=0.129.0
uvicorn>=0.41.0
python-multipart>=0.0.22
faster-whisper>=1.2.1
requests>=2.32.5
pyaudio>=0.2.14
numpy>=2.4.2
```

## 🎨 Browser Support

| Browser | Support | Notes |
|---------|---------|-------|
| ✅ Google Chrome | Full | **Recommended** - Best performance |
| ✅ Microsoft Edge | Full | **Recommended** - Chromium-based |
| ✅ Firefox | Full | Good support for Web Audio API |
| ⚠️ Safari | Partial | Limited Web Audio API support |
| ❌ Internet Explorer | None | Not supported |

## 📚 FAQ (Frequently Asked Questions)

### General Questions

**Q: Is JUNO free?**  
A: Yes! JUNO is completely free and open-source under MIT license.

**Q: Does JUNO send my data to the cloud?**  
A: No! Everything runs locally on your machine. Your conversations never leave your computer.

**Q: Can I use JUNO offline?**  
A: Yes, after initial setup. All models run locally without internet.

**Q: What languages does JUNO support?**  
A: Currently English. Whisper supports 99+ languages, but implementation needs updates.

**Q: Can I use JUNO commercially?**  
A: Yes! MIT license allows commercial use.

### Technical Questions

**Q: Why use Ollama instead of OpenAI API?**  
A: Privacy, no costs, works offline, and you own your data.

**Q: Can I use a different LLM?**  
A: Yes! Modify the Ollama model or adapt code for other LLMs.

**Q: How accurate is the speech recognition?**  
A: Very accurate! Whisper is state-of-the-art, achieving ~95%+ accuracy.

**Q: Can I add more voices?**  
A: Yes! Download more voice models from [Piper voices](https://huggingface.co/rhasspy/piper-voices/).

**Q: Does it work with CUDA/GPU?**  
A: Yes! faster-whisper auto-detects CUDA. Install `pip install torch` with CUDA support.

**Q: Can I run it on a Raspberry Pi?**  
A: Possible but slow. Use tiny Whisper model and smaller Ollama model.

### Usage Questions

**Q: How do I clear conversation history?**  
A: Say "clear memory" or delete `conversation_memory.json`.

**Q: Can I have multiple conversations?**  
A: Not currently, but you can modify code to support multiple memory files.

**Q: Why are responses sometimes slow?**  
A: Depends on your hardware. Use smaller models for faster responses.

**Q: Can I customize JUNO's personality?**  
A: Yes! Edit the system prompt in the code.

## 🔧 Advanced Features

### Custom Voice Models

Download additional voices from [Hugging Face](https://huggingface.co/rhasspy/piper-voices/):

```bash
# Example: Download a different English voice
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json

# Move to voices folder
mv en_US-amy-medium.* piper/voices/
```

### GPU Acceleration (CUDA)

For faster Whisper inference:

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# faster-whisper will automatically use GPU
```

### Docker Deployment (Coming Soon)

```bash
# Build Docker image
docker build -t juno-assistant .

# Run container
docker run -p 8000:8000 juno-assistant
```

## 🚀 Performance Tips

1. **Use SSD** - Faster model loading
2. **Close background apps** - More RAM for models
3. **Use `tiny` Whisper model** - 10x faster, slight accuracy loss
4. **Reduce MAX_TOKENS** - Shorter AI responses = faster
5. **Use smaller Ollama model** - `llama2:7b` instead of `llama3.1:8b`
6. **Enable GPU** - Massive speedup for Whisper (3-5x faster)
7. **Adjust beam size** - Lower beam size = faster inference

## 📝 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 11+
- **RAM**: 4GB (8GB recommended for smoother performance)
- **CPU**: Intel Core i3 / AMD Ryzen 3 (or equivalent)
- **Storage**: 3GB free space
  - 500MB for dependencies
  - 1GB for Whisper model
  - 500MB for Ollama model
  - 500MB for Piper TTS + voices
  - 500MB for project files
- **Microphone**: Any USB or built-in microphone
- **Internet**: Required for initial setup only

### Recommended Specifications
- **RAM**: 8GB+ for multiple models
- **CPU**: Intel Core i5 / AMD Ryzen 5 or better
- **GPU**: NVIDIA GPU with CUDA (optional, for faster Whisper)
- **Storage**: SSD for faster model loading

### Platform-Specific Notes

**Windows**
- Fully supported
- PyAudio requires Microsoft C++ Build Tools
- winsound built-in for audio playback

**Linux**
- Install PortAudio: `sudo apt-get install portaudio19-dev`
- Install PyAudio: `pip install pyaudio`
- Use `aplay` for audio playback (modify code)

**macOS**
- Install PortAudio: `brew install portaudio`
- Install PyAudio: `pip install pyaudio`
- Use `afplay` for audio playback (modify code)

## 🗺️ Roadmap

### Version 2.0 (Planned)
- [ ] Multi-language support (Spanish, French, German, Chinese)
- [ ] Mobile app (React Native)
- [ ] Wake word detection ("Hey JUNO")
- [ ] Emotion detection in voice
- [ ] Custom voice training
- [ ] Plugin system for extensions

### Version 1.5 (In Progress)
- [ ] Docker container for easy deployment
- [ ] Voice activity detection improvements
- [ ] Multiple conversation threads
- [ ] Export/import conversations
- [ ] Dark/light theme toggle
- [ ] Keyboard shortcuts

### Version 1.0 (Current)
- [x] Terminal version with full functionality
- [x] Web UI with beautiful design
- [x] Speech-to-text with Whisper
- [x] Text-to-speech with Piper
- [x] Conversation memory
- [x] Voice activity detection
- [x] Ollama LLM integration

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/JUNO.git
   cd JUNO
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Test your changes thoroughly

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request**
   - Describe your changes clearly
   - Reference any related issues
   - Wait for review

### Areas We Need Help With

- 🌍 **Translations** - Support for more languages
- 🐛 **Bug Fixes** - Find and fix issues
- 📚 **Documentation** - Improve guides and tutorials
- ✨ **Features** - Implement new capabilities
- 🧪 **Testing** - Write unit and integration tests
- 🎨 **UI/UX** - Improve web interface design
- 🔊 **Voice Models** - Test and recommend good voices
- ⚡ **Performance** - Optimize speed and memory

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Keep discussions professional

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 JUNO Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

You are free to:
- ✅ Use commercially
- ✅ Modify the code
- ✅ Distribute copies
- ✅ Use privately
- ✅ Sublicense

## 🙏 Acknowledgments

### Core Technologies
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Revolutionary speech recognition model
- **[faster-whisper](https://github.com/guillaumekln/faster-whisper)** - Optimized Whisper implementation by Guillaume Klein
- **[Piper TTS](https://github.com/rhasspy/piper)** - High-quality neural text-to-speech by Michael Hansen
- **[Ollama](https://ollama.ai)** - Easy-to-use local LLM runner
- **[FastAPI](https://fastapi.tiangolo.com)** - Modern Python web framework by Sebastián Ramírez

### Libraries & Tools
- **[PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)** - Python audio I/O
- **[NumPy](https://numpy.org/)** - Numerical computing
- **[Uvicorn](https://www.uvicorn.org/)** - Lightning-fast ASGI server

### Inspiration
- Star Trek's computer voice interface
- Jarvis from Iron Man
- Open-source AI community
- Privacy-focused technology advocates

### Special Thanks
- All contributors who help improve JUNO
- Users who provide feedback and bug reports
- The open-source community for amazing tools

## 📞 Support & Community

### Get Help
- 📖 **Documentation**: Read this README thoroughly
- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/JUNO/issues)
- 💡 **Feature Requests**: [Submit an idea](https://github.com/yourusername/JUNO/issues)
- 💬 **Discussions**: [Join the conversation](https://github.com/yourusername/JUNO/discussions)

### Stay Updated
- ⭐ **Star this repo** to follow updates
- 👀 **Watch** for new releases
- 🔔 **Enable notifications** for important changes

### Contact
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com (optional)

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/JUNO?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/JUNO?style=social)
![GitHub issues](https://img.shields.io/github/issues/yourusername/JUNO)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/JUNO)
![GitHub last commit](https://img.shields.io/github/last-commit/yourusername/JUNO)

---

<div align="center">

**Made with ❤️ by the JUNO Team**

*Empowering everyone with AI voice assistance*

[⭐ Star on GitHub](https://github.com/yourusername/JUNO) | [🐛 Report Bug](https://github.com/yourusername/JUNO/issues) | [✨ Request Feature](https://github.com/yourusername/JUNO/issues)

</div>
