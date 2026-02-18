# JUNO - AI Voice Assistant with Advanced Face Recognition

A sophisticated AI assistant featuring real-time face recognition with multi-person support, voice interaction capabilities, and memory tracking.

## 🌟 Key Features

### Advanced Face Recognition
- **Multi-Person Support**: Register and recognize multiple people with individual names
- **Multiple Recognition Models**:
  - InsightFace ArcFace-R50 (512-D embeddings) - State-of-the-art accuracy
  - ArcFace with PyTorch (512-D embeddings) - Excellent accuracy
  - Dlib CNN + Large ResNet (128-D encodings) - High accuracy
  - Enhanced OpenCV Multi-Feature (fallback)
- **Real-Time Recognition**: Optimized for smooth camera performance
- **Quality Assessment**: Automatic face quality checking for reliable registration
- **Memory System**: Tracks recognition history and session data

### Voice Capabilities
- **Text-to-Speech (TTS)**: Natural voice synthesis using Piper
- **Speech-to-Text (STT)**: Voice command recognition
- **Live Voice Interaction**: Real-time voice communication
- **Multiple Voice Models**: Support for various voice profiles

### Performance Optimizations
- Frame skipping for smooth camera display
- Reduced processing resolution with upscaling
- Result caching to minimize repeated computations
- Minimal camera buffer latency

## 📋 Requirements

- Python 3.8+
- Webcam
- Windows/Linux/macOS

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/advaiithh/JUNO.git
cd JUNO
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements_face_recognition.txt
```

### 4. Download Models (Optional but Recommended)

For best face recognition accuracy, download InsightFace models:
```bash
python setup_face_recognition.py
```

For voice capabilities:
```bash
python setup_voice.py
```

## 🎯 Usage

### Face Recognition System

Run the advanced face recognition application:
```bash
python recognition_advanced.py
```

#### Main Menu Options:
1. **Register Person Face** - Register a new person with their name
2. **Recognize/Verify Face** - Start real-time face recognition
3. **View Registered People** - List all registered individuals
4. **View Memory Summary** - See recognition statistics
5. **View Recognition History** - Review recent recognition events
6. **Delete Registered Person** - Remove a person from the database
7. **Clear Memory** - Reset all memory/statistics
8. **Exit**

### Voice Interaction

#### Text-to-Speech Testing
```bash
python test_tts_direct.py
```

#### Full Voice Interaction
```bash
python live_voice_client.py
```

#### Start Voice Server
```bash
python server.py
```

## 🎨 Features in Detail

### Multi-Person Recognition

The system supports registering multiple people:
- Each person is identified by name
- 12 high-quality samples captured per person
- Strict multi-layer validation to prevent false positives
- Real-time display of person's name during recognition

### Face Quality Assessment

Automatic quality checking ensures reliable recognition:
- Size validation
- Aspect ratio checking
- Brightness assessment
- Blur detection
- Quality scoring: Excellent, Good, Fair, Poor

### Recognition Memory System

Tracks and logs:
- Recognition sessions with unique IDs
- Success/failure statistics
- Detection timestamps
- Confidence scores
- Face quality metrics

### Performance Features

- **Frame Skipping**: Processes every 3rd frame for smooth 30 FPS display
- **Resolution Scaling**: Detection on 50% scaled frames, upscaled for display
- **Caching**: Recognition results cached between processing cycles
- **Buffer Optimization**: Minimal latency camera buffer

## 📁 Project Structure

```
JUNO/
├── recognition_advanced.py      # Main face recognition system
├── arcface_model.py            # ArcFace PyTorch implementation
├── insightface_onnx.py         # InsightFace ONNX wrapper
├── face_alignment_utils.py     # Face preprocessing utilities
├── face_tracking.py            # Face tracking logic
├── tts.py                      # Text-to-speech engine
├── stt.py                      # Speech-to-text engine
├── server.py                   # Voice server
├── client.py                   # Voice client
├── voice_manager.py            # Voice management
├── memory/                     # Recognition memory storage
├── face_samples/               # Registered face samples
├── piper/                      # Piper TTS models
└── docs/                       # Documentation
    ├── ARCFACE_COMPLETE.md
    ├── FACE_RECOGNITION_GUIDE.md
    ├── QUICK_START_GUIDE.md
    └── PERFORMANCE_GUIDE.py
```

## 🔧 Configuration

### Camera Settings
Edit `camera_config.py` to adjust:
- Resolution (default: 1280x720)
- FPS (default: 30)
- Buffer size

### Recognition Thresholds
Modify in `recognition_advanced.py`:
- `threshold`: Distance threshold for matches
- `min_matches`: Minimum required matching samples
- `max_allowed_distance`: Maximum distance for best match

## 🐛 Troubleshooting

### Camera Issues
```bash
python fix_camera.py
python test_camera_fixed.py
```

### Voice Issues
```bash
python fix_voice.py
```

### Check System Status
```bash
python check_status.py
```

## 📚 Documentation

Detailed guides available in the repository:
- [Face Recognition Guide](FACE_RECOGNITION_GUIDE.md)
- [Quick Start Guide](QUICK_START_GUIDE.md)
- [ArcFace Setup](ARCFACE_COMPLETE.md)
- [InsightFace Setup](INSIGHTFACE_COMPLETE.md)
- [Performance Guide](PERFORMANCE_GUIDE.py)

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Advaith**
- GitHub: [@advaiithh](https://github.com/advaiithh)

## 🙏 Acknowledgments

- InsightFace for state-of-the-art face recognition models
- Dlib for CNN-based face detection
- Piper TTS for natural voice synthesis
- OpenCV for computer vision capabilities

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting guides

---

**Note**: Large model files (ONNX models, PyTorch checkpoints) are not included in the repository. Download them using the setup scripts or manually place them in the appropriate directories.
