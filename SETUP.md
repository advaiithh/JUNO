# 🚀 JUNO Setup Guide
### Complete Installation Instructions for New Devices

---

## 📋 Prerequisites

Before starting, ensure you have:
- **Python 3.8+** installed
- **Git** installed
- **Webcam** (for face authentication)
- **Microphone** (for voice interaction)
- **Ollama** (for AI chat functionality)

---

## 🔧 Step-by-Step Setup

### **Step 1: Clone the Repository**

```bash
git clone https://github.com/advaiithh/JUNO.git
cd JUNO
```

---

### **Step 2: Create Virtual Environment**

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### **Step 3: Install Dependencies**

**Basic Installation (Voice + Face):**
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_face_recognition.txt
```

**If you get errors, install system packages first:**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3-dev cmake build-essential
```

**macOS:**
```bash
brew install cmake
```

---

### **Step 4: Download Models**

#### **4.1 InsightFace Models (For Face Recognition)**
```bash
python setup_face_recognition.py
```

Or manually download:
- Download from: https://github.com/deepinsight/insightface/tree/master/model_zoo
- Place models in project root directory

#### **4.2 Voice Models (For Text-to-Speech)**
```bash
python setup_voice.py
```

Or download manually:
- Piper voices: https://github.com/rhasspy/piper/releases
- Place in `piper/voices/` directory

---

### **Step 5: Install and Setup Ollama (AI Backend)**

#### **Windows:**
1. Download Ollama from: https://ollama.ai/download/windows
2. Install the executable
3. Open PowerShell and run:
```powershell
ollama pull llama3.1:8b
```

#### **Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b
```

#### **macOS:**
```bash
brew install ollama
ollama serve &
ollama pull llama3.1:8b
```

**Verify Ollama is running:**
```bash
ollama list
```
You should see `llama3.1:8b` in the list.

---

### **Step 6: Register Your Face**

Before using the voice assistant, register your face:

```bash
python auto_register_face.py
```

Follow the on-screen instructions:
1. Position your face in the camera frame
2. System will capture 12 samples automatically
3. Wait for "Registration complete" message

---

### **Step 7: Start JUNO**

#### **Easy Way (Recommended):**

**Windows:**
```batch
START_JUNO.bat
```

**Linux/macOS:**
```bash
./start_juno.sh
```

#### **Manual Way:**
```bash
python server.py
```

Server will start on: `http://localhost:8000`

---

## 🎯 Using JUNO

### **Step 1: Face Authentication**
1. Open: `http://localhost:8000/ui/auth.html`
2. Allow camera access when browser asks
3. Position your face in the camera frame
4. System will verify your identity

### **Step 2: Voice Assistant**
1. After authentication, you'll be redirected automatically
2. OR manually open: `http://localhost:8000/ui/index.html`
3. Click the microphone button 🎤
4. **Allow microphone access** when browser asks (important!)
5. Speak your question
6. Click again to stop recording
7. JUNO will respond with voice!

---

## 🔧 Troubleshooting

### **Microphone Not Working**

1. **Check Browser Permissions:**
   - Click 🔒 icon in address bar
   - Set Microphone to "Allow"
   - Refresh page (F5)

2. **Test Microphone:**
   ```
   http://localhost:8000/ui/mic_test.html
   ```

3. **Use Real Browser:**
   - ❌ Don't use VS Code Simple Browser
   - ✅ Use Chrome, Edge, or Firefox

### **Camera Not Working**

```bash
python fix_camera.py
python test_camera_fixed.py
```

### **Face Recognition Issues**

Check if models are downloaded:
```bash
# Should see these files:
# - det_10g.onnx
# - genderage.onnx
# - w600k_r50.onnx
```

Re-download if missing:
```bash
python setup_face_recognition.py
```

### **AI Not Responding**

1. **Check Ollama is running:**
```bash
# Windows PowerShell
Get-Process | Where-Object {$_.ProcessName -like "*ollama*"}

# Linux/macOS
ps aux | grep ollama
```

2. **Start Ollama:**
```bash
ollama serve
```

3. **Test Ollama:**
```bash
ollama run llama3.1:8b "Hello"
```

### **Voice Errors**

```bash
python fix_voice.py
python setup_voice.py
```

### **Server Errors**

Check which features are available:
```bash
curl http://localhost:8000/status
```

---

## 📁 Important Files

- `server.py` - Main server (Face Auth + Voice AI)
- `face_auth.py` - Face authentication logic
- `auto_register_face.py` - Register new faces
- `START_JUNO.bat` - Quick start script (Windows)
- `start_juno.sh` - Quick start script (Linux/macOS)
- `ui/auth.html` - Face authentication UI
- `ui/index.html` - Voice assistant UI

---

## 🔐 Security Notes

- Face authentication runs locally on your device
- Voice data is processed locally (Whisper STT)
- AI responses generated locally (Ollama)
- No data sent to external servers
- Sessions expire after inactivity

---

## 📊 System Requirements

**Minimum:**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 10 GB free space
- Camera: 720p webcam
- Microphone: Any USB/built-in mic

**Recommended:**
- CPU: 8+ cores
- RAM: 16 GB
- Storage: 20 GB free space (for models)
- Camera: 1080p webcam
- Microphone: Good quality USB mic

---

## 🆘 Getting Help

1. **Check Status:**
   ```bash
   python check_status.py
   ```

2. **View Logs:**
   - Check server terminal output
   - Browser console (F12)

3. **Test Individual Components:**
   ```bash
   python test_camera_fixed.py
   python test_tts_direct.py
   ```

4. **Read Documentation:**
   - `FACE_RECOGNITION_GUIDE.md`
   - `QUICK_START_GUIDE.md`
   - `FIX_MICROPHONE.md`

---

## 🎓 Quick Start Summary

```bash
# 1. Clone repo
git clone https://github.com/advaiithh/JUNO.git
cd JUNO

# 2. Setup Python
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements_face_recognition.txt

# 4. Download models
python setup_face_recognition.py
python setup_voice.py

# 5. Setup Ollama
ollama pull llama3.1:8b

# 6. Register your face
python auto_register_face.py

# 7. Start JUNO
python server.py

# 8. Open browser
# Go to: http://localhost:8000/ui/auth.html
```

---

## 🌟 Features

✅ **Face Authentication** - Secure access with face verification  
✅ **Voice Recognition** - Speak naturally to JUNO  
✅ **AI Chat** - Powered by Llama 3.1 (8B)  
✅ **Voice Responses** - Natural TTS output  
✅ **Local Processing** - All data stays on your device  
✅ **Multi-Language** - Supports English, Hindi, Malayalam, and more  

---

**Enjoy using JUNO! 🤖🎤**
