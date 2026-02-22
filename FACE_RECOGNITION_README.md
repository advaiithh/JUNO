# 🎭 JUNO Face Recognition Module

Complete guide for JUNO's automatic face authentication system with OpenCV.

---

## 📋 Overview

The face recognition system provides:
- **Automatic face registration** with 15 samples
- **Real-time face authentication** for web UI access
- **Room occupancy tracking** with person detection
- **Secure session management** with token-based auth

---

## 🗂️ Module Files

### Essential Files

| File | Purpose | Required |
|------|---------|----------|
| `face_auth.py` | Core authentication module | ✅ Yes |
| `auto_register_face.py` | Face registration tool | ✅ Yes |
| `face_tracking.py` | Room occupancy tracker | Optional |
| `insightface_onnx.py` | Enhanced face detection | Optional |
| `registered_faces_advanced.pkl` | Stored face data | ✅ Yes |
| `ui/auth.html` | Web authentication page | ✅ Yes |
| `deepsort_utils/` | Person tracking library | Optional |
| `models/person_detection/` | OpenVINO models | Optional |

### Dependencies

```bash
pip install opencv-python numpy scipy openvino
```

---

## 🚀 Quick Start

### 1. Register Your Face

**Automatic Mode (Recommended):**
```powershell
python auto_register_face.py
```

**Process:**
1. Enter your name
2. Press ENTER to start
3. Look at camera (stays centered)
4. System auto-captures 15 samples (every 0.5 seconds)
5. Registration complete!

**Output:**
- Creates `registered_faces_advanced.pkl`
- 15 face samples at 64×64 = 4096 dimensions
- OpenCV-compatible encodings

---

### 2. Run Face Authentication

**A. Web-Based (with JUNO Server):**

```powershell
# Start server (in one terminal)
python server.py

# Open browser to:
http://localhost:8000/ui/auth.html
```

**Authentication Flow:**
1. Camera auto-starts
2. Face verification every 2 seconds
3. When recognized → Auto-login to JUNO
4. If unknown → "Authentication Failed"

**B. Standalone Test:**

```python
from face_auth import verify_owner, capture_and_verify
import cv2

# Test with image
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
is_auth, confidence, message = verify_owner(frame)
print(f"Authenticated: {is_auth}, Confidence: {confidence}%, Message: {message}")
cap.release()

# OR test with automatic camera capture
is_auth, confidence, message, frame = capture_and_verify(timeout=5)
```

---

### 3. Room Occupancy Tracking (Optional)

Track people entering/exiting with OpenVINO person detection:

```powershell
python face_tracking.py
```

**Features:**
- Detects all people (body detection)
- Identifies registered faces
- Tracks entry/exit events
- Shows real-time occupancy count

**Controls:**
- Press **Q** to quit
- Automatic logging to `logs/occupancy_log.json`

---

## 🔧 Technical Details

### Face Encoding Format

- **Method**: OpenCV Haar Cascade + Feature Extraction
- **Size**: 64×64 pixels = 4096 dimensions
- **Type**: Grayscale normalized (0-1 float)
- **Comparison**: Cosine similarity
- **Threshold**: 0.5 (default)

### Authentication Logic

```python
# Matching criteria:
✓ At least 60% of registered samples must match
✓ Minimum distance < threshold (0.5)
✓ Confidence score: 0-100%

# Response format:
{
    "authenticated": bool,
    "confidence": float (0-100),
    "session_token": str,
    "message": str
}
```

### File Structure

```
registered_faces_advanced.pkl = {
    "people": {
        "Your Name": {
            "encodings": [numpy.array (4096,), ...],  # 15 samples
            "registration_date": "2026-02-21 10:30:00",
            "sample_count": 15,
            "method": "opencv_64x64",
            "encoding_size": 4096
        }
    },
    "use_face_recognition": False
}
```

---

## 📖 API Usage

### In server.py

```python
from face_auth import verify_owner

@app.post("/auth/verify_frame")
async def verify_frame(file: UploadFile):
    # Decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Verify face
    is_auth, confidence, message = verify_owner(frame)
    
    if is_auth:
        session_token = generate_token()
        return {
            "authenticated": True,
            "confidence": confidence,
            "session_token": session_token,
            "message": message
        }
    else:
        return {
            "authenticated": False,
            "confidence": 0,
            "message": message
        }
```

---

## 🛠️ Configuration

### Adjust Recognition Threshold

Edit `face_auth.py`:

```python
# Line ~80
def verify_owner(frame, threshold=0.5):  # Change threshold (0.3-0.7)
    # Lower = more strict (fewer false positives)
    # Higher = more lenient (more false positives)
```

### Change Sample Count

Edit `auto_register_face.py`:

```python
# Line ~45
target_samples = 15  # Increase for better accuracy (10-20)
capture_interval = 0.5  # Time between captures
```

---

## 🔍 Troubleshooting

### Issue: "No face detected"
**Solution:**
- Ensure good lighting
- Face camera directly
- Remove glasses/obstructions
- Keep 1-2 feet from camera

### Issue: "Authentication Failed" for registered face
**Solution:**
- Re-register in similar lighting conditions
- Increase sample count to 20
- Adjust threshold to 0.6 (more lenient)
- Check if `registered_faces_advanced.pkl` exists

### Issue: "Encoding size mismatch"
**Solution:**
- Delete `registered_faces_advanced.pkl`
- Re-run `python auto_register_face.py`
- Verify all samples show "4096 dimensions"

### Issue: Camera not opening
**Solution:**
- Check camera permissions
- Close other apps using camera
- Try different camera index: `cv2.VideoCapture(1)`

---

## 📊 Performance

| Operation | Time | Hardware |
|-----------|------|----------|
| Face Detection | ~50ms | CPU (Haar Cascade) |
| Encoding Extraction | ~20ms | CPU (OpenCV) |
| Comparison (15 samples) | ~5ms | CPU (Cosine Similarity) |
| **Total Verification** | **~75ms** | **Per frame** |

**Throughput:** ~13 verifications/second

---

## 🔐 Security Notes

1. **Face data stored locally** - `registered_faces_advanced.pkl`
2. **Session tokens expire** - Server restart clears sessions
3. **No cloud upload** - All processing on-device
4. **Multiple users supported** - Can register multiple people
5. **Adjustable security** - Change threshold based on needs

---

## 🎯 Use Cases

### 1. Personal Voice Assistant
Lock JUNO UI behind face authentication (current implementation)

### 2. Multi-User System
Register family members, track who's using JUNO

### 3. Smart Home
Integrate with room occupancy for automation

### 4. Access Control
Use face verification for secure system access

---

## 📝 Manual Commands

### Register New Person
```powershell
python auto_register_face.py
```

### Test Authentication
```powershell
python -c "from face_auth import capture_and_verify; print(capture_and_verify())"
```

### View Registered People
```powershell
python -c "import pickle; data=pickle.load(open('registered_faces_advanced.pkl','rb')); print('Registered:', list(data['people'].keys()))"
```

### Clear Registration
```powershell
Remove-Item registered_faces_advanced.pkl
```

### Check Encoding Size
```powershell
python -c "import pickle; d=pickle.load(open('registered_faces_advanced.pkl','rb')); print('Samples:', len(list(d['people'].values())[0]['encodings']), 'Size:', len(list(d['people'].values())[0]['encodings'][0]))"
```

---

## 🌟 Advanced Features

### Multiple Person Registration

Edit `auto_register_face.py` to register multiple people by running it multiple times with different names. All stored in same `registered_faces_advanced.pkl`.

### Custom Face Detection

Replace Haar Cascade with DNN-based detection in `face_auth.py`:

```python
# Use OpenCV DNN face detector
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "model.caffemodel")
```

### Integration with Other Systems

```python
# Example: Home automation trigger
from face_auth import verify_owner

def on_face_detected():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    is_auth, confidence, message = verify_owner(frame)
    cap.release()
    
    if is_auth:
        # Trigger actions
        turn_on_lights()
        play_welcome_message()
        load_user_preferences()
```

---

## 📚 References

- **OpenCV Documentation**: https://docs.opencv.org/
- **Face Recognition Basics**: Haar Cascade + Feature Extraction
- **Cosine Similarity**: Distance metric for face comparison
- **OpenVINO**: Intel's computer vision toolkit

---

## 🆘 Support

For issues or questions:
1. Check Troubleshooting section above
2. Verify all dependencies installed: `pip list | grep -E "opencv|numpy|scipy"`
3. Test camera: `python -c "import cv2; cv2.VideoCapture(0).read()"`
4. Review server logs for errors

---

**✅ System Status Check:**

```powershell
# Quick diagnostic
python -c "
import cv2, os
print('OpenCV:', cv2.__version__)
print('Registration file:', 'Found' if os.path.exists('registered_faces_advanced.pkl') else 'Missing')
print('Camera:', 'Working' if cv2.VideoCapture(0).isOpened() else 'Error')
"
```

---

**Made with ❤️ for JUNO AI Voice Assistant**
