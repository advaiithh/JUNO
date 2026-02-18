import cv2
import numpy as np
import os
import pickle
import time
import json
from datetime import datetime
from collections import defaultdict

# Try to import face recognition libraries in order of preference
try:
    import face_recognition
    USE_FACE_RECOGNITION = True
    RECOGNITION_METHOD = "face_recognition"
    print("✓ Using face_recognition library with dlib (Best Accuracy - 128D)")
except ImportError:
    USE_FACE_RECOGNITION = False
    RECOGNITION_METHOD = None

# Try to import InsightFace ONNX if face_recognition not available
if not USE_FACE_RECOGNITION:
    try:
        from insightface_onnx import InsightFaceRecognition
        INSIGHTFACE_MODEL = InsightFaceRecognition()
        
        if INSIGHTFACE_MODEL.load_models():
            USE_INSIGHTFACE = True
            RECOGNITION_METHOD = "insightface"
            print("✓ Using InsightFace ONNX (Excellent Accuracy - 512D)")
            print("  State-of-the-art ArcFace-R50 model")
        else:
            USE_INSIGHTFACE = False
            INSIGHTFACE_MODEL = None
    except Exception as e:
        USE_INSIGHTFACE = False
        INSIGHTFACE_MODEL = None
        print(f"⚠ InsightFace not available: {e}")
else:
    USE_INSIGHTFACE = False
    INSIGHTFACE_MODEL = None

# Try to import ArcFace model if face_recognition and InsightFace not available
if not USE_FACE_RECOGNITION and not USE_INSIGHTFACE:
    try:
        from arcface_model import load_arcface_model, extract_arcface_embedding
        from face_alignment_utils import preprocess_face
        import torch
        
        # Try to load model
        ARCFACE_MODEL_PATH = "checkpoint/backbone.pth"
        ARCFACE_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        ARCFACE_MODEL = load_arcface_model(ARCFACE_MODEL_PATH, ARCFACE_DEVICE)
        
        if ARCFACE_MODEL is not None:
            USE_ARCFACE = True
            RECOGNITION_METHOD = "arcface"
            print(f"✓ Using ArcFace with PyTorch (Excellent Accuracy - 512D) on {ARCFACE_DEVICE}")
        else:
            USE_ARCFACE = False
    except Exception as e:
        USE_ARCFACE = False
        print(f"⚠ ArcFace not available: {e}")
else:
    USE_ARCFACE = False

if RECOGNITION_METHOD is None:
    RECOGNITION_METHOD = "opencv_fallback"
    print("✓ Using Enhanced OpenCV Multi-Feature (Good Accuracy - 800D)")
    print("   For better results: Install InsightFace models or face-recognition")

# File to store registered face data
REGISTERED_FACES_FILE = "registered_faces_advanced.pkl"
FACE_SAMPLES_DIR = "face_samples"
MEMORY_DIR = "memory"
MEMORY_FILE = os.path.join(MEMORY_DIR, "recognition_memory.json")

# OpenCV DNN Face Detection Models (fallback)
DNN_PROTO_PATH = "opencv_face_detector.pbtxt"
DNN_MODEL_PATH = "opencv_face_detector_uint8.pb"
USE_DNN_DETECTOR = False

# Try to download OpenCV DNN models if not available
def initialize_dnn_models():
    """Initialize or download OpenCV DNN face detection models"""
    global USE_DNN_DETECTOR
    
    if os.path.exists(DNN_MODEL_PATH) and os.path.exists(DNN_PROTO_PATH):
        USE_DNN_DETECTOR = True
        return True
    
    # Models not found - will use Haar Cascade
    return False

# Initialize DNN models on startup
initialize_dnn_models()

# Create directories for storing face samples and memory if they don't exist
if not os.path.exists(FACE_SAMPLES_DIR):
    os.makedirs(FACE_SAMPLES_DIR)

if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)


# ==================== MEMORY SYSTEM ====================

class RecognitionMemory:
    """
    Memory system to track face recognition events
    """
    def __init__(self, memory_file=MEMORY_FILE):
        self.memory_file = memory_file
        self.memory_data = self.load_memory()
    
    def load_memory(self):
        """Load memory from JSON file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠ Error loading memory: {e}")
                return self._initialize_memory()
        return self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize empty memory structure"""
        return {
            "total_recognitions": 0,
            "owner_recognitions": 0,
            "unknown_detections": 0,
            "recognition_history": [],
            "sessions": [],
            "last_seen": None,
            "statistics": {
                "average_confidence": 0.0,
                "total_sessions": 0
            }
        }
    
    def save_memory(self):
        """Save memory to JSON file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory_data, f, indent=2)
        except Exception as e:
            print(f"⚠ Error saving memory: {e}")
    
    def add_recognition_event(self, is_owner, confidence, face_quality="good"):
        """Add a recognition event to memory"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        event = {
            "timestamp": timestamp,
            "recognized_as": "OWNER" if is_owner else "UNKNOWN",
            "confidence": round(confidence, 2),
            "face_quality": face_quality
        }
        
        # Update counters
        self.memory_data["total_recognitions"] += 1
        if is_owner:
            self.memory_data["owner_recognitions"] += 1
            self.memory_data["last_seen"] = timestamp
        else:
            self.memory_data["unknown_detections"] += 1
        
        # Add to history (keep last 100 events)
        self.memory_data["recognition_history"].append(event)
        if len(self.memory_data["recognition_history"]) > 100:
            self.memory_data["recognition_history"] = self.memory_data["recognition_history"][-100:]
        
        # Update statistics
        self._update_statistics()
        
        self.save_memory()
    
    def start_session(self):
        """Start a new recognition session"""
        session = {
            "session_id": len(self.memory_data["sessions"]) + 1,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "owner_seen": False,
            "detections": 0
        }
        self.memory_data["sessions"].append(session)
        self.memory_data["statistics"]["total_sessions"] += 1
        self.save_memory()
        return session["session_id"]
    
    def end_session(self, session_id, owner_seen, detections):
        """End the current recognition session"""
        for session in self.memory_data["sessions"]:
            if session["session_id"] == session_id:
                session["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                session["owner_seen"] = owner_seen
                session["detections"] = detections
                break
        self.save_memory()
    
    def _update_statistics(self):
        """Update running statistics"""
        if len(self.memory_data["recognition_history"]) > 0:
            owner_events = [e for e in self.memory_data["recognition_history"] if e["recognized_as"] == "OWNER"]
            if owner_events:
                avg_conf = sum(e["confidence"] for e in owner_events) / len(owner_events)
                self.memory_data["statistics"]["average_confidence"] = round(avg_conf, 2)
    
    def get_summary(self):
        """Get a summary of recognition memory"""
        return f"""
╔══════════════════════════════════════════════════╗
║        RECOGNITION MEMORY SUMMARY                ║
╠══════════════════════════════════════════════════╣
║ Total Recognitions: {self.memory_data['total_recognitions']:>28} ║
║ Owner Recognized:   {self.memory_data['owner_recognitions']:>28} ║
║ Unknown Detected:   {self.memory_data['unknown_detections']:>28} ║
║ Average Confidence: {self.memory_data['statistics']['average_confidence']:>26.1f}% ║
║ Total Sessions:     {self.memory_data['statistics']['total_sessions']:>28} ║
║ Last Seen:          {(self.memory_data['last_seen'] or 'Never')[:27]:>28} ║
╚══════════════════════════════════════════════════╝
"""
    
    def show_recent_history(self, count=10):
        """Show recent recognition history"""
        print("\n" + "="*70)
        print("RECENT RECOGNITION HISTORY")
        print("="*70)
        history = self.memory_data["recognition_history"][-count:]
        if not history:
            print("No history available.")
        else:
            for event in reversed(history):
                status = "✓" if event["recognized_as"] == "OWNER" else "✗"
                print(f"{status} {event['timestamp']} | {event['recognized_as']:8} | "
                      f"Conf: {event['confidence']:5.1f}% | Quality: {event['face_quality']}")
        print("="*70)


def detect_faces_advanced(frame):
    """
    Detect faces using the advanced face_recognition library (dlib CNN-based)
    Falls back to OpenCV cascade detector
    """
    if USE_FACE_RECOGNITION:
        # Convert BGR to RGB for face_recognition library
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use 'cnn' model for BEST accuracy (GPU recommended)
        # Use 'hog' for speed (CPU-friendly)
        # Try CNN first, fall back to HOG if issues
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model='cnn', number_of_times_to_upsample=1)
            if len(face_locations) == 0:
                # Try HOG if CNN didn't find anything
                face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        except:
            # Fall back to HOG if CNN fails (no GPU or model issue)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        return face_locations
    else:
        # Enhanced fallback: Use multiple cascades for better accuracy
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # Use profile cascade as well for side faces
        profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml"
        )
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization for better detection
        gray = cv2.equalizeHist(gray)
        
        # Detect frontal faces with optimized parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05,  # Smaller scale factor for better detection
            minNeighbors=5,     # Lower threshold for more detection
            minSize=(80, 80),   # Slightly smaller minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to (top, right, bottom, left) format
        face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in faces]
        
        return face_locations


def extract_face_encoding(frame, face_location):
    """
    Extract face encoding using the best available method:
    1. face_recognition (dlib) - 128D
    2. InsightFace (ONNX ArcFace-R50) - 512D
    3. ArcFace (PyTorch) - 512D  
    4. OpenCV multi-feature - 800D
    """
    if USE_FACE_RECOGNITION:
        # Method 1: face_recognition library (dlib ResNet)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, [face_location], model='large', num_jitters=2)
        if len(encodings) > 0:
            return encodings[0]
        return None
    
    elif RECOGNITION_METHOD == "insightface" and USE_INSIGHTFACE:
        # Method 2: InsightFace ONNX (ArcFace-R50, state-of-the-art)
        top, right, bottom, left = face_location
        
        # Ensure valid region
        if top < 0 or left < 0 or bottom > frame.shape[0] or right > frame.shape[1]:
            return None
        if (bottom - top) < 20 or (right - left) < 20:
            return None
        
        try:
            # Extract 512-dimensional embedding using InsightFace
            bbox = [left, top, right, bottom]
            embedding = INSIGHTFACE_MODEL.get_face_embedding(frame, bbox)
            return embedding
        except Exception as e:
            print(f"⚠ InsightFace extraction failed: {e}")
            return None
    
    elif RECOGNITION_METHOD == "arcface" and USE_ARCFACE:
        # Method 3: ArcFace PyTorch model
        top, right, bottom, left = face_location
        
        # Ensure valid region
        if top < 0 or left < 0 or bottom > frame.shape[0] or right > frame.shape[1]:
            return None
        if (bottom - top) < 20 or (right - left) < 20:
            return None
        
        face_roi = frame[top:bottom, left:right]
        if face_roi.size == 0:
            return None
        
        try:
            # Preprocess face for ArcFace (align and resize to 112x112)
            preprocessed_face = preprocess_face(face_roi, image_size=112)
            
            # Extract 512-dimensional embedding
            embedding = extract_arcface_embedding(ARCFACE_MODEL, preprocessed_face, ARCFACE_DEVICE)
            return embedding
        except Exception as e:
            print(f"⚠ ArcFace extraction failed: {e}")
            return None
    
    else:
        # Method 4: Enhanced OpenCV fallback - Multi-scale feature extraction
        top, right, bottom, left = face_location
        
        # Ensure valid region
        if top < 0 or left < 0 or bottom > frame.shape[0] or right > frame.shape[1]:
            return None
        if (bottom - top) < 20 or (right - left) < 20:
            return None
        
        face_roi = frame[top:bottom, left:right]
        if face_roi.size == 0:
            return None
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (128, 128))
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better features
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_face = clahe.apply(gray_face)
        
        # Extract multiple feature types
        features = []
        
        # 1. Multi-scale histogram features
        for bins in [32, 64, 128]:
            hist = cv2.calcHist([gray_face], [0], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)
        
        # 2. LBP (Local Binary Patterns) for texture
        lbp = np.zeros_like(gray_face)
        for i in range(1, gray_face.shape[0]-1):
            for j in range(1, gray_face.shape[1]-1):
                center = gray_face[i, j]
                code = 0
                code |= (gray_face[i-1, j-1] > center) << 7
                code |= (gray_face[i-1, j] > center) << 6
                code |= (gray_face[i-1, j+1] > center) << 5
                code |= (gray_face[i, j+1] > center) << 4
                code |= (gray_face[i+1, j+1] > center) << 3
                code |= (gray_face[i+1, j] > center) << 2
                code |= (gray_face[i+1, j-1] > center) << 1
                code |= (gray_face[i, j-1] > center) << 0
                lbp[i, j] = code
        
        lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
        lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
        features.append(lbp_hist)
        
        # 3. HOG features for shape/structure
        hog = cv2.HOGDescriptor((128, 128), (32, 32), (16, 16), (16, 16), 9)
        hog_features = hog.compute(gray_face)
        if hog_features is not None:
            hog_features = cv2.normalize(hog_features, hog_features).flatten()[:256]
            features.append(hog_features)
        
        # 4. Sobel edge features
        sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        edge_hist = np.histogram(edge_magnitude, bins=64, range=(0, 255))[0]
        edge_hist = edge_hist / (np.sum(edge_hist) + 1e-6)
        features.append(edge_hist)
        
        # Combine all features
        combined_features = np.concatenate(features)
        
        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(combined_features)
        if norm > 0:
            combined_features = combined_features / norm
        
        return combined_features


def assess_face_quality(frame, face_location):
    """
    Assess the quality of detected face for reliable recognition
    Returns: quality_score (0-100), quality_label
    """
    top, right, bottom, left = face_location
    face_width = right - left
    face_height = bottom - top
    
    # Size check
    size_score = min(100, (face_width + face_height) / 4)
    
    # Aspect ratio check (face should be roughly square)
    aspect_ratio = face_width / max(face_height, 1)
    aspect_score = 100 if 0.7 < aspect_ratio < 1.3 else 50
    
    # Brightness check
    face_roi = frame[top:bottom, left:right]
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_face)
    brightness_score = 100 if 60 < brightness < 180 else 70 if 40 < brightness < 200 else 30
    
    # Blur detection using Laplacian variance
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
    blur_score = min(100, laplacian_var / 5)
    
    # Overall quality score
    quality_score = (size_score * 0.3 + aspect_score * 0.2 + 
                     brightness_score * 0.2 + blur_score * 0.3)
    
    if quality_score >= 80:
        quality_label = "excellent"
    elif quality_score >= 60:
        quality_label = "good"
    elif quality_score >= 40:
        quality_label = "fair"
    else:
        quality_label = "poor"
    
    return quality_score, quality_label


def register_face_owner():
    """
    Register a person's face using the best available method
    """
    # Ask for person's name
    person_name = input("\nEnter person's name: ").strip()
    if not person_name:
        print("❌ Name cannot be empty!")
        return False
    
    print("\n===ADVANCED FACE REGISTRATION MODE ===")
    print(f"Registering: {person_name}")
    if USE_FACE_RECOGNITION:
        print("✓ Using state-of-the-art dlib ResNet deep learning model (CNN + Large)")
    elif RECOGNITION_METHOD == "insightface":
        print("✓ Using InsightFace ArcFace-R50 model with ONNX (512-D embeddings)")
        print("  State-of-the-art face recognition accuracy!")
    elif RECOGNITION_METHOD == "arcface":
        print("✓ Using ArcFace model with PyTorch (512-D embeddings)")
        print("  State-of-the-art face recognition accuracy!")
    else:
        print("✓ Using enhanced multi-feature extraction (LBP + HOG + Histogram + Edge)")
        print("  For best accuracy: Install InsightFace models")
    print("\nCapturing HIGH-QUALITY face images")
    print("Please look directly at the camera")
    print("Press SPACE to capture, Q to quit, R to reset")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows reliability
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera! Please check your camera connection.")
        return False
    
    # Increase resolution for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    captured_encodings = []
    captured_count = 0
    samples_needed = 12  # More samples for better accuracy
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read from camera!")
            break
        
        # Detect faces
        face_locations = detect_faces_advanced(frame)
        
        # Display instructions
        face_color = (0, 255, 0) if len(face_locations) > 0 else (0, 0, 255)
        cv2.putText(frame, "Look DIRECTLY at camera", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples: {captured_count}/{samples_needed}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE=Capture | Q=Quit | R=Reset", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if USE_FACE_RECOGNITION:
            cv2.putText(frame, "Deep Learning Model (CNN+Large): ACTIVE", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif RECOGNITION_METHOD == "insightface":
            cv2.putText(frame, "InsightFace (ONNX ArcFace-R50 512-D): ACTIVE", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        elif RECOGNITION_METHOD == "arcface":
            cv2.putText(frame, "ArcFace Model (PyTorch 512-D): ACTIVE", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "Multi-Feature Recognition: ACTIVE", (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw rectangles around detected faces with quality info
        for face_location in face_locations:
            top, right, bottom, left = face_location
            
            # Assess face quality
            quality_score, quality_label = assess_face_quality(frame, face_location)
            
            # Color based on quality
            if quality_score >= 70:
                color = (0, 255, 0)  # Green - good quality
            elif quality_score >= 50:
                color = (0, 255, 255)  # Yellow - acceptable
            else:
                color = (0, 0, 255)  # Red - poor quality
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, f"Quality: {quality_label} ({quality_score:.0f})", 
                       (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imshow("Face Registration (Advanced)", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        elif key == ord('r'):
            print("Reset: Clearing all captured faces.")
            captured_encodings = []
            captured_count = 0
        
        elif key == ord(' '):  # SPACE key
            if len(face_locations) > 0:
                # Extract the largest face
                largest_face = max(face_locations, key=lambda f: (f[2]-f[0]) * (f[1]-f[3]))
                
                # Check face quality
                quality_score, quality_label = assess_face_quality(frame, largest_face)
                
                if quality_score < 50:
                    print(f"⚠ Face quality too low ({quality_label}). Please improve lighting/positioning.")
                    continue
                
                # Extract face encoding
                encoding = extract_face_encoding(frame, largest_face)
                
                if encoding is not None:
                    captured_encodings.append(encoding)
                    captured_count += 1
                    
                    print(f"✓ Captured sample {captured_count}/{samples_needed} - Quality: {quality_label} ({quality_score:.0f})")
                    
                    # Check if all samples are captured
                    if captured_count >= samples_needed:
                        print("\n✓ All samples captured successfully!")
                        break
                    else:
                        time.sleep(0.3)  # Small delay between captures
                else:
                    print("⚠ Failed to extract face encoding. Try again.")
            else:
                print("⚠ No face detected! Please look directly at the camera.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Check if we have enough samples
    if len(captured_encodings) < samples_needed:
        print(f"\n⚠ Registration cancelled. Only {len(captured_encodings)} samples captured.")
        return False
    
    # Load existing registered faces if any
    all_registered_people = {}
    if os.path.exists(REGISTERED_FACES_FILE):
        try:
            with open(REGISTERED_FACES_FILE, 'rb') as f:
                all_registered_people = pickle.load(f)
                # Handle old format (convert to new format)
                if "encodings" in all_registered_people and "people" not in all_registered_people:
                    print("Converting old format to new multi-person format...")
                    old_data = all_registered_people
                    all_registered_people = {
                        "people": {
                            "Owner": {
                                "encodings": old_data["encodings"],
                                "registration_date": old_data.get("registration_date", "Unknown"),
                                "sample_count": len(old_data["encodings"])
                            }
                        },
                        "recognition_method": old_data.get("recognition_method", RECOGNITION_METHOD)
                    }
        except:
            all_registered_people = {}
    
    # Initialize structure if empty
    if "people" not in all_registered_people:
        all_registered_people = {
            "people": {},
            "recognition_method": RECOGNITION_METHOD,
            "use_face_recognition": USE_FACE_RECOGNITION
        }
    
    # Add new person
    all_registered_people["people"][person_name] = {
        "encodings": captured_encodings,
        "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_count": len(captured_encodings)
    }
    
    # Save all registered faces
    with open(REGISTERED_FACES_FILE, 'wb') as f:
        pickle.dump(all_registered_people, f)
    
    total_people = len(all_registered_people["people"])
    print(f"\n✓ Registration complete for '{person_name}'! {len(captured_encodings)} encodings saved.")
    print(f"  Total registered people: {total_people}")
    if RECOGNITION_METHOD == "insightface":
        print(f"  Model: InsightFace ArcFace-R50 (ONNX) - 512-D embeddings")
    elif RECOGNITION_METHOD == "arcface":
        print(f"  Model: ArcFace (PyTorch) - 512-D embeddings")
    elif USE_FACE_RECOGNITION:
        print(f"  Model: CNN+Large ResNet (dlib) - 128-D encodings")
    else:
        print(f"  Model: Enhanced Multi-Feature (OpenCV) - 800-D features")
    return True


def recognize_face():
    """
    Recognize and verify faces using deep learning-based face encodings with memory tracking
    """
    # Load registered faces
    if not os.path.exists(REGISTERED_FACES_FILE):
        print("⚠ No registered faces found! Please register first.")
        return
    
    with open(REGISTERED_FACES_FILE, 'rb') as f:
        registered_data = pickle.load(f)
    
    # Handle both old and new format
    if "people" in registered_data:
        # New multi-person format
        all_people = registered_data["people"]
        print(f"\nLoaded {len(all_people)} registered people:")
        for name in all_people.keys():
            print(f"  - {name}")
    else:
        # Old single-person format - convert to new format
        all_people = {
            "Owner": {
                "encodings": registered_data["encodings"]
            }
        }
        print("\nUsing legacy single-person format")
    
    if len(all_people) == 0:
        print("⚠ No registered face data available!")
        return
    
    # Initialize memory system
    memory = RecognitionMemory()
    session_id = memory.start_session()
    
    # Display memory summary
    print(memory.get_summary())
    
    # ADJUSTED thresholds for better detection with multiple people
    # Lower = stricter, Higher = more lenient
    if USE_FACE_RECOGNITION:
        threshold = 0.50  # More lenient for face_recognition
        min_match_ratio = 0.60  # Must match at least 60% of person's samples
        max_allowed_distance = 0.50  # Maximum distance for best match
    elif RECOGNITION_METHOD == "insightface":
        # InsightFace uses cosine similarity
        threshold = 0.40  # More lenient - 0.40 is about 60% similar
        min_match_ratio = 0.58  # Must match 58% (7 out of 12)
        max_allowed_distance = 0.45  # More lenient for best match
    elif RECOGNITION_METHOD == "arcface":
        # ArcFace uses cosine similarity
        threshold = 0.40  # More lenient
        min_match_ratio = 0.58  # Must match 58%
        max_allowed_distance = 0.45  # More lenient for best match
    else:
        # Enhanced multi-feature fallback method
        threshold = 0.25  # More lenient
        min_match_ratio = 0.60  # Must match 60% of person's samples
        max_allowed_distance = 0.35  # More lenient for best match
    
    print("\n=== ADVANCED FACE RECOGNITION MODE WITH MEMORY ===")
    if USE_FACE_RECOGNITION:
        print("✓ Using state-of-the-art dlib CNN + Large ResNet model")
        print("  128-dimensional face encodings - HIGHLY ACCURATE")
    elif RECOGNITION_METHOD == "insightface":
        print("✓ Using InsightFace ArcFace-R50 model with ONNX")
        print("  512-dimensional face embeddings - EXCELLENT ACCURACY")
    elif RECOGNITION_METHOD == "arcface":
        print("✓ Using ArcFace model with PyTorch")
        print("  512-dimensional face embeddings - EXCELLENT ACCURACY")
    else:
        print("✓ Using enhanced multi-feature recognition (LBP + HOG + Histogram + Edge)")
        print("  Multi-scale feature extraction - GOOD ACCURACY")
        print("  For best results: Install InsightFace models")
    print(f"\nSettings: threshold={threshold:.2f}, min_match_ratio={min_match_ratio*100:.0f}%")
    print(f"Max allowed distance for best match: {max_allowed_distance:.2f}")
    print(f"Registered people: {len(all_people)}")
    print("\n✓ OPTIMIZED MODE: Balanced validation for reliable detection")
    print("Press Q to quit recognition mode\n")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera! Please check your camera connection.")
        return
    
    # Set camera resolution and FPS for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
    
    owner_seen = False
    total_detections = 0
    frame_count = 0
    recognition_cooldown = 0  # Prevent duplicate logging
    process_every_n_frames = 3  # Process every 3rd frame for smooth performance
    
    # Cache for face detection results
    cached_face_locations = []
    cached_results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read from camera!")
            break
        
        frame_count += 1
        recognition_cooldown = max(0, recognition_cooldown - 1)
        
        # Process face detection only every N frames for smooth display
        if frame_count % process_every_n_frames == 0:
            # Use smaller frame for face detection to speed up processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            face_locations = detect_faces_advanced(small_frame)
            # Scale back face locations to original frame size
            cached_face_locations = [(top * 2, right * 2, bottom * 2, left * 2) 
                                     for (top, right, bottom, left) in face_locations]
            cached_results = []  # Clear cached results for new detections
        
        # Use cached face locations for display
        for idx, face_location in enumerate(cached_face_locations):
            top, right, bottom, left = face_location
            
            # Check if we have cached result for this face
            if idx < len(cached_results):
                # Use cached result
                (is_owner, label, label_color, confidence, quality_label, 
                 match_count, total_samples, avg_distance, min_distance, rejection_reason) = cached_results[idx]
            else:
                # Process this face (only happens once every N frames)
                # Face quality check - skip if too small or poor quality
                quality_score, quality_label = assess_face_quality(frame, face_location)
                
                if (bottom - top) < 60 or (right - left) < 60 or quality_score < 40:
                    # Draw poor quality indicator
                    cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 128), 2)
                    cv2.putText(frame, "Poor Quality", (left, top-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                    continue
                
                # Extract face encoding
                encoding = extract_face_encoding(frame, face_location)
                
                if encoding is None:
                    continue
                
                # Compare with each registered person individually
                best_person = None
                best_person_score = 0
                best_person_confidence = 0
                best_person_stats = {}
                debug_info = []  # For debugging
                
                for person_name, person_data in all_people.items():
                    person_encodings = person_data["encodings"]
                    
                    # Compare with this person's encodings
                    if USE_FACE_RECOGNITION:
                        matches = face_recognition.compare_faces(person_encodings, encoding, tolerance=threshold)
                        distances = face_recognition.face_distance(person_encodings, encoding)
                    else:
                        # Fallback: compute distances manually using cosine similarity
                        matches = []
                        distances = []
                        for reg_encoding in person_encodings:
                            similarity = np.dot(encoding, reg_encoding) / (np.linalg.norm(encoding) * np.linalg.norm(reg_encoding) + 1e-6)
                            distance = 1 - similarity
                            distances.append(distance)
                            matches.append(distance < threshold)
                    
                    match_count = sum(matches)
                    min_distance = min(distances) if len(distances) > 0 else 1.0
                    avg_distance = np.mean(distances) if len(distances) > 0 else 1.0
                    max_distance = max(distances) if len(distances) > 0 else 1.0
                    
                    # Calculate match ratio for THIS person
                    match_ratio = match_count / len(person_encodings)
                    required_matches = int(len(person_encodings) * min_match_ratio)
                    
                    # More lenient matching - just need good average and some matches
                    person_matches = (
                        match_count >= required_matches and
                        min_distance < max_allowed_distance
                    )
                    
                    # Store debug info
                    debug_info.append(f"{person_name}: {match_count}/{len(person_encodings)} matches, min_dist={min_distance:.3f}, avg={avg_distance:.3f}")
                    
                    if person_matches:
                        # Calculate confidence score for this person
                        # More weight on minimum distance (best match)
                        distance_score = max(0, 1 - (avg_distance / threshold))
                        best_match_score = max(0, 1 - (min_distance / max_allowed_distance))
                        
                        # Give more weight to best match and match ratio
                        person_score = (
                            match_ratio * 0.25 +          # 25% weight on match ratio
                            distance_score * 0.25 +       # 25% weight on average
                            best_match_score * 0.50       # 50% weight on best match
                        )
                        
                        # Keep track of best matching person
                        if person_score > best_person_score:
                            best_person = person_name
                            best_person_score = person_score
                            best_person_confidence = person_score * 100
                            best_person_stats = {
                                'match_count': match_count,
                                'total_samples': len(person_encodings),
                                'avg_distance': avg_distance,
                                'min_distance': min_distance,
                                'max_distance': max_distance
                            }
                
                # Print debug info for unknown faces
                if not best_person and frame_count % process_every_n_frames == 0:
                    print(f"\n[DEBUG] No match found:")
                    for info in debug_info:
                        print(f"  {info}")
                    print(f"  Required: {int(12 * min_match_ratio)} matches, max_dist < {max_allowed_distance:.3f}\n")
                
                # Determine final result
                if best_person:
                    is_owner = True
                    matched_person_name = best_person
                    confidence = max(min(best_person_confidence, 94.0), 70.0)
                    match_count = best_person_stats['match_count']
                    total_samples = best_person_stats['total_samples']
                    avg_distance = best_person_stats['avg_distance']
                    min_distance = best_person_stats['min_distance']
                    rejection_reason = []
                    
                    # Print success message occasionally
                    if frame_count % (process_every_n_frames * 10) == 0:
                        print(f"✓ Detected: {best_person} (Confidence: {confidence:.1f}%, Matches: {match_count}/{total_samples})")
                else:
                    is_owner = False
                    matched_person_name = "Unknown"
                    confidence = 0
                    match_count = 0
                    total_samples = 0
                    avg_distance = 1.0
                    min_distance = 1.0
                    rejection_reason = ["No person matched criteria"]
            
                if is_owner:
                    label = matched_person_name.upper()
                    label_color = (0, 255, 0)  # Green
                    
                    owner_seen = True
                    
                    # Log to memory (with cooldown to avoid spam)
                    if recognition_cooldown == 0:
                        memory.add_recognition_event(True, confidence, quality_label)
                        recognition_cooldown = 30  # Log once every 30 frames (~1 second)
                        total_detections += 1
                        
                else:
                    label = "UNKNOWN PERSON"
                    label_color = (0, 0, 255)  # Red
                    confidence = 0
                    
                    # Log unknown detection
                    if recognition_cooldown == 0:
                        memory.add_recognition_event(False, 0, quality_label)
                        recognition_cooldown = 30
                        total_detections += 1
                        # Print rejection reason for debugging
                        if rejection_reason:
                            print(f"✗ Unknown face rejected: {', '.join(rejection_reason)}")
                
                # Cache the result
                cached_results.append((is_owner, label, label_color, confidence, quality_label,
                                     match_count, total_samples, avg_distance, min_distance, rejection_reason))
            
            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), label_color, 2)
            
            # Main label
            cv2.putText(frame, label, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
            
            # Show confidence for owner, rejection reason for unknown
            if is_owner:
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", (left, bottom+25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            else:
                # Show why rejected (first reason)
                if rejection_reason:
                    cv2.putText(frame, f"Rejected: {rejection_reason[0]}", (left, bottom+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
            
            # Detailed metrics
            if is_owner and total_samples > 0:
                cv2.putText(frame, f"M:{match_count}/{total_samples} Avg:{avg_distance:.3f} Min:{min_distance:.3f}", 
                            (left, bottom+50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            else:
                cv2.putText(frame, f"No match - Avg:{avg_distance:.3f} Min:{min_distance:.3f}", 
                            (left, bottom+50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            cv2.putText(frame, f"Quality: {quality_label}", (left, bottom+70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display session info
        cv2.putText(frame, f"Session: {session_id} | Detections: {total_detections}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display model info
        if USE_FACE_RECOGNITION:
            cv2.putText(frame, "Deep Learning (CNN+Large): ACTIVE", (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif RECOGNITION_METHOD == "insightface":
            cv2.putText(frame, "InsightFace (ONNX ArcFace-R50 512-D): ACTIVE", (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif RECOGNITION_METHOD == "arcface":
            cv2.putText(frame, "ArcFace (PyTorch 512-D): ACTIVE", (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Multi-Feature Recognition: ACTIVE", (10, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display instructions
        cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition (Advanced + Memory)", frame)
        
        # Increased waitKey for smoother display
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print("\nExiting recognition mode...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # End session
    memory.end_session(session_id, owner_seen, total_detections)
    
    # Show session summary
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    print(f"Session ID: {session_id}")
    print(f"Owner Detected: {'Yes ✓' if owner_seen else 'No ✗'}")
    print(f"Total Detections: {total_detections}")
    print("="*70)
    print("\nMemory updated and saved.")


def main():
    """
    Main menu for face registration and recognition with memory
    """
    memory = RecognitionMemory()
    
    while True:
        print("\n" + "="*70)
        print("ROBO BUDDY - ADVANCED FACE RECOGNITION SYSTEM WITH MEMORY")
        if USE_FACE_RECOGNITION:
            print("Deep Learning Model: dlib CNN + Large ResNet (128-D) ✓")
        elif RECOGNITION_METHOD == "insightface":
            print("Deep Learning Model: InsightFace ArcFace-R50 (512-D) ✓✓✓")
            print("State-of-the-art accuracy with ONNX runtime!")
        elif RECOGNITION_METHOD == "arcface":
            print("Deep Learning Model: ArcFace with PyTorch (512-D) ✓✓✓")
            print("State-of-the-art accuracy!")
        else:
            print("Enhanced Multi-Feature Recognition: LBP + HOG + Histogram ✓")
            print("For best accuracy: Install InsightFace models")
        print("="*70)
        print("1. Register Person Face (Deep Learning + Quality Check)")
        print("2. Recognize/Verify Face (Advanced + Memory Tracking)")
        print("3. View Registered People")
        print("4. View Memory Summary")
        print("5. View Recognition History")
        print("6. Delete Registered Person")
        print("7. Clear Memory")
        print("8. Exit")
        print("="*70)
        
        choice = input("Enter your choice (1-8): ").strip()
        
        if choice == "1":
            register_face_owner()
        elif choice == "2":
            recognize_face()
        elif choice == "3":
            # View registered people
            if os.path.exists(REGISTERED_FACES_FILE):
                with open(REGISTERED_FACES_FILE, 'rb') as f:
                    data = pickle.load(f)
                if "people" in data:
                    print("\n===REGISTERED PEOPLE===")
                    for name, info in data["people"].items():
                        print(f"  ✓ {name}")
                        print(f"    Registered: {info.get('registration_date', 'Unknown')}")
                        sample_count = info.get('sample_count', len(info.get('encodings', [])))
                        print(f"    Samples: {sample_count}")
                else:
                    print("\n⚠ Using old format - 1 person registered")
            else:
                print("\n⚠ No registered people found!")
        elif choice == "4":
            print(memory.get_summary())
        elif choice == "5":
            count = input("How many recent events to show? (default 10): ").strip()
            count = int(count) if count.isdigit() else 10
            memory.show_recent_history(count)
        elif choice == "6":
            # Delete person
            if os.path.exists(REGISTERED_FACES_FILE):
                with open(REGISTERED_FACES_FILE, 'rb') as f:
                    data = pickle.load(f)
                if "people" in data and len(data["people"]) > 0:
                    print("\nRegistered people:")
                    for name in data["people"].keys():
                        print(f"  - {name}")
                    del_name = input("Enter name to delete: ").strip()
                    if del_name in data["people"]:
                        del data["people"][del_name]
                        with open(REGISTERED_FACES_FILE, 'wb') as f:
                            pickle.dump(data, f)
                        print(f"✓ Deleted '{del_name}' successfully!")
                    else:
                        print("❌ Person not found!")
                else:
                    print("⚠ No registered people!")
            else:
                print("⚠ No registered faces file found!")
        elif choice == "7":
            confirm = input("Are you sure you want to clear all memory? (yes/no): ").strip().lower()
            if confirm == "yes":
                memory.memory_data = memory._initialize_memory()
                memory.save_memory()
                print("✓ Memory cleared successfully!")
            else:
                print("Memory clear cancelled.")
        elif choice == "8":
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please try again.")


if __name__ == "__main__":
    main()
