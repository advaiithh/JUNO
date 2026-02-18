"""
ULTRA ADVANCED FACE RECOGNITION - Using InsightFace/Hugging Face Models
This version uses the most accurate face recognition models available
"""

import cv2
import numpy as np
import os
import pickle
import time

# Try to import advanced libraries in order of preference
RECOGNITION_METHOD = None

# Try InsightFace (Best accuracy, industry standard)
try:
    from insightface.app import FaceAnalysis
    RECOGNITION_METHOD = 'insightface'
    print("✓ Using InsightFace - Industry-grade face recognition")
except ImportError:
    pass

# Try Hugging Face Transformers
if RECOGNITION_METHOD is None:
    try:
        from transformers import AutoImageProcessor, AutoModel
        import torch
        from PIL import Image
        RECOGNITION_METHOD = 'transformers'
        print("✓ Using Hugging Face Transformers")
    except ImportError:
        pass

# Fallback to face_recognition
if RECOGNITION_METHOD is None:
    try:
        import face_recognition
        RECOGNITION_METHOD = 'face_recognition'
        print("✓ Using face_recognition (dlib)")
    except ImportError:
        pass

if RECOGNITION_METHOD is None:
    print("\n⚠ WARNING: No advanced face recognition library found!")
    print("Install one of these:")
    print("  1. InsightFace (Best):     pip install insightface onnxruntime")
    print("  2. Face Recognition:        pip install face-recognition")
    print("  3. Hugging Face:            pip install transformers torch")
    exit(1)

# File to store registered face data
REGISTERED_FACES_FILE = "registered_faces_ultra.pkl"
FACE_SAMPLES_DIR = "face_samples"

# Create directory for storing face samples
if not os.path.exists(FACE_SAMPLES_DIR):
    os.makedirs(FACE_SAMPLES_DIR)


class AdvancedFaceRecognizer:
    """
    Advanced Face Recognition using state-of-the-art models
    """
    
    def __init__(self):
        self.method = RECOGNITION_METHOD
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the appropriate face recognition model"""
        
        if self.method == 'insightface':
            # Initialize InsightFace - uses ArcFace/RetinaFace models
            self.model = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            print("✓ InsightFace ArcFace model loaded (512-D embeddings)")
            
        elif self.method == 'transformers':
            # Use Hugging Face model for face recognition
            model_name = "microsoft/resnet-50"  # Can use face-specific models
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Hugging Face model loaded on {self.device}")
    
    def detect_and_encode_faces(self, frame):
        """
        Detect faces and extract embeddings using the advanced model
        Returns: [(face_location, embedding), ...]
        """
        results = []
        
        if self.method == 'insightface':
            # InsightFace handles both detection and recognition
            faces = self.model.get(frame)
            for face in faces:
                # face.bbox: [x1, y1, x2, y2]
                bbox = face.bbox.astype(int)
                location = (bbox[1], bbox[2], bbox[3], bbox[0])  # Convert to (top, right, bottom, left)
                embedding = face.embedding  # 512-dimensional embedding
                results.append((location, embedding))
        
        elif self.method == 'transformers':
            # Use OpenCV for detection, Transformers for encoding
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                location = (y, x+w, y+h, x)
                face_img = frame[y:y+h, x:x+w]
                
                # Extract embedding using Transformers
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                inputs = self.processor(images=face_pil, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use pooled output or last hidden state
                    if hasattr(outputs, 'pooler_output'):
                        embedding = outputs.pooler_output.cpu().numpy().flatten()
                    else:
                        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
                
                results.append((location, embedding))
        
        elif self.method == 'face_recognition':
            # Use face_recognition library
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model='large')
            
            for location, encoding in zip(face_locations, face_encodings):
                results.append((location, encoding))
        
        return results
    
    def compare_faces(self, known_encodings, face_encoding, threshold=0.6):
        """
        Compare a face encoding against known encodings
        Returns: (match_count, distances, is_match)
        """
        if self.method == 'insightface':
            # Use cosine similarity for InsightFace embeddings
            similarities = []
            for known_enc in known_encodings:
                similarity = np.dot(face_encoding, known_enc) / (np.linalg.norm(face_encoding) * np.linalg.norm(known_enc))
                similarities.append(similarity)
            
            # Convert similarity to distance (1 - similarity)
            distances = [1 - sim for sim in similarities]
            matches = [dist < threshold for dist in distances]
            
        elif self.method == 'transformers':
            # Cosine similarity for transformer embeddings
            distances = []
            for known_enc in known_encodings:
                # Cosine distance
                similarity = np.dot(face_encoding, known_enc) / (np.linalg.norm(face_encoding) * np.linalg.norm(known_enc))
                distance = 1 - similarity
                distances.append(distance)
            matches = [dist < threshold for dist in distances]
        
        elif self.method == 'face_recognition':
            # Use face_recognition's built-in comparison
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=threshold)
            distances = face_recognition.face_distance(known_encodings, face_encoding)
        
        return sum(matches), distances, any(matches)


def register_face_owner():
    """
    Register the device owner's face using ultra-advanced deep learning
    """
    print("\n" + "="*70)
    print("ULTRA ADVANCED FACE REGISTRATION")
    print("="*70)
    print(f"Using: {RECOGNITION_METHOD.upper()}")
    print("Press SPACE to capture, Q to quit, R to reset")
    print("="*70)
    
    recognizer = AdvancedFaceRecognizer()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera!")
        return False
    
    captured_encodings = []
    captured_count = 0
    samples_needed = 8  # Only 8 samples needed with advanced models
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read from camera!")
            break
        
        # Detect faces and get encodings
        face_results = recognizer.detect_and_encode_faces(frame)
        
        # Display info
        face_color = (0, 255, 0) if len(face_results) > 0 else (0, 0, 255)
        cv2.putText(frame, f"Model: {RECOGNITION_METHOD.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Samples: {captured_count}/{samples_needed}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE=Capture | Q=Quit | R=Reset", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw detected faces
        for (location, _) in face_results:
            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), face_color, 2)
        
        cv2.imshow("Ultra Advanced Registration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        elif key == ord('r'):
            captured_encodings = []
            captured_count = 0
            print("Reset: Cleared all samples.")
        
        elif key == ord(' '):
            if len(face_results) > 0:
                # Get largest face
                largest_face = max(face_results, key=lambda f: (f[0][2]-f[0][0]) * (f[0][1]-f[0][3]))
                _, encoding = largest_face
                
                captured_encodings.append(encoding)
                captured_count += 1
                
                print(f"✓ Captured sample {captured_count}/{samples_needed}")
                
                if captured_count >= samples_needed:
                    print("\n✓ All samples captured!")
                    break
                time.sleep(0.3)
            else:
                print("⚠ No face detected!")
    
    cap.release()
    cv2.destroyAllWindows()
    
    if len(captured_encodings) < samples_needed:
        print(f"\n⚠ Registration cancelled. Only {len(captured_encodings)} samples.")
        return False
    
    # Save encodings
    registered_data = {
        "encodings": captured_encodings,
        "method": RECOGNITION_METHOD
    }
    
    with open(REGISTERED_FACES_FILE, 'wb') as f:
        pickle.dump(registered_data, f)
    
    print(f"✓ Registration complete! {len(captured_encodings)} encodings saved using {RECOGNITION_METHOD}")
    return True


def recognize_face():
    """
    Recognize faces using ultra-advanced deep learning
    """
    if not os.path.exists(REGISTERED_FACES_FILE):
        print("⚠ No registered faces found! Please register first.")
        return
    
    with open(REGISTERED_FACES_FILE, 'rb') as f:
        registered_data = pickle.load(f)
    
    registered_encodings = registered_data["encodings"]
    
    print("\n" + "="*70)
    print("ULTRA ADVANCED FACE RECOGNITION")
    print("="*70)
    print(f"Using: {RECOGNITION_METHOD.upper()}")
    print("Press Q to quit")
    print("="*70)
    
    recognizer = AdvancedFaceRecognizer()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera!")
        return
    
    # Thresholds based on method
    if RECOGNITION_METHOD == 'insightface':
        threshold = 0.40  # Cosine distance threshold
        min_matches = 5
    elif RECOGNITION_METHOD == 'transformers':
        threshold = 0.35
        min_matches = 4
    else:  # face_recognition
        threshold = 0.45
        min_matches = 5
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and recognize faces
        face_results = recognizer.detect_and_encode_faces(frame)
        
        for (location, encoding) in face_results:
            top, right, bottom, left = location
            
            # Compare with registered faces
            match_count, distances, is_match = recognizer.compare_faces(
                registered_encodings, encoding, threshold
            )
            
            avg_distance = np.mean(distances) if len(distances) > 0 else 1.0
            
            # Strict matching criteria
            is_owner = (match_count >= min_matches and avg_distance < threshold)
            
            if is_owner:
                label = "OWNER DETECTED"
                label_color = (0, 255, 0)
                match_ratio = match_count / len(registered_encodings)
                distance_score = max(0, 1 - (avg_distance / threshold))
                confidence = (match_ratio * 0.5 + distance_score * 0.5) * 100
                confidence = min(confidence, 94.0)
            else:
                label = "UNKNOWN"
                label_color = (0, 0, 255)
                confidence = 0
            
            # Draw results
            cv2.rectangle(frame, (left, top), (right, bottom), label_color, 2)
            cv2.putText(frame, label, (left, top-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
            cv2.putText(frame, f"Conf: {confidence:.1f}%", (left, bottom+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            cv2.putText(frame, f"{match_count}/{len(registered_encodings)} | {avg_distance:.3f}", 
                        (left, bottom+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Display model info
        cv2.putText(frame, f"Model: {RECOGNITION_METHOD.upper()}", (10, frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Ultra Advanced Recognition", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main menu"""
    while True:
        print("\n" + "="*70)
        print("ULTRA ADVANCED FACE RECOGNITION SYSTEM")
        print(f"Model: {RECOGNITION_METHOD.upper()}")
        print("="*70)
        print("1. Register Owner Face")
        print("2. Recognize Face")
        print("3. Exit")
        print("="*70)
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            register_face_owner()
        elif choice == "2":
            recognize_face()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
