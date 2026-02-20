"""
Face Authentication Module for JUNO
Provides face recognition authentication before allowing voice assistant access
"""
import cv2
import numpy as np
import os
import pickle
import time
from datetime import datetime

# Try to import face recognition libraries
try:
    import face_recognition
    USE_FACE_RECOGNITION = True
    print("[AUTH] Using face_recognition library")
except ImportError:
    USE_FACE_RECOGNITION = False
    print("[AUTH] face_recognition not available, using OpenCV fallback")

# File to store registered faces
REGISTERED_FACES_FILE = "registered_faces_advanced.pkl"

def detect_face_opencv(frame):
    """Detect face using OpenCV Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )
    
    # Convert to (top, right, bottom, left) format
    face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in faces]
    return face_locations


def detect_face(frame):
    """Detect face using best available method"""
    if USE_FACE_RECOGNITION:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            return face_locations
        except:
            return detect_face_opencv(frame)
    else:
        return detect_face_opencv(frame)


def extract_encoding(frame, face_location):
    """Extract face encoding"""
    if USE_FACE_RECOGNITION:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, [face_location])
        return encodings[0] if len(encodings) > 0 else None
    else:
        # OpenCV fallback - use simple features
        top, right, bottom, left = face_location
        face_roi = frame[top:bottom, left:right]
        
        if face_roi.size == 0:
            return None
        
        # Resize to fixed size and flatten
        face_resized = cv2.resize(face_roi, (64, 64))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        encoding = gray.flatten().astype(np.float32) / 255.0
        return encoding


def verify_owner(frame, threshold=0.5):
    """
    Verify if the person in frame is the registered owner
    Returns: (is_authenticated, confidence, message)
    """
    # Check if registered faces exist
    if not os.path.exists(REGISTERED_FACES_FILE):
        return False, 0.0, "No registered owner. Please register first using recognition_advanced.py"
    
    # Load registered faces
    try:
        with open(REGISTERED_FACES_FILE, 'rb') as f:
            registered_data = pickle.load(f)
    except Exception as e:
        return False, 0.0, f"Error loading registered faces: {str(e)}"
    
    # Handle both old and new format
    if "people" in registered_data:
        all_people = registered_data["people"]
        if len(all_people) == 0:
            return False, 0.0, "No registered people found"
        # Get first person as owner (or you can specify)
        owner_encodings = list(all_people.values())[0]["encodings"]
    else:
        owner_encodings = registered_data.get("encodings", [])
    
    if len(owner_encodings) == 0:
        return False, 0.0, "No owner encodings found"
    
    # Detect face in frame
    face_locations = detect_face(frame)
    
    if len(face_locations) == 0:
        return False, 0.0, "No face detected"
    
    # Use the largest face (closest to camera)
    face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
    
    # Extract encoding from current frame
    current_encoding = extract_encoding(frame, face_location)
    
    if current_encoding is None:
        return False, 0.0, "Could not extract face features"
    
    # Compare with registered encodings
    if USE_FACE_RECOGNITION:
        matches = face_recognition.compare_faces(owner_encodings, current_encoding, tolerance=threshold)
        distances = face_recognition.face_distance(owner_encodings, current_encoding)
    else:
        # Fallback comparison using cosine similarity
        matches = []
        distances = []
        for reg_encoding in owner_encodings:
            similarity = np.dot(current_encoding, reg_encoding) / (
                np.linalg.norm(current_encoding) * np.linalg.norm(reg_encoding) + 1e-6
            )
            distance = 1 - similarity
            distances.append(distance)
            matches.append(distance < threshold)
    
    match_count = sum(matches)
    match_ratio = match_count / len(owner_encodings)
    min_distance = min(distances) if distances else 1.0
    avg_distance = np.mean(distances) if distances else 1.0
    
    # Determine if authenticated
    # Need at least 60% of samples to match and min distance < threshold
    is_authenticated = match_ratio >= 0.6 and min_distance < threshold
    
    # Calculate confidence (0-100)
    if is_authenticated:
        confidence = max(70, min(95, (1 - min_distance) * 100))
        message = f"Owner verified ({match_count}/{len(owner_encodings)} matches)"
    else:
        confidence = 0.0
        message = f"Unauthorized ({match_count}/{len(owner_encodings)} matches, dist: {min_distance:.3f})"
    
    return is_authenticated, confidence, message


def capture_and_verify(camera_index=0, timeout=10):
    """
    Capture from camera and verify owner with timeout
    Returns: (is_authenticated, confidence, message, frame)
    """
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        return False, 0.0, "Could not open camera", None
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    start_time = time.time()
    best_result = (False, 0.0, "No valid face detected", None)
    
    while (time.time() - start_time) < timeout:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Try to verify
        is_auth, confidence, message = verify_owner(frame)
        
        # Keep best result
        if confidence > best_result[1]:
            best_result = (is_auth, confidence, message, frame.copy())
        
        # If authenticated with good confidence, return immediately
        if is_auth and confidence > 80:
            cap.release()
            return best_result
        
        # Small delay
        time.sleep(0.1)
    
    cap.release()
    return best_result


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" JUNO FACE AUTHENTICATION TEST")
    print("="*60)
    
    # Test authentication
    is_auth, confidence, message, frame = capture_and_verify(timeout=5)
    
    print(f"\nAuthentication Result:")
    print(f"  Status: {'✓ AUTHENTICATED' if is_auth else '✗ FAILED'}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  Message: {message}")
    print("\n" + "="*60)
    
    if frame is not None and is_auth:
        # Show result
        cv2.imshow("Authentication Result", frame)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()
