"""
Debug script to test face recognition and identify issues
"""
import cv2
import numpy as np
import pickle
import os

# Check what libraries are available
print("="*60)
print("CHECKING INSTALLED LIBRARIES")
print("="*60)

try:
    import face_recognition
    print("✓ face_recognition library: INSTALLED")
    USE_FACE_RECOGNITION = True
except ImportError:
    print("❌ face_recognition library: NOT INSTALLED")
    print("   Install with: pip install face-recognition")
    USE_FACE_RECOGNITION = False

print("\n" + "="*60)
print("CHECKING REGISTERED FACE FILES")
print("="*60)

files_to_check = [
    "registered_faces.pkl",
    "registered_faces_advanced.pkl",
    "registered_faces_ultra.pkl"
]

for filename in files_to_check:
    if os.path.exists(filename):
        print(f"\n✓ Found: {filename}")
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, dict):
                print(f"  - Keys: {list(data.keys())}")
                if 'encodings' in data:
                    print(f"  - Number of encodings: {len(data['encodings'])}")
                    if len(data['encodings']) > 0:
                        print(f"  - Encoding dimension: {len(data['encodings'][0])}")
                if 'features' in data:
                    print(f"  - Number of features: {len(data['features'])}")
                    if len(data['features']) > 0:
                        print(f"  - Feature dimension: {len(data['features'][0])}")
                if 'use_face_recognition' in data:
                    print(f"  - Created with face_recognition: {data['use_face_recognition']}")
        except Exception as e:
            print(f"  ❌ Error reading file: {e}")
    else:
        print(f"❌ Not found: {filename}")

print("\n" + "="*60)
print("TESTING CAMERA")
print("="*60)

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✓ Camera: WORKING")
    ret, frame = cap.read()
    if ret:
        print(f"  - Resolution: {frame.shape[1]}x{frame.shape[0]}")
        
        if USE_FACE_RECOGNITION:
            print("\n  Testing face detection with face_recognition...")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            print(f"  - Faces detected: {len(face_locations)}")
            
            if len(face_locations) > 0:
                print("\n  Testing face encoding extraction...")
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if len(encodings) > 0:
                    print(f"  ✓ Face encoding extracted: {len(encodings[0])}-dimensional")
        else:
            print("\n  Testing OpenCV face detection...")
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(100, 100))
            print(f"  - Faces detected: {len(faces)}")
    
    cap.release()
else:
    print("❌ Camera: NOT WORKING")

print("\n" + "="*60)
print("LIVE RECOGNITION TEST")
print("="*60)

if USE_FACE_RECOGNITION and os.path.exists("registered_faces_advanced.pkl"):
    print("Starting live test with face_recognition library...")
    print("Press Q to quit, SPACE to capture and test a face")
    
    with open("registered_faces_advanced.pkl", 'rb') as f:
        registered_data = pickle.load(f)
    
    registered_encodings = registered_data["encodings"]
    print(f"Loaded {len(registered_encodings)} registered encodings")
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        
        cv2.putText(frame, f"Faces detected: {len(face_locations)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press SPACE to test match | Q to quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        cv2.imshow("Debug Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' ') and len(face_locations) > 0:
            print("\n" + "-"*60)
            print("TESTING FACE MATCH...")
            
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if len(face_encodings) > 0:
                test_encoding = face_encodings[0]
                
                # Test with multiple thresholds
                thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
                
                print(f"\nTesting against {len(registered_encodings)} registered faces:")
                
                for threshold in thresholds:
                    matches = face_recognition.compare_faces(registered_encodings, test_encoding, tolerance=threshold)
                    distances = face_recognition.face_distance(registered_encodings, test_encoding)
                    
                    match_count = sum(matches)
                    avg_distance = np.mean(distances)
                    min_distance = min(distances)
                    
                    print(f"\nThreshold {threshold:.2f}:")
                    print(f"  - Matches: {match_count}/{len(registered_encodings)}")
                    print(f"  - Min distance: {min_distance:.4f}")
                    print(f"  - Avg distance: {avg_distance:.4f}")
                    print(f"  - All distances: {[f'{d:.4f}' for d in distances[:5]]}...")
                
                print("\n" + "-"*60)
    
    cap.release()
    cv2.destroyAllWindows()
    
else:
    print("Cannot run live test:")
    if not USE_FACE_RECOGNITION:
        print("  - face_recognition library not installed")
    if not os.path.exists("registered_faces_advanced.pkl"):
        print("  - No registered faces file found")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
