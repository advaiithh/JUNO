"""
JUNO Face Registration - Auto Capture
Fully automatic registration with 64x64 encoding (4096 dimensions)
"""
import cv2
import numpy as np
import pickle
from datetime import datetime
import time

REGISTERED_FACES_FILE = "registered_faces_advanced.pkl"

def auto_register_face():
    print("\n" + "="*60)
    print(" JUNO AUTO FACE REGISTRATION")
    print("="*60)
    
    name = input("\nEnter your name: ").strip() or "Advaith"
    
    print(f"\nRegistering face for: {name}")
    print("Getting ready...")
    print("  - Look directly at camera")
    print("  - Keep face well-lit and centered")
    print("  - System will auto-capture 15 samples\n")
    
    input("Press ENTER when ready...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera!")
        return False
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    
    encodings = []
    target_samples = 15
    last_capture_time = 0
    capture_interval = 0.5  # seconds between captures
    
    print("\n🎥 Recording started... Stay in position!\n")
    
    while len(encodings) < target_samples:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        
        current_time = time.time()
        
        # Draw rectangles and auto-capture
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(display_frame, "Face Detected", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Auto-capture if enough time has passed
            if current_time - last_capture_time > capture_interval:
                # Use the largest face
                (fx, fy, fw, fh) = max(faces, key=lambda f: f[2] * f[3])
                face_roi = frame[fy:fy+fh, fx:fx+fw]
                
                if face_roi.size > 0:
                    # Create encoding EXACTLY 64x64 = 4096 dimensions
                    face_resized = cv2.resize(face_roi, (64, 64))
                    gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    encoding = gray_face.flatten().astype(np.float32) / 255.0
                    
                    # Verify encoding size
                    if len(encoding) == 4096:
                        encodings.append(encoding)
                        last_capture_time = current_time
                        print(f"✓ Auto-captured sample {len(encodings)}/{target_samples} (size: {len(encoding)})")
                    else:
                        print(f"⚠ Warning: Wrong encoding size {len(encoding)}, skipping")
        
        # Show progress
        cv2.putText(display_frame, f"Samples: {len(encodings)}/{target_samples}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Auto-capturing... Stay still", 
                   (10, display_frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("JUNO Face Registration", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n❌ Registration cancelled")
            cap.release()
            cv2.destroyAllWindows()
            return False
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Verify all encodings
    print(f"\n✓ Captured {len(encodings)} samples")
    print(f"✓ Verifying encoding sizes...")
    for i, enc in enumerate(encodings):
        print(f"  Sample {i+1}: {len(enc)} dimensions (shape: {enc.shape})")
    
    # Save registration
    registration_data = {
        "people": {
            name: {
                "encodings": encodings,
                "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sample_count": len(encodings),
                "method": "opencv_64x64",
                "encoding_size": 4096
            }
        },
        "use_face_recognition": False
    }
    
    with open(REGISTERED_FACES_FILE, 'wb') as f:
        pickle.dump(registration_data, f)
    
    print(f"\n✓ Registration complete!")
    print(f"  Name: {name}")
    print(f"  Samples: {len(encodings)}")
    print(f"  Encoding: 64x64 = 4096 dimensions")
    print(f"  File: {REGISTERED_FACES_FILE}")
    print("="*60 + "\n")
    
    return True

if __name__ == "__main__":
    success = auto_register_face()
    if success:
        print("✅ You can now use JUNO with face authentication!")
    else:
        print("❌ Registration failed. Please try again.")
    input("\nPress Enter to exit...")
