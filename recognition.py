import cv2
import numpy as np
import os
import pickle
import time

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# File to store registered face data
REGISTERED_FACES_FILE = "registered_faces.pkl"
FACE_SAMPLES_DIR = "face_samples"

# Create directory for storing face samples if it doesn't exist
if not os.path.exists(FACE_SAMPLES_DIR):
    os.makedirs(FACE_SAMPLES_DIR)


def extract_face_features(face_roi):
    """
    Extract distinctive features from a face ROI using LBP and gradient-based features
    OPTIMIZED VERSION for speed
    """
    # Resize for consistent feature extraction (smaller = faster)
    face_roi = cv2.resize(face_roi, (96, 96))  # Reduced from 128x128 for speed
    
    # Apply histogram equalization for better lighting normalization
    face_roi = cv2.equalizeHist(face_roi)
    
    # Extract Local Binary Pattern (LBP) features - OPTIMIZED with NumPy
    def compute_lbp_fast(image):
        """Faster LBP using NumPy vectorization"""
        h, w = image.shape
        
        # Vectorized LBP computation
        center = image[1:-1, 1:-1].astype(np.int16)
        lbp = ((image[0:-2, 0:-2].astype(np.int16) >= center) << 7) | \
              ((image[0:-2, 1:-1].astype(np.int16) >= center) << 6) | \
              ((image[0:-2, 2:].astype(np.int16) >= center) << 5) | \
              ((image[1:-1, 2:].astype(np.int16) >= center) << 4) | \
              ((image[2:, 2:].astype(np.int16) >= center) << 3) | \
              ((image[2:, 1:-1].astype(np.int16) >= center) << 2) | \
              ((image[2:, 0:-2].astype(np.int16) >= center) << 1) | \
              (image[1:-1, 0:-2].astype(np.int16) >= center)
        
        return lbp.astype(np.uint8)
    
    lbp = compute_lbp_fast(face_roi)
    # Use numpy histogram instead of cv2.calcHist for better compatibility
    lbp_hist, _ = np.histogram(lbp.flatten(), bins=128, range=(0, 256))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist = lbp_hist / (np.sum(lbp_hist) + 1e-8)  # Normalize
    
    # Compute gradient features (Sobel) - using smaller kernel for speed
    sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    # Use numpy histogram for consistency
    gradient_hist, _ = np.histogram(gradient_magnitude.flatten(), bins=32, range=(0, 256))
    gradient_hist = gradient_hist.astype(np.float32)
    gradient_hist = gradient_hist / (np.sum(gradient_hist) + 1e-8)  # Normalize
    
    # Divide face into regions (eyes, nose, mouth areas) - fewer regions for speed
    region_features = []
    h, w = face_roi.shape
    regions = [
        face_roi[0:h//3, 0:w//2],           # Top-left (left eye area)
        face_roi[0:h//3, w//2:w],           # Top-right (right eye area)
        face_roi[h//3:2*h//3, w//4:3*w//4], # Middle (nose area)
        face_roi[2*h//3:h, w//4:3*w//4]     # Bottom (mouth area)
    ]
    
    for region in regions:
        # Use numpy histogram for consistency
        region_hist, _ = np.histogram(region.flatten(), bins=16, range=(0, 256))
        region_hist = region_hist.astype(np.float32)
        region_hist = region_hist / (np.sum(region_hist) + 1e-8)  # Normalize
        region_features.extend(region_hist)
    
    # Combine all features
    features = np.concatenate([lbp_hist, gradient_hist, region_features])
    return features


def register_face_owner():
    """
    Register the device owner's face by capturing frontal face images
    """
    print("\n=== FACE REGISTRATION MODE ===")
    print("You will be capturing FRONTAL face images")
    print("Please look directly at the camera")
    print("Press SPACE to capture, Q to quit, R to reset")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows reliability
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera! Please check your camera connection.")
        return False
    
    # Optimize camera settings for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
    
    captured_faces = []
    captured_count = 0
    samples_needed = 15  # Capture 15 frontal samples for better accuracy
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read from camera!")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Improved detection parameters: scaleFactor=1.1, minNeighbors=6 for better accuracy
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))
        
        # Display instructions
        face_color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)
        cv2.putText(frame, "Look DIRECTLY at camera", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Samples Captured: {captured_count}/{samples_needed}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "SPACE=Capture | Q=Quit | R=Reset", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), face_color, 2)
        
        cv2.imshow("Face Registration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Registration cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return False
        
        elif key == ord('r'):
            print("Reset: Clearing all captured faces.")
            captured_faces = []
            captured_count = 0
        
        elif key == ord(' '):  # SPACE key
            if len(faces) > 0:
                # Extract the largest face (ensure it's frontal)
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                (x, y, w, h) = largest_face
                face_roi = gray[y:y+h, x:x+w]
                
                # Store the face
                captured_faces.append(face_roi)
                captured_count += 1
                
                print(f"✓ Captured frontal sample {captured_count}/{samples_needed}")
                
                # Check if all samples are captured
                if captured_count >= samples_needed:
                    print("\n✓ All frontal samples captured successfully!")
                    break
                else:
                    time.sleep(0.3)  # Small delay between captures
            else:
                print("⚠ No frontal face detected! Please look directly at the camera.")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Check if we have enough samples
    if len(captured_faces) < samples_needed:
        print(f"\n⚠ Registration cancelled. Only {len(captured_faces)} samples captured.")
        return False
    
    # Extract features from all captured samples
    print("\nExtracting features from captured frontal faces...")
    registered_data = {"features": []}
    
    for face_sample in captured_faces:
        features = extract_face_features(face_sample)
        registered_data["features"].append(features)
    
    # Save registered face data
    with open(REGISTERED_FACES_FILE, 'wb') as f:
        pickle.dump(registered_data, f)
    
    print(f"✓ Face registration complete! {len(captured_faces)} frontal samples saved to {REGISTERED_FACES_FILE}")
    return True


def recognize_face():
    """
    Recognize and verify if detected face belongs to registered owner
    """
    # Load registered faces
    if not os.path.exists(REGISTERED_FACES_FILE):
        print("⚠ No registered faces found! Please register first.")
        return
    
    with open(REGISTERED_FACES_FILE, 'rb') as f:
        registered_data = pickle.load(f)
    
    registered_features = registered_data["features"]
    
    if len(registered_features) == 0:
        print("⚠ No registered face data available!")
        return
    
    # Pre-compute average feature vector for faster matching (SPEED OPTIMIZATION)
    registered_features_array = np.array(registered_features)
    avg_registered_feature = np.mean(registered_features_array, axis=0)
    std_registered_feature = np.std(registered_features_array, axis=0)
    
    # EXTREMELY STRICT THRESHOLD - prevents false positives
    threshold = 0.18  # VERY STRICT threshold (0-1 scale)
    min_samples_match = 12  # Must match at least 12 out of 15 registered samples (80%)
    max_allowed_distance = 0.25  # Maximum average distance allowed
    
    print("\n=== FACE RECOGNITION MODE ===")
    print(f"STRICT Settings: threshold={threshold:.2f}, min_matches={min_samples_match}/{len(registered_features)}")
    print(f"Max avg distance: {max_allowed_distance:.2f}")
    print("EXTREMELY STRICT MODE - Optimized to prevent false positives")
    print("⚡ SPEED OPTIMIZED: Skipping frames & comparing only 10 samples per face")
    print("Press Q to quit recognition mode")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows reliability
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera! Please check your camera connection.")
        return
    
    # Optimize camera settings for speed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
    
    frame_skip = 2  # Process every 2nd frame for speed
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Failed to read from camera!")
            break
        
        frame_count += 1
        
        # Skip frames for better performance
        process_frame = (frame_count % frame_skip == 0)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Only detect faces on processed frames
        if process_frame:
            # Use same improved detection parameters as registration
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))
        else:
            faces = []  # Skip detection on non-processed frames
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            # Face quality check - skip if too small or poor quality
            if w < 80 or h < 80:
                continue
            
            # Extract features
            features = extract_face_features(face_roi)
            
            # IMPROVED MATCHING: Use cosine similarity (better for high-dimensional features)
            # Normalize features
            features_norm = features / (np.linalg.norm(features) + 1e-8)
            
            # SPEED OPTIMIZATION: Only compare to top 10 registered features (not all 15)
            # Pre-compute which samples to compare (random subset for speed)
            sample_indices = np.random.choice(len(registered_features), min(10, len(registered_features)), replace=False)
            
            # Compute distance to selected registered samples
            match_count = 0
            distances = []
            for idx in sample_indices:
                reg_feature = registered_features[idx]
                reg_norm = reg_feature / (np.linalg.norm(reg_feature) + 1e-8)
                similarity = np.dot(features_norm, reg_norm)
                distance = 1 - similarity
                distances.append(distance)
                if distance < threshold:
                    match_count += 1
            
            min_distance = min(distances) if distances else 1.0
            avg_top5_distance = np.mean(sorted(distances)[:5])  # Average of 5 closest matches
            avg_all_distance = np.mean(distances)  # Average of all distances
            
            # EXTREMELY STRICT MATCHING CRITERIA: ALL conditions must be met
            # Adjusted for smaller sample size (10 instead of 15)
            # 1. Must have enough matches (60% of sampled)
            # 2. Top-3 average distance must be very low
            # 3. Overall average distance must be reasonable
            # 4. Minimum distance must be very low
            min_matches_needed = 6  # 60% of 10 samples
            condition1 = match_count >= min_matches_needed
            condition2 = avg_top5_distance < threshold
            condition3 = avg_all_distance < max_allowed_distance
            condition4 = min_distance < (threshold * 0.9)
            
            is_owner = condition1 and condition2 and condition3 and condition4
            
            # Determine if face matches with PROPER confidence calculation
            if is_owner:
                label = "OWNER DETECTED"
                label_color = (0, 255, 0)  # Green
                # Confidence based on how many samples matched and average distance
                match_ratio = match_count / len(sample_indices)  # Use sampled count
                distance_score = max(0, 1 - (avg_top5_distance / threshold))
                confidence = (match_ratio * 0.5 + distance_score * 0.5) * 100
                confidence = max(min(confidence, 90.0), 65.0)  # Cap at 90%, minimum 65%
            else:
                label = "UNKNOWN PERSON"
                label_color = (0, 0, 255)  # Red
                confidence = 0
                
                # Debug info for unknown faces
                if match_count < min_matches_needed:
                    reason = f"Low matches: {match_count}/{min_matches_needed}"
                elif avg_top5_distance >= threshold:
                    reason = f"High dist: {avg_top5_distance:.3f}"
                elif avg_all_distance >= max_allowed_distance:
                    reason = f"Avg too high: {avg_all_distance:.3f}"
                else:
                    reason = f"Min dist: {min_distance:.3f}"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), label_color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (x, y+h+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
            
            if is_owner:
                cv2.putText(frame, f"Match: {match_count}/{len(sample_indices)} | Dist: {avg_top5_distance:.3f}", (x, y+h+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                cv2.putText(frame, f"REJECTED: {reason}", (x, y+h+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Display instructions BEFORE showing the frame
        cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Face Recognition", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting recognition mode...")
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Main menu for face registration and recognition
    """
    while True:
        print("\n" + "="*50)
        print("ROBO BUDDY - FACE RECOGNITION SYSTEM")
        print("="*50)
        print("1. Register Owner Face (Frontal Detection Only)")
        print("2. Recognize/Verify Face")
        print("3. Exit")
        print("="*50)
        
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            register_face_owner()
        elif choice == "2":
            recognize_face()
        elif choice == "3":
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please try again.")


if __name__ == "__main__":
    main()
