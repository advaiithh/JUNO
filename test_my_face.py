"""
Quick test to verify registered face and show matching scores
"""
import cv2
import numpy as np
import pickle
import os

# Check if registered file exists
if not os.path.exists("registered_faces.pkl"):
    print("❌ No registered face found!")
    print("Run: python recognition.py")
    print("Then choose option 1 to register your face")
    exit()

# Load registered data
with open("registered_faces.pkl", 'rb') as f:
    registered_data = pickle.load(f)

registered_features = registered_data["features"]

print("="*70)
print("REGISTERED FACE INFO")
print("="*70)
print(f"Number of samples: {len(registered_features)}")
print(f"Feature dimension: {len(registered_features[0])}")
print(f"File: registered_faces.pkl")

# Analyze consistency of registered samples
if len(registered_features) > 1:
    print("\nAnalyzing sample consistency...")
    distances = []
    for i in range(len(registered_features)):
        for j in range(i+1, len(registered_features)):
            f1 = registered_features[i] / (np.linalg.norm(registered_features[i]) + 1e-8)
            f2 = registered_features[j] / (np.linalg.norm(registered_features[j]) + 1e-8)
            similarity = np.dot(f1, f2)
            distance = 1 - similarity
            distances.append(distance)
    
    avg_internal_distance = np.mean(distances)
    max_internal_distance = max(distances)
    min_internal_distance = min(distances)
    
    print(f"  Average inter-sample distance: {avg_internal_distance:.4f}")
    print(f"  Max distance between samples: {max_internal_distance:.4f}")
    print(f"  Min distance between samples: {min_internal_distance:.4f}")
    
    if avg_internal_distance < 0.3:
        print("  ✓ Samples are very consistent (good registration)")
    elif avg_internal_distance < 0.45:
        print("  ✓ Samples are reasonably consistent")
    else:
        print("  ⚠ Samples vary significantly - consider re-registering")

print("\n" + "="*70)
print("LIVE TEST - Press SPACE to test match, Q to quit")
print("="*70)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def extract_face_features(face_roi):
    """Extract same features as in recognition.py"""
    face_roi = cv2.resize(face_roi, (128, 128))
    face_roi = cv2.equalizeHist(face_roi)
    
    def compute_lbp(image):
        lbp = np.zeros_like(image)
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                center = image[i, j]
                code = 0
                code |= (image[i-1, j-1] >= center) << 7
                code |= (image[i-1, j] >= center) << 6
                code |= (image[i-1, j+1] >= center) << 5
                code |= (image[i, j+1] >= center) << 4
                code |= (image[i+1, j+1] >= center) << 3
                code |= (image[i+1, j] >= center) << 2
                code |= (image[i+1, j-1] >= center) << 1
                code |= (image[i, j-1] >= center) << 0
                lbp[i, j] = code
        return lbp
    
    lbp = compute_lbp(face_roi)
    lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    lbp_hist = cv2.normalize(lbp_hist, lbp_hist).flatten()
    
    sobelx = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_hist = cv2.calcHist([gradient_magnitude.astype(np.uint8)], [0], None, [64], [0, 256])
    gradient_hist = cv2.normalize(gradient_hist, gradient_hist).flatten()
    
    region_features = []
    h, w = face_roi.shape
    regions = [
        face_roi[0:h//3, 0:w//2],
        face_roi[0:h//3, w//2:w],
        face_roi[h//3:2*h//3, w//4:3*w//4],
        face_roi[2*h//3:h, w//4:3*w//4]
    ]
    
    for region in regions:
        region_hist = cv2.calcHist([region], [0], None, [32], [0, 256])
        region_hist = cv2.normalize(region_hist, region_hist).flatten()
        region_features.extend(region_hist)
    
    features = np.concatenate([lbp_hist, gradient_hist, region_features])
    return features

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow for Windows reliability

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 6, minSize=(100, 100))
    
    cv2.putText(frame, "Press SPACE to test, Q to quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Test Registered Face", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord(' ') and len(faces) > 0:
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        (x, y, w, h) = largest_face
        face_roi = gray[y:y+h, x:x+w]
        
        features = extract_face_features(face_roi)
        features_norm = features / (np.linalg.norm(features) + 1e-8)
        
        # Test with EXTREMELY STRICT threshold (updated)
        threshold = 0.18  # VERY STRICT to prevent false positives
        min_samples_match = 12  # Must match 12 out of 15
        max_allowed_distance = 0.25
        
        print("\n" + "-"*70)
        print("MATCH TEST RESULTS:")
        print("-"*70)
        
        distances = []
        match_count = 0
        for reg_feature in registered_features:
            reg_norm = reg_feature / (np.linalg.norm(reg_feature) + 1e-8)
            similarity = np.dot(features_norm, reg_norm)
            distance = 1 - similarity
            distances.append(distance)
            if distance < threshold:
                match_count += 1
        
        avg_top5_distance = np.mean(sorted(distances)[:5])
        min_distance = min(distances)
        avg_all_distance = np.mean(distances)
        
        print(f"Matches: {match_count}/{len(registered_features)} (need {min_samples_match})")
        print(f"Avg top-5 distance: {avg_top5_distance:.4f} (threshold: {threshold:.4f})")
        print(f"Avg all distance: {avg_all_distance:.4f} (max: {max_allowed_distance:.4f})")
        print(f"Min distance: {min_distance:.4f}")
        print(f"All distances: {[f'{d:.4f}' for d in sorted(distances)[:10]]}")
        
        # EXTREMELY STRICT matching (updated logic)
        condition1 = match_count >= min_samples_match
        condition2 = avg_top5_distance < threshold
        condition3 = avg_all_distance < max_allowed_distance
        condition4 = min_distance < (threshold * 0.9)
        
        is_owner = condition1 and condition2 and condition3 and condition4
        
        print(f"\nCondition checks:")
        print(f"  1. Match count ({match_count} >= {min_samples_match}): {'✓' if condition1 else '❌'}")
        print(f"  2. Top-5 dist ({avg_top5_distance:.4f} < {threshold:.4f}): {'✓' if condition2 else '❌'}")
        print(f"  3. Avg dist ({avg_all_distance:.4f} < {max_allowed_distance:.4f}): {'✓' if condition3 else '❌'}")
        print(f"  4. Min dist ({min_distance:.4f} < {threshold*0.9:.4f}): {'✓' if condition4 else '❌'}")
        
        if is_owner:
            print("\n✓✓✓ MATCH: Would be recognized as OWNER ✓✓✓")
            match_ratio = match_count / len(registered_features)
            distance_score = max(0, 1 - (avg_top5_distance / threshold))
            confidence = (match_ratio * 0.5 + distance_score * 0.5) * 100
            confidence = max(min(confidence, 90.0), 65.0)
            print(f"  Confidence: {confidence:.1f}%")
        else:
            print("\n❌❌❌ NO MATCH: Would be marked as UNKNOWN ❌❌❌")
            print("\nWhy not recognized:")
            if not condition1:
                print(f"  ❌ Not enough matches: {match_count} < {min_samples_match} required")
            if not condition2:
                print(f"  ❌ Top-5 distance too high: {avg_top5_distance:.4f} >= {threshold:.4f}")
            if not condition3:
                print(f"  ❌ Average distance too high: {avg_all_distance:.4f} >= {max_allowed_distance:.4f}")
            if not condition4:
                print(f"  ❌ Minimum distance too high: {min_distance:.4f} >= {threshold*0.9:.4f}")
            print("\n⚠️ IF THIS IS YOU - YOU NEED TO RE-REGISTER!")
            print("  Run: python recognition.py")
            print("  Choose option 1 and register with:")
            print("    - Very good, direct lighting on your face")
            print("    - Look DIRECTLY at camera (no angles)")
            print("    - Keep face still and consistent")
            print("    - Same distance from camera for all captures")

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*70)
print("Test complete!")
print("="*70)
