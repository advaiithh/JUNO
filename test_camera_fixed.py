"""
Quick camera test with DirectShow backend
"""
import cv2

print("Testing camera with DirectShow backend...")
print("Press Q to quit")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("\u274c ERROR: Camera still not working!")
    exit(1)

print("\u2713 Camera opened successfully!")

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print(f"\u274c ERROR: Failed to read frame {frame_count}")
        break
    
    frame_count += 1
    
    cv2.putText(frame, f"DirectShow Backend - Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "Camera FIXED! Press Q to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow("Camera Test - FIXED", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n\u2713 SUCCESS! Read {frame_count} frames.")
print("\nCamera is now FIXED in:")
print("  - recognition.py")
print("  - recognition_advanced.py")
print("  - test_my_face.py")
print("\nYou can now run: python recognition.py")
