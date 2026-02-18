import cv2
import numpy as np
import os
import pickle
import time
import json
from datetime import datetime
from collections import defaultdict

# Try to import face recognition libraries
try:
    import face_recognition
    USE_FACE_RECOGNITION = True
except ImportError:
    USE_FACE_RECOGNITION = False

# Try to import InsightFace
try:
    from insightface_onnx import InsightFaceRecognition
    INSIGHTFACE_MODEL = InsightFaceRecognition()
    if INSIGHTFACE_MODEL.load_models():
        USE_INSIGHTFACE = True
    else:
        USE_INSIGHTFACE = False
        INSIGHTFACE_MODEL = None
except:
    USE_INSIGHTFACE = False
    INSIGHTFACE_MODEL = None

# File paths
REGISTERED_FACES_FILE = "registered_faces_advanced.pkl"
OCCUPANCY_LOG_FILE = "logs/occupancy_log.json"

class RoomOccupancyTracker:
    """Track people entering and exiting the room"""
    
    def __init__(self):
        self.people_in_room = {}  # {person_name: entry_time}
        self.people_last_seen = {}  # {person_name: timestamp}
        self.exit_timeout = 5  # Seconds before considering person left
        self.entry_log = []
        self.exit_log = []
        self.total_visitors = 0
        
        # Load existing logs if available
        self.load_logs()
    
    def load_logs(self):
        """Load previous occupancy logs"""
        if os.path.exists(OCCUPANCY_LOG_FILE):
            try:
                with open(OCCUPANCY_LOG_FILE, 'r') as f:
                    data = json.load(f)
                    self.entry_log = data.get('entry_log', [])
                    self.exit_log = data.get('exit_log', [])
                    self.total_visitors = data.get('total_visitors', 0)
            except:
                pass
    
    def save_logs(self):
        """Save occupancy logs to file"""
        os.makedirs(os.path.dirname(OCCUPANCY_LOG_FILE), exist_ok=True)
        data = {
            'entry_log': self.entry_log[-100:],  # Keep last 100 entries
            'exit_log': self.exit_log[-100:],
            'total_visitors': self.total_visitors,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(OCCUPANCY_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def person_detected(self, person_name):
        """Called when a person is detected in frame"""
        current_time = time.time()
        
        # Update last seen time
        self.people_last_seen[person_name] = current_time
        
        # Check if this is a new entry
        if person_name not in self.people_in_room:
            self.people_in_room[person_name] = current_time
            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.entry_log.append({
                'person': person_name,
                'action': 'ENTERED',
                'timestamp': entry_time
            })
            self.total_visitors += 1
            print(f"\n✓ {person_name} ENTERED the room at {entry_time}")
            print(f"  Current occupancy: {len(self.people_in_room)} person(s)")
            self.save_logs()
            return "ENTERED"
        return "PRESENT"
    
    def check_exits(self):
        """Check if anyone has left the room"""
        current_time = time.time()
        people_to_remove = []
        
        for person_name, last_seen in self.people_last_seen.items():
            # If person hasn't been seen for exit_timeout seconds
            if current_time - last_seen > self.exit_timeout:
                if person_name in self.people_in_room:
                    people_to_remove.append(person_name)
        
        for person_name in people_to_remove:
            del self.people_in_room[person_name]
            exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.exit_log.append({
                'person': person_name,
                'action': 'EXITED',
                'timestamp': exit_time
            })
            print(f"\n✗ {person_name} EXITED the room at {exit_time}")
            print(f"  Current occupancy: {len(self.people_in_room)} person(s)")
            self.save_logs()
    
    def get_occupancy_count(self):
        """Get current number of people in room"""
        return len(self.people_in_room)
    
    def get_people_in_room(self):
        """Get list of people currently in room"""
        return list(self.people_in_room.keys())
    
    def get_summary(self):
        """Get occupancy summary"""
        return {
            'current_occupancy': len(self.people_in_room),
            'people_present': list(self.people_in_room.keys()),
            'total_visitors_today': self.total_visitors,
            'recent_entries': self.entry_log[-5:],
            'recent_exits': self.exit_log[-5:]
        }


def load_registered_people():
    """Load registered people from file"""
    if not os.path.exists(REGISTERED_FACES_FILE):
        print("❌ No registered faces found! Please register faces first.")
        return None
    
    with open(REGISTERED_FACES_FILE, 'rb') as f:
        registered_data = pickle.load(f)
    
    if "people" in registered_data:
        all_people = registered_data["people"]
        print(f"\n✓ Loaded {len(all_people)} registered people:")
        for name in all_people.keys():
            print(f"  - {name}")
        return all_people
    else:
        # Old format
        print("⚠ Old format detected. Please re-register faces.")
        return None


def detect_faces_simple(frame):
    """Simple face detection"""
    if USE_FACE_RECOGNITION:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        return face_locations
    elif USE_INSIGHTFACE:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = INSIGHTFACE_MODEL.detector.detect_faces(frame)
        if faces is not None and len(faces) > 0:
            face_locations = []
            for face in faces:
                bbox = face['bbox']
                x, y, w, h = bbox
                face_locations.append((y, x+w, y+h, x))
            return face_locations
    else:
        # Fallback to Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_locations = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x+w, y+h, x))
        return face_locations


def extract_face_encoding_simple(frame, face_location):
    """Extract face encoding"""
    top, right, bottom, left = face_location
    
    if USE_FACE_RECOGNITION:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_frame, [face_location])
        if len(encodings) > 0:
            return encodings[0]
    elif USE_INSIGHTFACE:
        bbox = [left, top, right-left, bottom-top]
        embedding = INSIGHTFACE_MODEL.get_face_embedding(frame, bbox)
        return embedding
    
    return None


def recognize_person(encoding, all_people, threshold=0.40):
    """Recognize which person this encoding belongs to"""
    best_person = None
    best_distance = float('inf')
    
    for person_name, person_data in all_people.items():
        person_encodings = person_data["encodings"]
        
        # Compare with this person's encodings
        if USE_FACE_RECOGNITION:
            distances = face_recognition.face_distance(person_encodings, encoding)
        else:
            distances = []
            for reg_encoding in person_encodings:
                similarity = np.dot(encoding, reg_encoding) / (np.linalg.norm(encoding) * np.linalg.norm(reg_encoding) + 1e-6)
                distance = 1 - similarity
                distances.append(distance)
        
        min_distance = min(distances) if len(distances) > 0 else float('inf')
        
        # Keep best match
        if min_distance < best_distance and min_distance < threshold:
            best_distance = min_distance
            best_person = person_name
    
    return best_person, best_distance


def main():
    """Main room occupancy tracking function"""
    print("\n" + "="*70)
    print("ROOM OCCUPANCY TRACKING SYSTEM")
    print("="*70)
    
    # Load registered people
    all_people = load_registered_people()
    if all_people is None:
        return
    
    # Initialize tracker
    tracker = RoomOccupancyTracker()
    
    print("\n✓ Starting occupancy tracking...")
    print(f"  Exit timeout: {tracker.exit_timeout} seconds")
    print("  Press Q to quit\n")
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    frame_count = 0
    process_every_n_frames = 5  # Process every 5th frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process recognition every N frames
        if frame_count % process_every_n_frames == 0:
            # Detect faces
            face_locations = detect_faces_simple(frame)
            
            detected_people = set()
            
            for face_location in face_locations:
                top, right, bottom, left = face_location
                
                # Extract encoding
                encoding = extract_face_encoding_simple(frame, face_location)
                
                if encoding is not None:
                    # Recognize person
                    person_name, distance = recognize_person(encoding, all_people)
                    
                    if person_name:
                        detected_people.add(person_name)
                        status = tracker.person_detected(person_name)
                        
                        # Draw rectangle and name
                        color = (0, 255, 0) if status == "ENTERED" else (0, 200, 0)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, person_name, (left, top-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        cv2.putText(frame, f"Dist: {distance:.3f}", (left, bottom+25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    else:
                        # Unknown person
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (left, top-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # Check for exits
            tracker.check_exits()
        
        # Display occupancy info
        occupancy = tracker.get_occupancy_count()
        people_in_room = tracker.get_people_in_room()
        
        # Occupancy counter
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (350, 120), (255, 255, 255), 2)
        
        cv2.putText(frame, f"OCCUPANCY: {occupancy}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        if people_in_room:
            people_text = ", ".join(people_in_room[:3])
            if len(people_in_room) > 3:
                people_text += f" +{len(people_in_room)-3} more"
            cv2.putText(frame, f"Present: {people_text}", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "No one in room", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Total visitors
        cv2.putText(frame, f"Total Visitors: {tracker.total_visitors}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Room Occupancy Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    summary = tracker.get_summary()
    print("\n" + "="*70)
    print("SESSION SUMMARY")
    print("="*70)
    print(f"Final Occupancy: {summary['current_occupancy']} person(s)")
    print(f"People Present: {', '.join(summary['people_present']) if summary['people_present'] else 'None'}")
    print(f"Total Visitors: {summary['total_visitors_today']}")
    print("="*70)


if __name__ == "__main__":
    main()
