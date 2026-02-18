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

# Initialize HOG person detector
HOG_PERSON_DETECTOR = cv2.HOGDescriptor()
HOG_PERSON_DETECTOR.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

class RoomOccupancyTracker:
    """Track people entering and exiting the room"""
    
    def __init__(self):
        self.people_in_room = {}  # {person_name: entry_time}
        self.people_last_seen = {}  # {person_name: timestamp}
        self.unknown_faces = {}  # {face_id: last_seen_time}
        self.detected_bodies = {}  # {body_id: last_seen_time} - for distant people
        self.unknown_counter = 0  # Counter for unknown person IDs
        self.body_counter = 0  # Counter for body-only detections
        self.exit_timeout = 5  # Seconds before considering person left
        self.entry_log = []
        self.exit_log = []
        self.total_visitors = 0
        self.total_unknown_visitors = 0
        self.total_body_detections = 0
        
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
                    self.total_unknown_visitors = data.get('total_unknown_visitors', 0)
                    self.total_body_detections = data.get('total_body_detections', 0)
            except:
                pass
    
    def save_logs(self):
        """Save occupancy logs to file"""
        os.makedirs(os.path.dirname(OCCUPANCY_LOG_FILE), exist_ok=True)
        data = {
            'entry_log': self.entry_log[-100:],  # Keep last 100 entries
            'exit_log': self.exit_log[-100:],
            'total_visitors': self.total_visitors,
            'total_unknown_visitors': self.total_unknown_visitors,
            'total_body_detections': self.total_body_detections,
            'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(OCCUPANCY_LOG_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    
    def body_detected(self, body_id):
        """Called when a person body is detected (no face visible)"""
        current_time = time.time()
        
        # Update last seen time for this body
        if body_id not in self.detected_bodies:
            self.detected_bodies[body_id] = current_time
            self.total_body_detections += 1
            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.entry_log.append({
                'person': f'Person_Body_{body_id}',
                'action': 'DETECTED',
                'timestamp': entry_time
            })
            print(f"\n👤 Person detected (body only) #{body_id} at {entry_time}")
            print(f"  Current occupancy: {self.get_occupancy_count()} person(s)")
            self.save_logs()
        else:
            self.detected_bodies[body_id] = current_time
    
    def unknown_person_detected(self, face_id):
        """Called when an unknown person is detected"""
        current_time = time.time()
        
        # Update last seen time for this unknown face
        if face_id not in self.unknown_faces:
            self.unknown_faces[face_id] = current_time
            self.total_unknown_visitors += 1
            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.entry_log.append({
                'person': f'Unknown_{face_id}',
                'action': 'ENTERED',
                'timestamp': entry_time
            })
            print(f"\n⚠ Unknown person #{face_id} detected at {entry_time}")
            print(f"  Current occupancy: {self.get_occupancy_count()} person(s)")
            self.save_logs()
        else:
            self.unknown_faces[face_id] = current_time
    
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
        unknown_to_remove = []
        bodies_to_remove = []
        
        # Check registered people
        for person_name, last_seen in self.people_last_seen.items():
            # If person hasn't been seen for exit_timeout seconds
            if current_time - last_seen > self.exit_timeout:
                if person_name in self.people_in_room:
                    people_to_remove.append(person_name)
        
        # Check unknown people
        for face_id, last_seen in list(self.unknown_faces.items()):
            if current_time - last_seen > self.exit_timeout:
                unknown_to_remove.append(face_id)
        
        # Check detected bodies
        for body_id, last_seen in list(self.detected_bodies.items()):
            if current_time - last_seen > self.exit_timeout:
                bodies_to_remove.append(body_id)
        
        # Remove people who left
        for person_name in people_to_remove:
            del self.people_in_room[person_name]
            exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.exit_log.append({
                'person': person_name,
                'action': 'EXITED',
                'timestamp': exit_time
            })
            print(f"\n✗ {person_name} EXITED the room at {exit_time}")
            print(f"  Current occupancy: {self.get_occupancy_count()} person(s)")
            self.save_logs()
        
        # Remove unknown people who left
        for face_id in unknown_to_remove:
            del self.unknown_faces[face_id]
            exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.exit_log.append({
                'person': f'Unknown_{face_id}',
                'action': 'EXITED',
                'timestamp': exit_time
            })
            print(f"\n✗ Unknown person #{face_id} exited at {exit_time}")
            print(f"  Current occupancy: {self.get_occupancy_count()} person(s)")
            self.save_logs()
        
        # Remove bodies that left
        for body_id in bodies_to_remove:
            del self.detected_bodies[body_id]
            exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.exit_log.append({
                'person': f'Person_Body_{body_id}',
                'action': 'LEFT',
                'timestamp': exit_time
            })
            print(f"\n✗ Person (body #{body_id}) left at {exit_time}")
            print(f"  Current occupancy: {self.get_occupancy_count()} person(s)")
            self.save_logs()
    
    def get_occupancy_count(self):
        """Get current number of people in room (including unknowns and bodies)"""
        return len(self.people_in_room) + len(self.unknown_faces) + len(self.detected_bodies)
    
    def get_people_in_room(self):
        """Get list of people currently in room"""
        return list(self.people_in_room.keys())
    
    def get_summary(self):
        """Get occupancy summary"""
        return {
            'current_occupancy': self.get_occupancy_count(),
            'known_people': list(self.people_in_room.keys()),
            'unknown_count': len(self.unknown_faces),
            'body_only_count': len(self.detected_bodies),
            'total_visitors_today': self.total_visitors,
            'total_unknown_visitors': self.total_unknown_visitors,
            'total_body_detections': self.total_body_detections,
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


def detect_people_bodies(frame):
    """Detect full bodies/people even at distance using HOG"""
    # Resize frame for faster detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    # Detect people
    bodies, weights = HOG_PERSON_DETECTOR.detectMultiScale(
        small_frame, 
        winStride=(8, 8),
        padding=(4, 4),
        scale=1.05,
        finalThreshold=1.5
    )
    
    # Scale back to original size
    scaled_bodies = []
    for (x, y, w, h) in bodies:
        scaled_bodies.append((x*2, y*2, w*2, h*2))
    
    return scaled_bodies


def detect_faces_simple(frame):
    """Simple face detection"""
    if USE_FACE_RECOGNITION:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog')
        return face_locations
    elif USE_INSIGHTFACE:
        # Use InsightFace detector
        try:
            faces, _ = INSIGHTFACE_MODEL.detector.detect(frame)
            if faces is not None and len(faces) > 0:
                face_locations = []
                for face in faces:
                    x1, y1, x2, y2, score = face
                    face_locations.append((int(y1), int(x2), int(y2), int(x1)))
                return face_locations
        except:
            pass
    
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
            
            # Detect bodies (people at distance)
            body_locations = detect_people_bodies(frame)
            
            detected_people = set()
            unknown_face_id = 0
            body_id = 0
            
            # Process face detections first (higher priority)
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
                        # Unknown person - track them too
                        unknown_face_id += 1
                        tracker.unknown_person_detected(unknown_face_id)
                        
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 2)
                        cv2.putText(frame, f"Unknown #{unknown_face_id}", (left, top-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                        cv2.putText(frame, "Counting in occupancy", (left, bottom+25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    # Face detected but no encoding - still count it
                    cv2.rectangle(frame, (left, top), (right, bottom), (128, 128, 128), 2)
                    cv2.putText(frame, "Detected", (left, top-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
            
            # Process body detections (people at distance without clear face)
            for (x, y, w, h) in body_locations:
                # Check if this body overlaps with any detected face
                is_overlapping = False
                for face_location in face_locations:
                    face_top, face_right, face_bottom, face_left = face_location
                    # Check if body area contains this face
                    if (face_left >= x and face_right <= x+w and 
                        face_top >= y and face_bottom <= y+h):
                        is_overlapping = True
                        break
                
                # Only count body if no face was detected in this area
                if not is_overlapping:
                    body_id += 1
                    tracker.body_detected(body_id)
                    
                    # Draw body rectangle
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                    cv2.putText(frame, f"Person #{body_id}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, "(Body Detection)", (x, y+h+25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Check for exits
            tracker.check_exits()
        
        # Display occupancy info
        occupancy = tracker.get_occupancy_count()
        known_people = tracker.get_people_in_room()
        unknown_count = len(tracker.unknown_faces)
        body_count = len(tracker.detected_bodies)
        
        # Occupancy counter
        cv2.rectangle(frame, (10, 10), (450, 180), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 180), (255, 255, 255), 2)
        
        cv2.putText(frame, f"TOTAL OCCUPANCY: {occupancy}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Show known people
        if known_people:
            people_text = ", ".join(known_people[:2])
            if len(known_people) > 2:
                people_text += f" +{len(known_people)-2}"
            cv2.putText(frame, f"Known: {people_text}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Known: None", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Show unknown count
        if unknown_count > 0:
            cv2.putText(frame, f"Unknown Faces: {unknown_count}", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "Unknown Faces: 0", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Show body-only detections
        if body_count > 0:
            cv2.putText(frame, f"Distant People: {body_count}", (20, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Distant People: 0", (20, 145),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Total visitors
        cv2.putText(frame, f"Total: {tracker.total_visitors}K + {tracker.total_unknown_visitors}U + {tracker.total_body_detections}D", (20, 170),
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
    print(f"Known People: {', '.join(summary['known_people']) if summary['known_people'] else 'None'}")
    print(f"Unknown Faces: {summary['unknown_count']}")
    print(f"Distant People (Body Only): {summary['body_only_count']}")
    print(f"Total Known Visitors: {summary['total_visitors_today']}")
    print(f"Total Unknown Visitors: {summary['total_unknown_visitors']}")
    print(f"Total Body Detections: {summary['total_body_detections']}")
    print("="*70)


if __name__ == "__main__":
    main()
