import cv2
import numpy as np
import os
import pickle
import time
import json
from datetime import datetime
from collections import defaultdict
import openvino as ov

# Import DeepSORT utilities
from deepsort_utils.tracker import Tracker
from deepsort_utils.nn_matching import NearestNeighborDistanceMetric
from deepsort_utils.detection import Detection, xywh_to_tlwh

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
PERSON_MODEL_XML = "models/person_detection/person-detection-retail-0013.xml"

# Initialize OpenVINO person detector
print("Loading OpenVINO person detection model...")
core = ov.Core()
person_model = core.read_model(model=PERSON_MODEL_XML)
compiled_person_model = core.compile_model(model=person_model, device_name="CPU")
person_input_layer = compiled_person_model.input(0)
person_output_layer = compiled_person_model.output(0)

# Initialize DeepSORT tracker
max_cosine_distance = 0.4
nn_budget = None
metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
deepsort_tracker = Tracker(metric, max_iou_distance=0.7, max_age=30, n_init=3)

print("✓ OpenVINO model loaded and ready!")

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
    """Detect people using OpenVINO person detection model"""
    # Get input shape
    n, c, h, w = person_input_layer.shape
    
    # Preprocess frame
    resized_frame = cv2.resize(frame, (w, h))
    input_frame = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)
    
    # Run inference
    results = compiled_person_model([input_frame])[person_output_layer]
    
    # Parse detections
    # Output shape: [1, 1, N, 7] where N is number of detections
    # Each detection: [image_id, label, conf, x_min, y_min, x_max, y_max]
    detections = []
    frame_height, frame_width = frame.shape[:2]
    
    detection_count = 0
    for detection in results[0][0]:
        confidence = detection[2]
        if confidence > 0.3:  # Lowered confidence threshold for better detection
            detection_count += 1
            x_min = int(detection[3] * frame_width)
            y_min = int(detection[4] * frame_height)
            x_max = int(detection[5] * frame_width)
            y_max = int(detection[6] * frame_height)
            
            # Convert to (x, y, w, h) format
            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min
            
            # Validate bounding box
            if w > 20 and h > 40:  # Minimum size filter
                detections.append((x, y, w, h, confidence))
    
    # Debug: Print detection count periodically
    if detection_count > 0:
        print(f"[DEBUG] Detected {detection_count} people with confidence > 0.3")
    
    return detections
    
    return detections


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
    
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"✓ Camera opened: {int(actual_width)}x{int(actual_height)}")
    print("  Detection confidence threshold: 0.3")
    print("  Processing every 3rd frame\n")
    
    frame_count = 0
    process_every_n_frames = 3  # Process every 3rd frame (more frequent)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process recognition every N frames
        if frame_count % process_every_n_frames == 0:
            # Detect bodies/people using OpenVINO (PRIMARY DETECTION)
            body_locations = detect_people_bodies(frame)
            
            detected_people = set()
            
            # Update DeepSORT tracker with body detections
            if body_locations:
                # Create Detection objects for DeepSORT
                detections_list = []
                for (x, y, w, h, confidence) in body_locations:
                    # Convert to tlwh format for DeepSORT
                    bbox = [x, y, w, h]
                    # Use simple feature vector (position + size normalized)
                    feature = np.array([x/frame.shape[1], y/frame.shape[0], 
                                       w/frame.shape[1], h/frame.shape[0]])
                    detection = Detection(bbox, confidence, feature)
                    detections_list.append(detection)
                
                # Update tracker
                deepsort_tracker.predict()
                deepsort_tracker.update(detections_list)
                
                # Draw tracked people (HUMAN DETECTION - BODY BOXES)
                for track in deepsort_tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    
                    bbox = track.to_tlbr()
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    
                    # Extract the person's region for face recognition (optional identification)
                    person_roi = frame[max(0,y1):min(frame.shape[0],y2), 
                                      max(0,x1):min(frame.shape[1],x2)]
                    
                    person_name = None
                    if person_roi.size > 0:
                        # Try to detect face inside body box for identification
                        face_locations = detect_faces_simple(person_roi)
                        
                        if face_locations:
                            # Get the largest face (closest to camera)
                            face_location = face_locations[0]
                            top, right, bottom, left = face_location
                            
                            # Adjust coordinates relative to full frame
                            face_top = y1 + top
                            face_left = x1 + left
                            face_bottom = y1 + bottom
                            face_right = x1 + right
                            
                            # Extract encoding for identification
                            encoding = extract_face_encoding_simple(frame, 
                                (face_top, face_right, face_bottom, face_left))
                            
                            if encoding is not None:
                                # Recognize person
                                person_name, distance = recognize_person(encoding, all_people)
                                
                                if person_name:
                                    detected_people.add(person_name)
                                    status = tracker.person_detected(person_name)
                                    
                                    # Draw BODY rectangle with identified name
                                    color = (0, 255, 0) if status == "ENTERED" else (0, 200, 0)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                                    cv2.putText(frame, f"{person_name} (ID:{track.track_id})", 
                                               (x1, y1-10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                                    cv2.putText(frame, f"Confidence: {distance:.3f}", 
                                               (x1, y2+25),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                                    continue
                    
                    # If no face or unknown person - show as generic human detection
                    if person_name is None:
                        tracker.body_detected(track.track_id)
                        
                        # Draw BODY rectangle for detected human
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                        cv2.putText(frame, f"Person ID:{track.track_id}", (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                        cv2.putText(frame, "Human Detected", (x1, y2+25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Check for exits
            tracker.check_exits()
        
        # Display occupancy info
        occupancy = tracker.get_occupancy_count()
        known_people = tracker.get_people_in_room()
        unknown_count = len(tracker.unknown_faces)
        body_count = len(tracker.detected_bodies)
        
        # Occupancy counter
        cv2.rectangle(frame, (10, 10), (450, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 200), (255, 255, 255), 2)
        
        # Show detection status
        if occupancy == 0:
            cv2.putText(frame, f"TOTAL OCCUPANCY: {occupancy}", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 3)
            cv2.putText(frame, "[Scanning for humans...]", (20, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        else:
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
        
        # Show unidentified humans count
        total_unidentified = unknown_count + body_count
        if total_unidentified > 0:
            cv2.putText(frame, f"Unidentified Humans: {total_unidentified}", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Unidentified Humans: 0", (20, 115),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
        
        # Detection method info
        cv2.putText(frame, "[OpenVINO Human Detection]", (20, 145),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        # Total humans detected (all sessions)
        total_all_humans = tracker.total_visitors + tracker.total_unknown_visitors + tracker.total_body_detections
        cv2.putText(frame, f"Total Humans Detected: {total_all_humans}", (20, 170),
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
