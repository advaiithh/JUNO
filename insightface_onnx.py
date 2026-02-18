"""
InsightFace ONNX Face Recognition
==================================

Uses InsightFace's pre-built ONNX models for face recognition.
No C++ compilation required!

Models automatically downloaded from: https://github.com/deepinsight/insightface/releases
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from urllib.request import urlretrieve
import sys

# Model configuration
MODELS_DIR = "buffalo_l"  # Models are in buffalo_l folder at root

# Required models (already downloaded locally)
REQUIRED_MODELS = {
    "det_10g.onnx": None,  # Local file
    "w600k_r50.onnx": None,  # Local file
}


def download_progress(block_num, block_size, total_size):
    """Show download progress"""
    downloaded = block_num * block_size
    percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
    bar = "#" * int(percent / 2)
    sys.stdout.write(f"\r[{bar:<50}] {percent:.1f}%")
    sys.stdout.flush()


def download_models():
    """Check if InsightFace models exist locally"""
    if not os.path.exists(MODELS_DIR):
        print(f"✗ Models directory not found: {MODELS_DIR}")
        return False
    
    all_exist = True
    for model_name in REQUIRED_MODELS.keys():
        model_path = os.path.join(MODELS_DIR, model_name)
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"✓ {model_name} found ({file_size:.1f} MB)")
        else:
            print(f"✗ {model_name} not found at {model_path}")
            all_exist = False
    
    return all_exist


class InsightFaceRecognition:
    """Face recognition using InsightFace ONNX models"""
    
    def __init__(self):
        self.detector = None
        self.recognizer = None
        self.input_size = (112, 112)  # ArcFace input size
        
    def load_models(self):
        """Load ONNX models"""
        det_path = os.path.join(MODELS_DIR, "det_10g.onnx")
        rec_path = os.path.join(MODELS_DIR, "w600k_r50.onnx")
        
        if not os.path.exists(det_path) or not os.path.exists(rec_path):
            print("Models not found. Checking...")
            if not download_models():
                print(f"\nPlease ensure models are in: {MODELS_DIR}/")
                return False
        
        try:
            print("\nLoading models...")
            # Load face detector
            self.detector = ort.InferenceSession(
                det_path,
                providers=['CPUExecutionProvider']
            )
            print(f"✓ Face detector loaded")
            
            # Load face recognizer (ArcFace w600k_r50)
            self.recognizer = ort.InferenceSession(
                rec_path,
                providers=['CPUExecutionProvider']
            )
            print(f"✓ Face recognizer loaded (InsightFace ArcFace-R50)")
            print("✓ InsightFace ONNX system ready!")
            print("  - Using 512-dimensional embeddings")
            print("  - State-of-the-art accuracy")
            return True
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            return False
    
    def detect_faces(self, image):
        """Detect faces in image"""
        if self.detector is None:
            return []
        
        try:
            # Prepare input
            input_size = (640, 640)
            img = cv2.resize(image, input_size)
            blob = cv2.dnn.blobFromImage(img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)
            
            # Run detection
            input_name = self.detector.get_inputs()[0].name
            outputs = self.detector.run(None, {input_name: blob})
            
            # Parse detections
            faces = []
            for output in outputs:
                if output.shape[1] >= 15:  # Has bbox + landmarks
                    for detection in output:
                        if detection[4] > 0.5:  # Confidence threshold
                            # Scale bbox back to original size
                            h, w = image.shape[:2]
                            bbox = detection[:4].copy()
                            bbox[0] *= w / input_size[0]
                            bbox[1] *= h / input_size[1]
                            bbox[2] *= w / input_size[0]
                            bbox[3] *= h / input_size[1]
                            
                            faces.append({
                                'bbox': bbox.astype(int),
                                'confidence': detection[4]
                            })
            
            return faces
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def align_face(self, image, bbox):
        """Align face to 112x112 for ArcFace"""
        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Resize to 112x112
        face = cv2.resize(face, self.input_size)
        return face
    
    def extract_embedding(self, face_img):
        """Extract 512-D face embedding using ArcFace"""
        if self.recognizer is None or face_img is None:
            return None
        
        try:
            # Prepare input (normalize to [-1, 1])
            blob = cv2.dnn.blobFromImage(
                face_img, 
                1.0/127.5, 
                self.input_size, 
                (127.5, 127.5, 127.5), 
                swapRB=True
            )
            
            # Extract embedding
            input_name = self.recognizer.get_inputs()[0].name
            embedding = self.recognizer.run(None, {input_name: blob})[0]
            
            # Normalize (L2)
            embedding = embedding.flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Embedding extraction error: {e}")
            return None
    
    def get_face_embedding(self, image, bbox=None):
        """Get face embedding from image (with optional bbox)"""
        if bbox is None:
            # Detect face first
            faces = self.detect_faces(image)
            if not faces:
                return None
            bbox = faces[0]['bbox']
        
        # Align face
        face = self.align_face(image, bbox)
        if face is None:
            return None
        
        # Extract embedding
        return self.extract_embedding(face)
    
    @staticmethod
    def compare_embeddings(emb1, emb2):
        """Compare two embeddings using cosine similarity"""
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity (already normalized, so just dot product)
        similarity = np.dot(emb1, emb2)
        
        # Convert to distance (0 = same, 1 = different)
        distance = 1.0 - similarity
        
        return distance


def test_insightface():
    """Test InsightFace ONNX system"""
    print("=" * 60)
    print("InsightFace ONNX Test")
    print("=" * 60)
    
    # Initialize
    recognizer = InsightFaceRecognition()
    
    if not recognizer.load_models():
        print("\n✗ Failed to load models")
        print(f"\nPlease ensure buffalo_l folder contains:")
        print("  - det_10g.onnx")
        print("  - w600k_r50.onnx")
        return
    
    # Test with webcam
    print("\n" + "=" * 60)
    print("Testing with webcam...")
    print("Press 'q' to quit")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        faces = recognizer.detect_faces(frame)
        
        # Draw detections
        for face in faces:
            bbox = face['bbox']
            conf = face['confidence']
            
            # Draw bbox
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            
            # Extract embedding
            embedding = recognizer.get_face_embedding(frame, bbox)
            
            if embedding is not None:
                text = f"Face: {conf:.2f}, Dim: {len(embedding)}"
                cv2.putText(frame, text, 
                           (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 255, 0), 2)
        
        # Show
        cv2.imshow("InsightFace ONNX Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✓ Test complete!")


if __name__ == "__main__":
    test_insightface()
