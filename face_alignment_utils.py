"""
Face Alignment Utilities for ArcFace
Aligns and crops faces to 112x112 for ArcFace model input
"""

import cv2
import numpy as np
from skimage import transform as trans


# Reference facial points for alignment (5-point landmarks)
# Standard reference points for 112x112 aligned face
SRC_POINTS_112 = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]], dtype=np.float32)


def align_face(img, landmarks, image_size=112):
    """
    Align face using 5-point facial landmarks
    
    Args:
        img: Input face image
        landmarks: 5-point facial landmarks [(x,y), ...] for 
                  left_eye, right_eye, nose, left_mouth, right_mouth
        image_size: Output size (default 112 for ArcFace)
    
    Returns:
        aligned_face: Aligned and cropped face image
    """
    if landmarks is None or len(landmarks) != 5:
        # If no landmarks, just resize
        return cv2.resize(img, (image_size, image_size))
    
    # Convert landmarks to numpy array
    dst = np.array(landmarks, dtype=np.float32)
    
    # Get transformation matrix
    tform = trans.SimilarityTransform()
    tform.estimate(dst, SRC_POINTS_112)
    M = tform.params[0:2, :]
    
    # Apply transformation
    aligned_face = cv2.warpAffine(img, M, (image_size, image_size), 
                                   borderValue=0.0)
    
    return aligned_face


def align_face_simple(img, image_size=112):
    """
    Simple face alignment - just crop and resize
    
    Args:
        img: Input face image (already cropped face region)
        image_size: Output size (default 112 for ArcFace)
    
    Returns:
        aligned_face: Resized face image
    """
    # Ensure square aspect ratio by padding if needed
    h, w = img.shape[:2]
    
    if h != w:
        # Pad to square
        max_dim = max(h, w)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        img = cv2.copyMakeBorder(img, pad_h, max_dim - h - pad_h,
                                 pad_w, max_dim - w - pad_w,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Resize to target size
    aligned_face = cv2.resize(img, (image_size, image_size))
    
    return aligned_face


def preprocess_face(face_roi, image_size=112):
    """
    Preprocess face for ArcFace model
    
    Args:
        face_roi: Face region of interest from detected face
        image_size: Target size (112 for ArcFace)
    
    Returns:
        preprocessed_face: Ready for model input
    """
    # Align (simple version)
    aligned = align_face_simple(face_roi, image_size)
    
    # Optional: Apply histogram equalization for better quality
    # Convert to LAB color space
    lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return preprocessed


# Optional: 68-point landmark estimation for better alignment
def estimate_5point_from_bbox(bbox, frame_shape):
    """
    Estimate 5-point landmarks from bounding box
    Simple estimation based on face proportions
    
    Args:
        bbox: Face bounding box (top, right, bottom, left)
        frame_shape: Shape of the frame (height, width, channels)
    
    Returns:
        landmarks: Estimated 5-point landmarks
    """
    top, right, bottom, left = bbox
    
    face_width = right - left
    face_height = bottom - top
    
    # Estimate landmark positions based on typical face proportions
    landmarks = [
        (left + face_width * 0.35, top + face_height * 0.35),  # Left eye
        (left + face_width * 0.65, top + face_height * 0.35),  # Right eye
        (left + face_width * 0.50, top + face_height * 0.55),  # Nose
        (left + face_width * 0.38, top + face_height * 0.75),  # Left mouth
        (left + face_width * 0.62, top + face_height * 0.75)   # Right mouth
    ]
    
    return landmarks


if __name__ == "__main__":
    print("Face alignment utilities loaded successfully!")
    print("Functions available:")
    print("  - align_face(img, landmarks)")
    print("  - align_face_simple(img)")
    print("  - preprocess_face(face_roi)")
    print("  - estimate_5point_from_bbox(bbox, frame_shape)")
