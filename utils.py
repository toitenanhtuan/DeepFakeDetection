import cv2
import numpy as np
import base64
from deepfake_detector import DeepfakeDetector

# Initialize deepfake detector
detector = DeepfakeDetector()

def process_video(video_path):
    """Process video file and extract frames and faces."""
    frames = []
    face_frames = []
    original_frames = []  # Store original frames for analysis

    # Load the cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 10 == 0:  # Process every 10th frame
            # Store original frame for analysis
            original_frames.append(frame)

            # Convert to base64 for display, keeping original colors
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            frames.append(frame_base64)

            # Detect faces
            faces = detect_faces(frame, face_cascade)
            face_frames.extend(faces)

        frame_count += 1

    cap.release()
    return frames, face_frames, original_frames

def detect_faces(frame, face_cascade):
    """Detect faces in a frame using OpenCV."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    face_images = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        # Convert to base64, keeping original colors
        _, buffer = cv2.imencode('.jpg', face_img)
        face_base64 = base64.b64encode(buffer).decode('utf-8')
        face_images.append(face_base64)

    return face_images

def analyze_deepfake(frames, original_frames):
    """Analyze frames using the deepfake detection model."""
    try:
        result, confidence = detector.analyze_video_frames(original_frames)
        return result
    except Exception as e:
        print(f"Error in deepfake analysis: {str(e)}")
        return "ERROR"