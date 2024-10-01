import os
import cv2
import numpy as np
from tensorflow import keras

# Step 1: Load the trained model
model_path = './ai_face_detector_model.keras'  # Path to your saved model
model = keras.models.load_model(model_path)
print("Model loaded successfully.")

# Step 2: Load and preprocess the test video
def load_and_preprocess_test_video(video_path, frame_count=10):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Couldn't open video {video_path}")
        return None
    
    frames = []
    frame_idx = 0
    while frame_idx < frame_count and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (128, 128))  # Resize to match model input
            frames.append(frame)
            frame_idx += 1
        else:
            break

    cap.release()
    
    if len(frames) == frame_count:
        frames = np.array(frames) / 255.0  # Normalize frames
        frames = np.expand_dims(frames, axis=0)  # Add batch dimension
        return frames
    else:
        print(f"Error: Not enough frames extracted from {video_path}")
        return None

# Step 3: Test the model on a specific video
def test_video(video_path):
    test_video = load_and_preprocess_test_video(video_path)
    
    if test_video is not None:
        prediction = model.predict(test_video)
        predicted_label = 1 if prediction[0][0] > 0.5 else 0  # Threshold of 0.5 for binary classification
        result = "Deepfake" if predicted_label == 1 else "Real"
        print(f"The video '{os.path.basename(video_path)}' is predicted to be: {result}")

# Main execution
if __name__ == "__main__":
    test_video_path = '../data/test_video.mp4'  # Path to your test video
    test_video(test_video_path)
