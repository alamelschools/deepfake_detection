import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint

# Custom progress callback to monitor training progress
class ProgressCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        total_epochs = self.params['epochs']
        percentage = (epoch + 1) / total_epochs * 100
        print(f"\nEpoch {epoch + 1}/{total_epochs} - {percentage:.2f}% complete")

# Step 1: Load and preprocess videos (load fewer frames per video to save memory)
def load_and_preprocess_videos(directory, label, frame_count=5):
    images = []
    labels = []

    for video_name in os.listdir(directory):
        video_path = os.path.join(directory, video_name)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Couldn't open video {video_name}")
            continue

        frames = []
        frame_idx = 0
        while frame_idx < frame_count and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (128, 128))  # Resize to 128x128
                frames.append(frame)
                frame_idx += 1
            else:
                break

        cap.release()

        if len(frames) == frame_count:  # Only keep videos that had enough frames
            images.append(frames)
            labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    images = images / 255.0  # Normalize frames to reduce processing time
    return images, labels

# Load datasets (real and deepfake videos)
real_videos_dir = r"./data/real_videos"
deepfake_videos_dir = r"./data/deepfake_videos"

# Load and preprocess videos from both directories
real_images, real_labels = load_and_preprocess_videos(real_videos_dir, label=0)  # Label 0 for real videos
deepfake_images, deepfake_labels = load_and_preprocess_videos(deepfake_videos_dir, label=1)  # Label 1 for deepfake videos

# Combine datasets
images = np.concatenate((real_images, deepfake_images))
labels = np.concatenate((real_labels, deepfake_labels))

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 3: Build a 3D Convolutional Neural Network model to process video frames
inputs = keras.Input(shape=(5, 128, 128, 3))  # Use Input layer for clarity
model = keras.Sequential([
    layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=(5, 128, 128, 3)),  # Adjusted padding to 'same'
    layers.MaxPooling3D(pool_size=(2, 2, 2)),
    layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),  # Adjusted padding to 'same'
    layers.MaxPooling3D(pool_size=(2, 2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model with binary cross-entropy and accuracy metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Load model weights if they already exist
model_path = './ai_face_detector_model.h5'
weights_path = './ai_face_detector_model.weights.h5'

if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded existing model weights.")

# Create a callback to save the model's weights during training
checkpoint = ModelCheckpoint(weights_path, save_weights_only=True, save_best_only=True, verbose=1)

# Step 5: Train the model (with reduced batch size to avoid memory overflow)
model.fit(X_train, y_train, epochs=10, batch_size=4, validation_split=0.1,
          callbacks=[ProgressCallback(), checkpoint])

# Step 6: Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")

# Save the trained model for future use
model.save('./ai_face_detector_model.keras')